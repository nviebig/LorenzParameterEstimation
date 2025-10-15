module TrainingClimate

# Draft: minimal, clean, modular training on TIME-SUBSAMPLED statistics (mean only for now).
# - No warmup/window logic.
# - One integration per IC for 'steps' and we sample k time rows per epoch.
# - Optimisers.jl for updates, Enzyme for gradients.
# - Update mask lets you choose which parameters to learn.
#
# Extend later by:
#   * adding more stats in _accumulate_stat!, _finalize_stat!, and _loss_dispatch
#   * swapping the time sampler strategy
# ------------------------------------------------------------------------------

using Enzyme
using Optimisers
using Random
using Printf: @printf, @sprintf

import ..LorenzParameterEstimation
using  ..LorenzParameterEstimation: L63Parameters, integrate

export train_climate, evaluate_statistics, ClimateConfig

# ================================
# Configuration
# ================================

Base.@kwdef struct ClimateConfig{T}
    dt::T            = T(0.01)     # integrator time step
    steps::Int       = 4000        # total integration steps (use full trajectory)
    initial_conditions::Union{Nothing,AbstractMatrix{T}} = nothing  # 3×B matrix of ICs; if nothing, base_u0 is used once
    samples_per_epoch::Int = 256    # number of time rows to sample each epoch (2:steps+1)
    rng::Random.AbstractRNG = Random.default_rng()
end

const DEFAULT_MASK = (σ=true, ρ=true, β=true, x_s=false, y_s=false, z_s=false, θ=false)

# ================================
# Utilities
# ================================

@inline function _assert_ics(U)
    size(U,1) == 3 || throw(ArgumentError("initial_conditions must be a 3×B matrix"))
end

@inline function _fmt_params(s)
    @sprintf("σ=%.6f ρ=%.6f β=%.6f x_s=%.6f y_s=%.6f z_s=%.6f θ=%.6f",
             s.σ[1], s.ρ[1], s.β[1], s.x_s[1], s.y_s[1], s.z_s[1], s.θ[1])
end

# Optimisers-compatible state (NamedTuple of length-1 arrays)
@inline function _to_state(p::L63Parameters{T}) where {T}
    (σ=[p.σ], ρ=[p.ρ], β=[p.β], x_s=[p.x_s], y_s=[p.y_s], z_s=[p.z_s], θ=[p.θ])
end
@inline function _from_state(s)
    L63Parameters(s.σ[1], s.ρ[1], s.β[1], s.x_s[1], s.y_s[1], s.z_s[1], s.θ[1])
end

@inline function _mask_grads(g, mask)
    zero1(x) = zero(eltype(x)).*one.(x)
    (σ   = mask.σ   ? g.σ   : zero1(g.σ),
     ρ   = mask.ρ   ? g.ρ   : zero1(g.ρ),
     β   = mask.β   ? g.β   : zero1(g.β),
     x_s = mask.x_s ? g.x_s : zero1(g.x_s),
     y_s = mask.y_s ? g.y_s : zero1(g.y_s),
     z_s = mask.z_s ? g.z_s : zero1(g.z_s),
     θ   = mask.θ   ? g.θ   : zero1(g.θ))
end

@inline function _gradnorm(g)
    sqrt(g.σ[1]^2 + g.ρ[1]^2 + g.β[1]^2 + g.x_s[1]^2 + g.y_s[1]^2 + g.z_s[1]^2 + g.θ[1]^2)
end

# Enzyme gradient tuple → plain tuple
@inline function _flatten_grads(g)
    g isa Tuple || error("Unexpected Enzyme gradient shape: $(typeof(g))")
    (length(g) == 1 && g[1] isa Tuple) ? g[1] : g
end

# SSE on vectors/matrices
@inline function _sse(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    size(a) == size(b) || throw(ArgumentError("shape mismatch in loss"))
    acc = zero(T)
    @inbounds for i in eachindex(a,b)
        d = a[i] - b[i]; acc += d*d
    end
    acc
end

# ================================
# Statistics (mean only for now)
# ================================

# Accumulate time-subsampled mean for a single trajectory matrix 'U' on selected rows 'I'
# returns a length-3 vector
@inline function _time_mean_subsample(U::AbstractMatrix{T}, I::AbstractVector{Int}) where {T}
    k = length(I)
    invk = one(T)/T(k)
    m = zeros(T, size(U,2))  # components along columns of U (x,y,z)
    @inbounds for j in 1:size(U,2)
        s = zero(T)
        for r in I
            s += U[r, j]
        end
        m[j] = s * invk
    end
    return m
end

# Registry (extensible): accumulate stat into 'acc' (Dict) from one trajectory's sampled rows
function _accumulate_stat!(acc::Dict{Symbol,Any}, name::Symbol, traj_u::AbstractMatrix, I::AbstractVector{Int})
    if name === :mean
        m = _time_mean_subsample(traj_u, I)           # length-3
        acc[:mean] = haskey(acc,:mean) ? (acc[:mean] .+ m) : m
    elseif name === :std
        # Compute std over the sampled rows for each coordinate
        # std(x) = sqrt(mean((x - mean(x))^2))
        m = _time_mean_subsample(traj_u, I)
        v = zeros(eltype(traj_u), size(traj_u,2))
        for j in 1:size(traj_u,2)
            s = zero(eltype(traj_u))
            for r in I
                s += (traj_u[r, j] - m[j])^2
            end
            v[j] = s / length(I)
        end
        stds = sqrt.(v)
        acc[:std] = haskey(acc,:std) ? (acc[:std] .+ stds) : stds
    else
        error("Unsupported statistic $(name). Add it to _accumulate_stat!.")
    end
end

# Finalize across batch of size B
function _finalize_stat!(out::Dict{Symbol,Any}, acc::Dict{Symbol,Any}, name::Symbol, B::Integer)
    if name === :mean
        out[:mean] = acc[:mean] .* (one(eltype(acc[:mean])) / eltype(acc[:mean])(B))
    elseif name === :std
        out[:std] = acc[:std] .* (one(eltype(acc[:std])) / eltype(acc[:std])(B))
    else
        error("Unsupported statistic $(name). Add it to _finalize_stat!.")
    end
end

# Loss dispatch across stats (for now: mean → SSE)
function _loss_dispatch(pred::NamedTuple, target, stats::NTuple)
    T = eltype(pred[1])
    loss = zero(T)
    for s in stats
        if s === :mean
            hasproperty(target, :mean) || throw(ArgumentError("target_stats missing :mean"))
            loss += _sse(getfield(pred,:mean), getfield(target,:mean))
        else
            error("Unsupported stat $(s). Extend _loss_dispatch.")
        end
    end
    loss
end

# ================================
# Forward: stats over batch with time subsampling
# ================================

"""
_stats_over_batch_subsample(params, cfg, base_u0, stats, I)

Integrate each IC once for 'steps', then compute requested stats using only rows in I.
Returns NamedTuple with fields in 'stats' order.
"""
# --- replace _stats_over_batch_subsample with this (no threading, no duplicates)
function _stats_over_batch_subsample(params::L63Parameters{T},
                                     cfg::ClimateConfig{T},
                                     base_u0::AbstractVector{T},
                                     stats::NTuple{N,Symbol},
                                     I::AbstractVector{Int}) where {T,N}

    U = cfg.initial_conditions === nothing ? reshape(T.(base_u0), 3, 1) : T.(cfg.initial_conditions)
    _assert_ics(U)

    tspan = (zero(T), T(cfg.steps) * cfg.dt)
    acc = Dict{Symbol,Any}()

    @inbounds for b in 1:size(U, 2)
        u0 = @view U[:, b]
        sol = integrate(params, u0, tspan, cfg.dt)  # sol.u is (steps+1) × 3
        for s in stats
            _accumulate_stat!(acc, s, sol.u, I)
        end
    end

    outD = Dict{Symbol,Any}()
    for s in stats
        _finalize_stat!(outD, acc, s, size(U,2))
    end
    return NamedTuple{stats}(map(k -> outD[k], stats))
end

# Enzyme-friendly scalar loss (indices I are Const)
function _climate_loss_subsample(σ, ρ, β, x_s, y_s, z_s, θ,
                                 cfg::ClimateConfig{T},
                                 base_u0::AbstractVector{T},
                                 stats::NTuple{N,Symbol},
                                 target_stats,
                                 I::AbstractVector{Int}) where {T,N}
    params = L63Parameters(T(σ),T(ρ),T(β),T(x_s),T(y_s),T(z_s),T(θ))
    pred = _stats_over_batch_subsample(params, cfg, base_u0, stats, I)
    _loss_dispatch(pred, target_stats, stats)
end

# ================================
# Public API
# ================================

"""
    evaluate_statistics(params; cfg=ClimateConfig(dt=0.01,steps=4000),
                        base_u0=[1,1,1], stats=(:mean,), full_trajectory=true, I=nothing)

Compute statistics either on the full trajectory (default) or on provided time indices `I`.
"""
function evaluate_statistics(params::L63Parameters;
                             cfg::ClimateConfig=ClimateConfig(dt=0.01, steps=4000),
                             base_u0::AbstractVector=[1.0,1.0,1.0],
                             stats::Union{Tuple,AbstractVector}=(:mean,),
                             full_trajectory::Bool=true,
                             I=nothing)

    stats_tuple = stats isa Tuple ? stats : tuple(stats...)
    T = promote_type(typeof(params.σ), eltype(base_u0), typeof(cfg.dt))
    cfgT = ClimateConfig{T}(dt=T(cfg.dt), steps=cfg.steps,
                            initial_conditions=(cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
                            samples_per_epoch=cfg.samples_per_epoch, rng=cfg.rng)
    base_u0T = T.(base_u0)

    if full_trajectory
        # use all rows 2:steps+1 (skip t=0 row)
        Iall = collect(2:cfg.steps+1)
        return _stats_over_batch_subsample(L63Parameters(T(params.σ),T(params.ρ),T(params.β),
                                                         T(params.x_s),T(params.y_s),T(params.z_s),T(params.θ)),
                                           cfgT, base_u0T, stats_tuple, Iall)
    else
        I === nothing && error("Provide I when full_trajectory=false")
        return _stats_over_batch_subsample(L63Parameters(T(params.σ),T(params.ρ),T(params.β),
                                                         T(params.x_s),T(params.y_s),T(params.z_s),T(params.θ)),
                                           cfgT, base_u0T, stats_tuple, I)
    end
end

"""
    train_climate(initial_params;
                  target_stats,                # NamedTuple, e.g. (mean = [mx,my,mz],)
                  stats=(:mean,),
                  cfg=ClimateConfig(dt=0.01, steps=4000, samples_per_epoch=256),
                  base_u0=[1,1,1],
                  optimizer=Optimisers.Adam(1e-3),
                  update_mask=DEFAULT_MASK,
                  epochs=200,
                  verbose=true)

Stochastic time-subsampling training:
- Integrate each IC once for 'steps'.
- Per epoch, sample `cfg.samples_per_epoch` time rows (same I for all ICs that epoch).
- Compute chosen stats (currently `:mean`) on those rows and backprop with Enzyme.
- Optimisers.jl applies the masked update.

Returns (params, loss_history, final_statistics).
"""
function train_climate(
    initial_params::L63Parameters;                  # initial guess         
    target_stats,                                   # NamedTuple, e.g. (mean = [mx,my,mz],)     
    stats::Union{Tuple,AbstractVector}=(:mean,),    # which stats to match
    cfg::ClimateConfig=ClimateConfig(               # configuration
        dt=0.01, 
        steps=4000,                                 
        samples_per_epoch=256),

    base_u0::AbstractVector=[1.0,1.0,1.0],           # base IC if cfg.initial_conditions=nothing
    optimizer=Optimisers.Adam(1e-3),                 # Optimisers.jl optimizer
    update_mask=DEFAULT_MASK,                        # which params to update (NamedTuple of Bools)
    epochs::Int=200,                                 # training epochs 
    verbose::Bool=true)                              # verbose output


    stats_tuple = stats isa Tuple ? stats : tuple(stats...)
    # Type promotion
    T = promote_type(typeof(initial_params.σ), eltype(base_u0), typeof(cfg.dt))
    cfgT = ClimateConfig{T}(dt=T(cfg.dt), steps=cfg.steps,
                            initial_conditions=(cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
                            samples_per_epoch=cfg.samples_per_epoch, rng=cfg.rng)
    base_u0T = T.(base_u0)

    # Type-coerced targets (expect fields matching 'stats')
    tgt = target_stats isa NamedTuple ? NamedTuple{keys(target_stats)}(map(x->T.(x), values(target_stats))) :
          error("target_stats must be a NamedTuple, e.g., (mean = T[...],).")

    # Optimiser state
    pstate = _to_state(L63Parameters(T(initial_params.σ),T(initial_params.ρ),T(initial_params.β),
                                     T(initial_params.x_s),T(initial_params.y_s),T(initial_params.z_s),T(initial_params.θ)))
    ost = Optimisers.setup(optimizer, pstate)

    losses = Vector{T}(undef, epochs)

    if verbose
        numICs = (cfgT.initial_conditions === nothing) ? 1 : size(cfgT.initial_conditions,2)
        @printf("[Climate] setup | stats=%s | mask=%s | dt=%.4g steps=%d | ICs=%d | k=%d\n",
                string(stats_tuple), string(update_mask), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch)
        @printf("[Climate] init  | %s\n", _fmt_params(pstate))
    end

    # Training loop
    for epoch in 1:epochs
        # Time subsample indices for this epoch: sample from rows 2..steps+1 (skip t=0)
        k = min(cfgT.samples_per_epoch, cfgT.steps)
        @assert k > 0
        I = Random.sample(cfgT.rng, 2:cfgT.steps+1, k; replace=false)
        sort!(I)  # stable order helps numerics a bit

        # Alias scalars for Enzyme call
        σ, ρ, β = pstate.σ[1], pstate.ρ[1], pstate.β[1]
        x_s, y_s, z_s, θ = pstate.x_s[1], pstate.y_s[1], pstate.θ[1], pstate.θ[1]  # temp alias; fixed below
        # NOTE: Correct θ alias:
        x_s, y_s, z_s, θ = pstate.x_s[1], pstate.y_s[1], pstate.z_s[1], pstate.θ[1]

        # Forward loss
        L = _climate_loss_subsample(σ,ρ,β,x_s,y_s,z_s,θ, cfgT, base_u0T, stats_tuple, tgt, I)
        losses[epoch] = L

        # Backward (mask later)
        gtuple = Enzyme.autodiff(
            Enzyme.Reverse,                 # mode 
            climate_loss_subsample,         # function
            Enzyme.Active(σ),               # arguments, scalars
            Enzyme.Active(ρ),               # arguments, scalars
            Enzyme.Active(β),               # arguments, scalars
            Enzyme.Active(x_s),             # arguments, scalars
            Enzyme.Active(y_s),             # arguments, scalars
            Enzyme.Active(z_s),             # arguments, scalars
            Enzyme.Active(θ),               # arguments, scalars
            Enzyme.Const(cfgT),             # constants (immutable struct)
            Enzyme.Const(base_u0T),         # constants (immutable struct)
            Enzyme.Const(stats_tuple),      # constants (immutable struct)
            Enzyme.Const(tgt),              # constants (immutable struct)
            Enzyme.Const(I)                 # constants (immutable struct)
        )
        g = _flatten_grads(gtuple)
        gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                      x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
        gstate = _mask_grads(gstate_all, update_mask)

        # Optimiser update
        ost, pstate = Optimisers.update!(ost, pstate, gstate)

        if verbose
            @printf("[Climate] epoch %4d | loss = %.6e | g_norm = %.3e | %s\n",
                    epoch, L, _gradnorm(gstate), _fmt_params(pstate))
        end
    end

    final_params = _from_state(pstate)
    # For reporting: evaluate on FULL trajectory (all rows)
    final_stats  = evaluate_statistics(final_params; cfg=cfgT, base_u0=base_u0T, stats=stats_tuple, full_trajectory=true)

    return (params=final_params, loss_history=losses, final_statistics=final_stats)
end

end