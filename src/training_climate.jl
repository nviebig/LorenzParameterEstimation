module TrainingClimate

# Draft: minimal, clean, modular training on TIME-SUBSAMPLED statistics (mean only for now).
# - No warmup/window logic.
# - One integration per IC for 'steps' and we sample k time rows per epoch.
# - Optimisers.jl for updates, Enzyme for gradients.
# - Update mask lets you choose which parameters to learn.
#
# Extend later by:
#   * swapping the time sampler strategy
# ------------------------------------------------------------------------------

using Enzyme
using Optimisers
using StatsBase
using Random
using Printf: @printf, @sprintf

import ..LorenzParameterEstimation
using  ..LorenzParameterEstimation: L63Parameters, integrate


export train_climate, evaluate_statistics, ClimateConfig, train_statistics

# ================================
# Configuration
# ================================

Base.@kwdef struct ClimateConfig{T}
    dt::T            = T(0.01)
    steps::Int       = 4000
    initial_conditions::Union{Nothing,AbstractMatrix{T}} = nothing
    samples_per_epoch::Int = 256
    rng::Random.AbstractRNG = Random.default_rng()
    loss_mode::Symbol = :sse            # NEW: choose :sse, :mse, :rmse, :mae
end
"""
    train_statistics(initial_params;
                    target,
                    stats=(:mean,),
                    cfg=ClimateConfig(dt=0.01, steps=4000, samples_per_epoch=256),
                    base_u0=[1,1,1],
                    optimizer=Optimisers.Adam(1e-3),
                    update_mask=DEFAULT_MASK,
                    epochs=200,
                    verbose=true,
                    early_stopping_patience=20,
                    early_stopping_min_delta=1e-6,
                    gradient_verbose=false,
                    refresh=10,
                    pdf_loss_mode=:kl,
                    bandwidth=0.75,
                    sinkhorn_ε=1e-2,
                    sinkhorn_iters=50)

Unified training for mean, std, and 3D PDF statistics. Selects appropriate loss for each statistic type.
- For mean/std: uses cfg.loss_mode (:sse, :mse, etc.)
- For PDF: uses pdf_loss_mode (:kl, :cross_entropy, :wasserstein)
- target: NamedTuple for mean/std, NamedTuple with :centers and :p3d for PDF
Returns (params, loss_history, final_statistics, stats_history)
"""
function train_statistics(
    initial_params::L63Parameters;
    target,
    stats::Union{Tuple,AbstractVector}=(:mean,),
    cfg::ClimateConfig=ClimateConfig(dt=0.01, steps=4000, samples_per_epoch=256),
    base_u0::AbstractVector=[1.0,1.0,1.0],
    optimizer=Optimisers.Adam(1e-3),
    update_mask=DEFAULT_MASK,
    epochs::Int=200,
    verbose::Bool=true,
    early_stopping_patience::Int=20,
    early_stopping_min_delta::Float64=1e-6,
    gradient_verbose::Bool=false,
    refresh::Int=10,
    pdf_loss_mode::Symbol=:kl,
    bandwidth::Real=0.75,
    sinkhorn_ε::Real=1e-2,
    sinkhorn_iters::Int=50
)
    stats_tuple = stats isa Tuple ? stats : tuple(stats...)
    T = promote_type(typeof(initial_params.σ), eltype(base_u0), typeof(cfg.dt))
    cfgT = ClimateConfig{T}(
        dt = T(cfg.dt),
        steps = cfg.steps,
        initial_conditions = (cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch = cfg.samples_per_epoch,
        rng = cfg.rng,
        loss_mode = cfg.loss_mode
    )
    base_u0T = T.(base_u0)

    # PDF mode detection
    is_pdf = any(s -> s == :pdf3d, stats_tuple)

    # Setup targets
    if is_pdf
        centers = T.(getfield(target, :centers))
        p3d_tgt = T.(getfield(target, :p3d))
        s_p = sum(p3d_tgt); s_p > 0 && (p3d_tgt ./= s_p)
        hT = T(bandwidth)
        sinkεT = T(sinkhorn_ε)
    else
        if !(target isa NamedTuple)
            error("target must be a NamedTuple, e.g., (mean = [mx,my,mz],).")
        end
        tgt = (; (k => T.(v) for (k,v) in pairs(target))... )
        target_vec = Vector{T}(undef, 3 * length(stats_tuple))
        idxt = 1
        for s in stats_tuple
            hasproperty(tgt, s) || error("target must contain field $(s)")
            v = getfield(tgt, s)
            @assert length(v) == 3 "each target stat must be length-3"
            target_vec[idxt:idxt+2] = T.(v)
            idxt += 3
        end
    end

    # Optimiser state
    pstate = _to_state(L63Parameters(T(initial_params.σ),T(initial_params.ρ),T(initial_params.β),
                                     T(initial_params.x_s),T(initial_params.y_s),T(initial_params.z_s),T(initial_params.θ)))
    ost = Optimisers.setup(optimizer, pstate)

    losses = Vector{T}(undef, epochs)
    stats_history = Vector{Any}(undef, epochs)
    best_loss = Inf; best_params = nothing; patience_counter = 0; actual_epochs = 0

    I = stratified_indices(cfgT.rng, cfgT.steps, min(cfgT.samples_per_epoch, cfgT.steps))

    if verbose
        numICs = (cfgT.initial_conditions === nothing) ? 1 : size(cfgT.initial_conditions,2)
        if is_pdf
            @printf("UnifiedTrain | PDF3D loss=%s | dt=%.4g steps=%d | ICs=%d | k=%d | bins=%d^3\n",
                    string(pdf_loss_mode), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch, length(centers))
        else
            @printf("UnifiedTrain | stats=%s | mask=%s | dt=%.4g steps=%d | ICs=%d | k=%d\n",
                    string(stats_tuple), string(update_mask), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch)
        end
        @printf("Init  | %s\n", _fmt_params(pstate))
    end

    for epoch in 1:epochs
        k = min(cfgT.samples_per_epoch, cfgT.steps); @assert k > 0
        if epoch == 1 || (epoch % refresh == 1)
            I = stratified_indices(cfgT.rng, cfgT.steps, k)
        end

        σ,ρ,β = pstate.σ[1], pstate.ρ[1], pstate.β[1]
        x_s,y_s,z_s,θ = pstate.x_s[1], pstate.y_s[1], pstate.z_s[1], pstate.θ[1]
        U_epoch = (cfgT.initial_conditions === nothing) ? reshape(T.(base_u0T),3,1) : T.(cfgT.initial_conditions)
        dt_epoch, steps_epoch = cfgT.dt, cfgT.steps

        if is_pdf
            @noinline function loss_entry_pdf3d(σ, ρ, β, x_s, y_s, z_s, θ,
                                                dt::S, steps::Int,
                                                U::AbstractMatrix{S}, I::AbstractVector{Int},
                                                centers::AbstractVector{S}, h::S,
                                                p3d_t::AbstractVector{S},
                                                mode::Symbol, sinkε::S, sinkiters::Int)::S where {S<:Real}
                _loss_pdf_core_3d(σ, ρ, β, x_s, y_s, z_s, θ,
                                  dt, steps, U, I, centers, h, p3d_t, mode, sinkε, sinkiters)
            end
            L = loss_entry_pdf3d(σ,ρ,β,x_s,y_s,z_s,θ,
                                 dt_epoch, steps_epoch, U_epoch, I,
                                 centers, hT, p3d_tgt,
                                 pdf_loss_mode, sinkεT, sinkhorn_iters)
            losses[epoch] = L
            gtuple = Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                loss_entry_pdf3d,
                Enzyme.Active(σ), Enzyme.Active(ρ), Enzyme.Active(β),
                Enzyme.Active(x_s), Enzyme.Active(y_s), Enzyme.Active(z_s), Enzyme.Active(θ),
                Enzyme.Const(dt_epoch), Enzyme.Const(steps_epoch),
                Enzyme.Const(U_epoch), Enzyme.Const(I),
                Enzyme.Const(centers), Enzyme.Const(hT),
                Enzyme.Const(p3d_tgt),
                Enzyme.Const(pdf_loss_mode), Enzyme.Const(sinkεT), Enzyme.Const(sinkhorn_iters),
            )
            g = _flatten_grads(gtuple)
            gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                          x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
            gstate = _mask_grads(gstate_all, update_mask)
            gradient_verbose && @show gstate
            ost, pstate = Optimisers.update!(ost, pstate, gstate)
            verbose && @printf("Epoch %4d | pdf3d-loss = %.6e | g_norm = %.3e\n", epoch, L, _gradnorm(gstate))
        else
            @noinline function loss_entry(σ, ρ, β, x_s, y_s, z_s, θ,
                                          dt::S, steps::Int,
                                          U::AbstractMatrix{S},
                                          target::AbstractVector{S},
                                          I::AbstractVector{Int},
                                          stats_nt::NTuple{M,Symbol},
                                          loss_mode::Symbol)::S where {S<:Real, M}
                _loss_subsample_core(σ, ρ, β, x_s, y_s, z_s, θ,
                                    dt, steps, U, target, I, stats_nt, loss_mode)
            end
            loss_mode_sym = cfgT.loss_mode
            L = loss_entry(σ, ρ, β, x_s, y_s, z_s, θ,
                        dt_epoch, steps_epoch, U_epoch, target_vec, I, stats_tuple, loss_mode_sym)
            losses[epoch] = L
            gtuple = Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                loss_entry,
                Enzyme.Active(σ), Enzyme.Active(ρ), Enzyme.Active(β),
                Enzyme.Active(x_s), Enzyme.Active(y_s), Enzyme.Active(z_s), Enzyme.Active(θ),
                Enzyme.Const(dt_epoch), Enzyme.Const(steps_epoch),
                Enzyme.Const(U_epoch), Enzyme.Const(target_vec), Enzyme.Const(I),
                Enzyme.Const(stats_tuple), Enzyme.Const(loss_mode_sym),
            )
            g = _flatten_grads(gtuple)
            gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                          x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
            gstate = _mask_grads(gstate_all, update_mask)
            if gradient_verbose
                println("--- Gradient for epoch ", epoch, " ---")
                println("gstate (parameter gradients):")
                for (k, v) in pairs(gstate)
                    println("  ", k, ": ", v)
                end
                println("-----------------------------")
            end
            ost, pstate = Optimisers.update!(ost, pstate, gstate)
            verbose && @printf("Epoch %4d | loss = %.6e | g_norm = %.3e | %s\n",
                    epoch, L, _gradnorm(gstate), _fmt_params(pstate))
        end

        # --- record statistics for this epoch (full trajectory) ---
        cur_params = _from_state(pstate)
        if is_pdf
            stats_history[epoch] = (; pdf3d = _compute_pdf3d_statistic(
                cur_params, cfgT, base_u0T, centers, hT, I))
        else
            stats_history[epoch] = evaluate_statistics(
                cur_params; cfg=cfgT, base_u0=base_u0T,
                stats=stats_tuple, full_trajectory=true
            )
        end

        # Early stopping logic
        if L < best_loss - early_stopping_min_delta
            best_loss = L; best_params = deepcopy(cur_params); patience_counter = 0
        else
            patience_counter += 1
        end
        if patience_counter >= early_stopping_patience
            verbose && @printf("Early stopping at epoch %d | best_loss = %.6e\n", epoch, best_loss)
            break
        end
        actual_epochs += 1
    end

    # Truncate arrays to actual number of completed epochs
    losses = losses[1:actual_epochs]
    stats_history = stats_history[1:actual_epochs]
    final_params = best_params === nothing ? _from_state(pstate) : best_params
    if is_pdf
        Iall = collect(2:cfgT.steps+1)
        final_stats = (; pdf3d = _compute_pdf3d_statistic(
            final_params, cfgT, base_u0T, centers, hT, Iall))
    else
        final_stats  = evaluate_statistics(final_params; cfg=cfgT, base_u0=base_u0T, stats=stats_tuple, full_trajectory=true)
    end
    return (params=final_params, loss_history=losses, final_statistics=final_stats, stats_history=stats_history)
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

@inline _mse(a,b)  = _sse(a,b) / length(a)
@inline _rmse(a,b) = sqrt(_mse(a,b))
@inline _mae(a,b)  = sum(abs.(a .- b)) / length(a)

# central dispatcher used everywhere for stat-vs-stat loss
@inline function _stat_loss(a, b, mode::Symbol)
    if mode === :sse
        return _sse(a,b)
    elseif mode === :mse
        return _mse(a,b)
    elseif mode === :rmse
        return _rmse(a,b)
    elseif mode === :mae
        return _mae(a,b)
    else
        error("Unknown loss mode: $mode. Use one of :sse, :mse, :rmse, :mae")
    end
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

# ================================
# Smarter time sampling
# ================================

"""
    stratified_indices(rng, steps::Int, k::Int)

Return `k` time indices roughly evenly spaced across the trajectory
range `2:steps+1`, one sample per segment. Guarantees broad coverage
and sorted order.
"""
function stratified_indices(rng::AbstractRNG, steps::Int, k::Int)
    k = min(k, steps)
    bins = range(2, steps + 1; length = k + 1)
    I = Vector{Int}(undef, k)
    @inbounds for j in 1:k
        lo = ceil(Int, bins[j])
        hi = floor(Int, bins[j + 1] - eps())
        if lo > hi
            mid = clamp(round(Int, (bins[j] + bins[j + 1]) / 2), 2, steps + 1)
            lo = hi = mid
        end
        I[j] = rand(rng, lo:hi)
    end
    sort!(I)
    return I
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


# In _loss_subsample, call the in-place stepper and DO NOT assign:
#    was:   u = rk4_step(u, p, cfg.dt)
#    now:   rk4_step!(u, p, cfg.dt)

function _loss_subsample(σ, ρ, β, x_s, y_s, z_s, θ,
                              cfg::ClimateConfig{T},
                              base_u0::AbstractVector{T},
                              target_mean::AbstractVector{T},
                              I::AbstractVector{Int}) where {T}

    p = L63Parameters(T(σ),T(ρ),T(β),T(x_s),T(y_s),T(z_s),T(θ))

    U = cfg.initial_conditions === nothing ? reshape(T.(base_u0), 3, 1) : T.(cfg.initial_conditions)
    B = size(U,2)
    k = length(I); invk = one(T)/T(k)

    acc = zeros(T,3)

    @inbounds for b in 1:B
        u   = copy(@view U[:,b])      # current state
        k1  = similar(u); k2 = similar(u); k3 = similar(u); k4 = similar(u)
        tmp = similar(u)

        trow = 1
        idxp = 1
        nextI = I[idxp]

        for step in 1:cfg.steps
            rk4_step!(u, p, cfg.dt, k1, k2, k3, k4, tmp) 
            trow += 1
            if trow == nextI
                acc[1] += u[1]*invk
                acc[2] += u[2]*invk
                acc[3] += u[3]*invk
                idxp += 1
                if idxp > k
                    # (optional) break
                    # break
                else
                    nextI = I[idxp]
                end
            end
        end
    end

    mean_batch = acc .* (one(T)/T(B))

    L = _stat_loss(mean_batch, target_mean, cfg.loss_mode)
    return L::T
end



# === union-free loss core (no cfg capture, no Union fields) ===
function _loss_subsample_core(σ, ρ, β, x_s, y_s, z_s, θ,
                              dt::S, steps::Int,
                              U::AbstractMatrix{S},           # 3×B
                              target::AbstractVector{S},      # flattened target (3 * length(stats))
                              I::AbstractVector{Int},
                              stats::NTuple{M,Symbol},
                              loss_mode::Symbol)::S where {S<:Real, M}

    p = L63Parameters(S(σ),S(ρ),S(β),S(x_s),S(y_s),S(z_s),S(θ))
    B = size(U,2)
    k = length(I); invk = one(S)/S(k)

    acc_mean = zeros(S,3)
    acc_std  = zeros(S,3)   # accumulate per-trajectory stds only if requested
    compute_mean = any(s -> s === :mean, stats)
    compute_std  = any(s -> s === :std, stats)

    @inbounds for b in 1:B
        u   = copy(@view U[:,b])
        k1  = similar(u); k2 = similar(u); k3 = similar(u); k4 = similar(u)
        tmp = similar(u)
        trow = 1; idxp = 1; nextI = I[idxp]

        # per-trajectory accumulators for mean/std
        sum_traj = zeros(S,3)
        sumsq_traj = zeros(S,3)

        for _ in 1:steps
            rk4_step!(u, p, dt, k1, k2, k3, k4, tmp)
            trow += 1
            if trow == nextI
                sum_traj[1] += u[1]; sum_traj[2] += u[2]; sum_traj[3] += u[3]
                sumsq_traj[1] += u[1]*u[1]; sumsq_traj[2] += u[2]*u[2]; sumsq_traj[3] += u[3]*u[3]
                idxp += 1
                idxp <= k && (nextI = I[idxp])
            end
        end

        if compute_mean
            mean_traj = sum_traj .* (invk)
            acc_mean .+= mean_traj
        end

        if compute_std
            # variance = E[x^2] - (E[x])^2
            mean_traj = sum_traj .* (invk)
            ex2 = sumsq_traj .* (invk)
            var_traj = ex2 .- (mean_traj .* mean_traj)
            var_traj .= clamp.(var_traj, zero(S), Inf)   # guard numerical negatives
            std_traj = sqrt.(var_traj)
            acc_std .+= std_traj
        end
    end

    mean_batch = acc_mean .* (one(S)/S(B))
    std_batch  = acc_std  .* (one(S)/S(B))

    # build flattened prediction vector in same order as stats
    pred = Vector{S}(undef, 3 * length(stats))
    idx = 1
    for s in stats
        if s === :mean
            pred[idx:idx+2] = mean_batch
        elseif s === :std
            pred[idx:idx+2] = std_batch
        else
            error("Unsupported statistic $(s) in loss core. Add it to _loss_subsample_core.")
        end
        idx += 3
    end

    return _stat_loss(pred, target, loss_mode)::S
end


@inline function lorenz_rhs!(du, u, p::L63Parameters{T}) where {T}
    # classic L63 with (x_s,y_s,z_s) shifts; drop θ if unused in dynamics
    x = u[1] - p.x_s
    y = u[2] - p.y_s
    z = u[3] - p.z_s
    du[1] = p.σ * (y - x)
    du[2] = x * (p.ρ - z) - y
    du[3] = x*y - p.β * z
    return nothing
end

# Replace your rk4_step with an in-place version that returns nothing
# preallocate-once, fully in-place stepper
@inline function rk4_step!(u::AbstractVector{T}, p::L63Parameters{T}, dt::T,
                           k1::AbstractVector{T}, k2::AbstractVector{T},
                           k3::AbstractVector{T}, k4::AbstractVector{T},
                           tmp::AbstractVector{T}) where {T}
    lorenz_rhs!(k1, u, p)

    @inbounds for i in 1:3; tmp[i] = u[i] + (dt/2)*k1[i]; end
    lorenz_rhs!(k2, tmp, p)

    @inbounds for i in 1:3; tmp[i] = u[i] + (dt/2)*k2[i]; end
    lorenz_rhs!(k3, tmp, p)

    @inbounds for i in 1:3; tmp[i] = u[i] + dt*k3[i]; end
    lorenz_rhs!(k4, tmp, p)

    @inbounds for i in 1:3
        u[i] = u[i] + (dt/6)*(k1[i] + 2k2[i] + 2k3[i] + k4[i])
    end
    return nothing
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
    cfgT = ClimateConfig{T}(
        dt = T(cfg.dt),
        steps = cfg.steps,
        initial_conditions = (cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch = cfg.samples_per_epoch,
        rng = cfg.rng,
        loss_mode = cfg.loss_mode,          # <-- keep the chosen loss here
    )
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
    verbose::Bool=true,                              # verbose output
    early_stopping_patience::Int=20,                 # early stopping patience (epochs)

    early_stopping_min_delta::Float64=1e-6,          # min improvement to reset patience
    gradient_verbose::Bool=false,                     # print gradients if true
    refresh::Int=10     
    )

    stats_tuple = stats isa Tuple ? stats : tuple(stats...)
    # Type promotion
    T = promote_type(typeof(initial_params.σ), eltype(base_u0), typeof(cfg.dt))
    cfgT = ClimateConfig{T}(
        dt = T(cfg.dt),
        steps = cfg.steps,
        initial_conditions = (cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch = cfg.samples_per_epoch,
        rng = cfg.rng,
        loss_mode = cfg.loss_mode
    )
    base_u0T = T.(base_u0)

    # Type-coerced targets (expect fields matching 'stats')
    if !(target_stats isa NamedTuple)
        error("target_stats must be a NamedTuple, e.g., (mean = [mx,my,mz],).")
    end
    # coerce each field to T and build a NamedTuple with same keys
    tgt = (; (k => T.(v) for (k,v) in pairs(target_stats))... )

    # Build flattened target vector in the same order as stats_tuple
    target_vec = Vector{T}(undef, 3 * length(stats_tuple))
    idxt = 1
    for s in stats_tuple
        hasproperty(tgt, s) || error("target_stats must contain field $(s)")
        v = getfield(tgt, s)
        @assert length(v) == 3 "each target stat must be length-3"
        target_vec[idxt:idxt+2] = T.(v)
        idxt += 3
    end

    # Optimiser state
    pstate = _to_state(L63Parameters(T(initial_params.σ),T(initial_params.ρ),T(initial_params.β),
                                     T(initial_params.x_s),T(initial_params.y_s),T(initial_params.z_s),T(initial_params.θ)))
    ost = Optimisers.setup(optimizer, pstate)

    losses = Vector{T}(undef, epochs)
    stats_history = Vector{NamedTuple}(undef, epochs)  # <— per-epoch statistics

    if verbose
        numICs = (cfgT.initial_conditions === nothing) ? 1 : size(cfgT.initial_conditions,2)
        @printf("Setup | stats=%s | mask=%s | dt=%.4g steps=%d | ICs=%d | k=%d\n",
                string(stats_tuple), string(update_mask), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch)
        @printf("Init  | %s\n", _fmt_params(pstate))
    end

    # Early stopping state
    best_loss = Inf
    best_params = nothing
    patience_counter = 0
    actual_epochs = 0

    # Initial stratified indices
    I = stratified_indices(cfgT.rng, cfgT.steps, min(cfgT.samples_per_epoch, cfgT.steps))

    # Training loop
    for epoch in 1:epochs
        # sample time indices (as you already do)
        k = min(cfgT.samples_per_epoch, cfgT.steps)
        @assert k > 0

        # Replace with stratified sampling to ensure better coverage
        if epoch == 1
            I = stratified_indices(cfgT.rng, cfgT.steps, k)
        elseif epoch % refresh == 1
            I = stratified_indices(cfgT.rng, cfgT.steps, k)
        end

        # current scalar params
        σ, ρ, β = pstate.σ[1], pstate.ρ[1], pstate.β[1]
        x_s, y_s, z_s, θ = pstate.x_s[1], pstate.y_s[1], pstate.z_s[1], pstate.θ[1]

        # concretize ICs to avoid Union in closure
        U_epoch = (cfgT.initial_conditions === nothing) ?
                reshape(T.(base_u0T), 3, 1) :
                T.(cfgT.initial_conditions)

        # Non-capturing loss entry: ALL context passed as arguments
        @noinline function loss_entry(σ, ρ, β, x_s, y_s, z_s, θ,
                                      dt::S, steps::Int,
                                      U::AbstractMatrix{S},
                                      target::AbstractVector{S},
                                      I::AbstractVector{Int},
                                      stats_nt::NTuple{M,Symbol},
                                      loss_mode::Symbol)::S where {S<:Real, M}
            _loss_subsample_core(σ, ρ, β, x_s, y_s, z_s, θ,
                                dt, steps, U, target, I, stats_nt, loss_mode)
        end

        # ---- prepare concrete context ----
        U_epoch = (cfgT.initial_conditions === nothing) ? reshape(T.(base_u0T), 3, 1) : T.(cfgT.initial_conditions)
        dt_epoch    = cfgT.dt
        steps_epoch = cfgT.steps
        loss_mode_sym = cfgT.loss_mode

        # ---- forward ----
        L = loss_entry(σ, ρ, β, x_s, y_s, z_s, θ,
                    dt_epoch, steps_epoch, U_epoch, target_vec, I, stats_tuple, loss_mode_sym)
        losses[epoch] = L

        # ---- reverse (note the extra Const args) ----
        # Use runtime-activity mode so Enzyme can handle any runtime-allocated constant memory
        # (avoids "Constant memory is stored (or returned) to a differentiable variable" errors)
        gtuple = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            loss_entry,
            Enzyme.Active(σ), Enzyme.Active(ρ), Enzyme.Active(β),
            Enzyme.Active(x_s), Enzyme.Active(y_s), Enzyme.Active(z_s), Enzyme.Active(θ),
            Enzyme.Const(dt_epoch), Enzyme.Const(steps_epoch),
            Enzyme.Const(U_epoch), Enzyme.Const(target_vec), Enzyme.Const(I),
            Enzyme.Const(stats_tuple), Enzyme.Const(loss_mode_sym),
        )

        g = _flatten_grads(gtuple)
        gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                    x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
        gstate = _mask_grads(gstate_all, update_mask)


        # Print gradients for debugging if requested
        if gradient_verbose
            println("--- Gradient for epoch ", epoch, " ---")
            println("gstate (parameter gradients):")
            for (k, v) in pairs(gstate)
                println("  ", k, ": ", v)
            end
            println("-----------------------------")
        end

        # optimiser step
        ost, pstate = Optimisers.update!(ost, pstate, gstate)

        # --- record statistics for this epoch (full trajectory) ---
        cur_params = _from_state(pstate)
        stats_history[epoch] = evaluate_statistics(
            cur_params; cfg=cfgT, base_u0=base_u0T,
            stats=stats_tuple, full_trajectory=true
        )

        # Early stopping logic
        if L < best_loss - early_stopping_min_delta
            println("[EarlyStopping] best_loss updated: ", best_loss, " → ", L, " at epoch ", epoch)
            best_loss = L
            best_params = deepcopy(cur_params)
            patience_counter = 0
        else
            patience_counter += 1
        end

        if verbose
            @printf("Epoch %4d | loss = %.6e | g_norm = %.3e | %s\n",
                    epoch, L, _gradnorm(gstate), _fmt_params(pstate))
        end

        if patience_counter >= early_stopping_patience
            if verbose
                @printf("Early stopping at epoch %d | best_loss = %.6e\n", epoch, best_loss)
            end
            break
        end
        actual_epochs += 1
    end

    # Truncate arrays to actual number of completed epochs
    losses = losses[1:actual_epochs]
    stats_history = stats_history[1:actual_epochs]

    # Use best_params if early stopping triggered, else last
    final_params = best_params === nothing ? _from_state(pstate) : best_params
    # For reporting: evaluate on FULL trajectory (all rows)
    final_stats  = evaluate_statistics(final_params; cfg=cfgT, base_u0=base_u0T, stats=stats_tuple, full_trajectory=true)

    return (params=final_params, loss_history=losses, final_statistics=final_stats, stats_history=stats_history)
end

# ================================
# 3D histogram and divergence losses
# ================================

# --- 3D soft histogram (Gaussian kernel) -------------------------------------
@inline function soft_hist_3d(xs::AbstractVector{T}, ys::AbstractVector{T}, zs::AbstractVector{T},
                             centers::AbstractVector{T}, h::T, out::AbstractVector{T}) where {T}
    m = length(centers)
    @assert length(out) == m*m*m
    fill!(out, zero(T))
    inv2h2 = one(T) / (2*h*h)
    ns = length(xs)
    # loop over samples and accumulate into flattened index (i,j,k) -> idx = (i-1)*m*m + (j-1)*m + k
    @inbounds for s = 1:ns
        sx = xs[s]; sy = ys[s]; sz = zs[s]
        for i in 1:m
            dx2 = (sx - centers[i])^2
            for j in 1:m
                dy2 = (sy - centers[j])^2
                for k in 1:m
                    dz2 = (sz - centers[k])^2
                    idx = (i-1)*m*m + (j-1)*m + k
                    out[idx] += exp( - (dx2 + dy2 + dz2) * inv2h2 )
                end
            end
        end
    end
    s = sum(out)
    if s > eps(T)
        @inbounds out ./= s
    else
        @inbounds fill!(out, one(T)/length(out))
    end
    return nothing
end

# --- vector versions of CE / KL for arbitrary-length vectors -----------------
@inline function _cross_entropy_vec(p::AbstractVector{T}, q::AbstractVector{T}) where {T}
    ϵ = T(1e-12); acc = zero(T)
    @inbounds for i in eachindex(p,q)
        acc += -p[i] * log(q[i] + ϵ)
    end
    acc
end

@inline function _kl_divergence_vec(p::AbstractVector{T}, q::AbstractVector{T}) where {T}
    ϵ = T(1e-12); acc = zero(T)
    @inbounds for i in eachindex(p,q)
        pi = max(p[i], ϵ); qi = max(q[i], ϵ)
        acc += pi * (log(pi) - log(qi))
    end
    acc
end

# --- full (flattened) Sinkhorn Wasserstein (ε-regularized) --------------------
# WARNING: cost matrix size is n×n where n = m^3. Use only for small m.
function _sinkhorn_wasserstein_nd(p::AbstractVector{T}, q::AbstractVector{T},
                                   centers::AbstractVector{T}; # centers is 1D centers for each axis (shared)
                                   ε::T=T(1e-2), n_iter::Int=50)::T where {T}
    m = length(centers)
    n = m*m*m
    @assert length(p) == n == length(q)
    # build 3D positions array flattened
    xs = Vector{T}(undef, n); ys = similar(xs); zs = similar(xs)
    idx = 1
    @inbounds for i in 1:m
        for j in 1:m
            for k in 1:m
                xs[idx] = centers[i]; ys[idx] = centers[j]; zs[idx] = centers[k]
                idx += 1
            end
        end
    end
    # cost matrix C_{ab} = ||pos_a - pos_b||^2
    C = Matrix{T}(undef, n, n)
    @inbounds for a in 1:n
        xa = xs[a]; ya = ys[a]; za = zs[a]
        for b in 1:n
            d2 = (xa - xs[b])^2 + (ya - ys[b])^2 + (za - zs[b])^2
            C[a,b] = d2
        end
    end
    K = @. exp( - C / ε )
    u = fill(one(T)/n, n)
    v = fill(one(T)/n, n)
    tiny = T(1e-12)
    for _ in 1:n_iter
        Kv = K * v
        @inbounds for i in 1:n
            u[i] = p[i] / max(Kv[i], tiny)
        end
        Ktu = K' * u
        @inbounds for j in 1:n
            v[j] = q[j] / max(Ktu[j], tiny)
        end
    end
    Kv = (K .* C) * v
    acc = zero(T)
    @inbounds for i in 1:n
        acc += u[i] * Kv[i]
    end
    return acc
end

# --- 3D pdf loss core (Enzyme-friendly) --------------------------------------
function _loss_pdf_core_3d(σ, ρ, β, x_s, y_s, z_s, θ,
                           dt::S, steps::Int,
                           U::AbstractMatrix{S},              # 3×B ICs
                           I::AbstractVector{Int},            # subsampled time indices
                           centers::AbstractVector{S},        # 1D bin centers (shared for x,y,z)
                           h::S,                              # Gaussian bandwidth
                           p3d_tgt::AbstractVector{S},        # target flattened 3D pdf (length m^3)
                           loss_mode::Symbol,
                           sink_ε::S, sink_iters::Int)::S where {S<:Real}

    p = L63Parameters(S(σ),S(ρ),S(β),S(x_s),S(y_s),S(z_s),S(θ))
    B = size(U,2)
    k = length(I)
    nx = k*B
    xs = Vector{S}(undef, nx); ys = similar(xs); zs = similar(xs)
    idx = 1

    @inbounds for b in 1:B
        u   = copy(@view U[:,b])
        k1  = similar(u); k2 = similar(u); k3 = similar(u); k4 = similar(u)
        tmp = similar(u)
        trow = 1; idxp = 1; nextI = I[idxp]
        for _ in 1:steps
            rk4_step!(u, p, dt, k1, k2, k3, k4, tmp)
            trow += 1
            if trow == nextI
                xs[idx] = u[1]; ys[idx] = u[2]; zs[idx] = u[3]
                idx += 1
                idxp += 1
                idxp <= k && (nextI = I[idxp])
            end
        end
    end

    m = length(centers)
    n = m*m*m
    p3d = Vector{S}(undef, n)
    soft_hist_3d(xs, ys, zs, centers, h, p3d)

    if loss_mode === :cross_entropy
        return _cross_entropy_vec(p3d_tgt, p3d)
    elseif loss_mode === :kl
        return _kl_divergence_vec(p3d_tgt, p3d)
    elseif loss_mode === :wasserstein
        return _sinkhorn_wasserstein_nd(p3d_tgt, p3d, centers; ε=sink_ε, n_iter=sink_iters)
    else
        error("Unknown pdf loss mode: $loss_mode")
    end
end

function _compute_pdf3d_statistic(params::L63Parameters{S},
                                  cfg::ClimateConfig{S},
                                  base_u0::AbstractVector{S},
                                  centers::AbstractVector{S},
                                  h::S,
                                  I::AbstractVector{Int}) where {S<:Real}
    U = cfg.initial_conditions === nothing ? reshape(S.(base_u0), 3, 1) : S.(cfg.initial_conditions)
    _assert_ics(U)

    p = L63Parameters(S(params.σ), S(params.ρ), S(params.β),
                      S(params.x_s), S(params.y_s), S(params.z_s), S(params.θ))

    B = size(U,2)
    k = length(I)
    @assert k > 0 "PDF statistic requires at least one sampled time index"

    nx = k * B
    xs = Vector{S}(undef, nx)
    ys = similar(xs)
    zs = similar(xs)
    idx = 1

    @inbounds for b in 1:B
        u   = copy(@view U[:,b])
        k1  = similar(u); k2 = similar(u); k3 = similar(u); k4 = similar(u)
        tmp = similar(u)
        trow = 1; idxp = 1; nextI = I[idxp]
        for _ in 1:cfg.steps
            rk4_step!(u, p, cfg.dt, k1, k2, k3, k4, tmp)
            trow += 1
            if trow == nextI
                xs[idx] = u[1]; ys[idx] = u[2]; zs[idx] = u[3]
                idx += 1
                idxp += 1
                idxp <= k && (nextI = I[idxp])
            end
        end
    end

    m = length(centers)
    p3d = Vector{S}(undef, m*m*m)
    soft_hist_3d(xs, ys, zs, centers, h, p3d)
    sum_p = sum(p3d)
    sum_p > 0 && (p3d ./= sum_p)
    return (; centers=centers, p3d=p3d)
end

# ================================
# Public API for PDF training
# ================================

function train_pdf_3d(
    initial_params::L63Parameters;
    target_pdf::NamedTuple,                      # (centers=..., p3d=...) where p3d is flattened vector length m^3
    cfg::ClimateConfig=ClimateConfig(dt=0.01, steps=4000, samples_per_epoch=512),
    base_u0::AbstractVector=[1.0,1.0,1.0],
    optimizer=Optimisers.Adam(1e-3),
    update_mask=DEFAULT_MASK,
    loss_mode::Symbol=:kl,                       # :kl | :cross_entropy | :wasserstein
    bandwidth::Real=0.75,
    sinkhorn_ε::Real=1e-2, sinkhorn_iters::Int=50,
    refresh::Int=10,
    epochs::Int=200, verbose::Bool=true,
    early_stopping_patience::Int=20, early_stopping_min_delta::Float64=1e-6,
    gradient_verbose::Bool=false
)
    T = promote_type(typeof(initial_params.σ), eltype(base_u0), typeof(cfg.dt))
    cfgT = ClimateConfig{T}(
        dt=T(cfg.dt), steps=cfg.steps,
        initial_conditions=(cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch=cfg.samples_per_epoch, rng=cfg.rng, loss_mode=cfg.loss_mode
    )
    base_u0T = T.(base_u0)

    centers = T.(getfield(target_pdf, :centers)) # 1D centers
    p3d_tgt = T.(getfield(target_pdf, :p3d))     # flattened target
    s_p = sum(p3d_tgt); s_p > 0 && (p3d_tgt ./= s_p)

    pstate = _to_state(L63Parameters(T(initial_params.σ),T(initial_params.ρ),T(initial_params.β),
                                     T(initial_params.x_s),T(initial_params.y_s),T(initial_params.z_s),T(initial_params.θ)))
    ost = Optimisers.setup(optimizer, pstate)

    losses = Vector{T}(undef, epochs)
    best_loss = Inf; best_params = nothing
    patience = 0; actual_epochs = 0

    I = stratified_indices(cfgT.rng, cfgT.steps, min(cfgT.samples_per_epoch, cfgT.steps))

    @noinline function loss_entry_pdf3d(σ, ρ, β, x_s, y_s, z_s, θ,
                                        dt::S, steps::Int,
                                        U::AbstractMatrix{S}, I::AbstractVector{Int},
                                        centers::AbstractVector{S}, h::S,
                                        p3d_t::AbstractVector{S},
                                        mode::Symbol, sinkε::S, sinkiters::Int)::S where {S<:Real}
        _loss_pdf_core_3d(σ, ρ, β, x_s, y_s, z_s, θ,
                          dt, steps, U, I, centers, h, p3d_t, mode, sinkε, sinkiters)
    end

    if verbose
        numICs = (cfgT.initial_conditions === nothing) ? 1 : size(cfgT.initial_conditions,2)
        @printf("PDF-Train-3D | loss=%s | dt=%.4g steps=%d | ICs=%d | k=%d | bins=%d^3\n",
                string(loss_mode), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch, length(centers))
    end

    for epoch in 1:epochs
        k = min(cfgT.samples_per_epoch, cfgT.steps); @assert k > 0
        if epoch == 1 || (epoch % refresh == 1)
            I = stratified_indices(cfgT.rng, cfgT.steps, k)
        end

        σ,ρ,β = pstate.σ[1], pstate.ρ[1], pstate.β[1]
        x_s,y_s,z_s,θ = pstate.x_s[1], pstate.y_s[1], pstate.z_s[1], pstate.θ[1]

        U_epoch = (cfgT.initial_conditions === nothing) ? reshape(T.(base_u0T),3,1) : T.(cfgT.initial_conditions)
        dt_epoch, steps_epoch = cfgT.dt, cfgT.steps
        hT = T(bandwidth); sinkεT = T(sinkhorn_ε)

        # forward
        L = loss_entry_pdf3d(σ,ρ,β,x_s,y_s,z_s,θ,
                             dt_epoch, steps_epoch, U_epoch, I,
                             centers, hT, p3d_tgt,
                             loss_mode, sinkεT, sinkhorn_iters)
        losses[epoch] = L

        # reverse via Enzyme
        gtuple = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            loss_entry_pdf3d,
            Enzyme.Active(σ), Enzyme.Active(ρ), Enzyme.Active(β),
            Enzyme.Active(x_s), Enzyme.Active(y_s), Enzyme.Active(z_s), Enzyme.Active(θ),
            Enzyme.Const(dt_epoch), Enzyme.Const(steps_epoch),
            Enzyme.Const(U_epoch), Enzyme.Const(I),
            Enzyme.Const(centers), Enzyme.Const(hT),
            Enzyme.Const(p3d_tgt),
            Enzyme.Const(loss_mode), Enzyme.Const(sinkεT), Enzyme.Const(sinkhorn_iters),
        )

        g = _flatten_grads(gtuple)
        gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                      x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
        gstate = _mask_grads(gstate_all, update_mask)

        gradient_verbose && @show gstate
        ost, pstate = Optimisers.update!(ost, pstate, gstate)

        if L < best_loss - T(early_stopping_min_delta)
            best_loss = L; best_params = _from_state(pstate); patience = 0
        else
            patience += 1
        end
        verbose && @printf("Epoch %4d | pdf3d-loss = %.6e | g_norm = %.3e\n", epoch, L, _gradnorm(gstate))
        if patience >= early_stopping_patience; break; end
        actual_epochs += 1
    end

    losses = losses[1:max(actual_epochs,1)]
    final_params = best_params === nothing ? _from_state(pstate) : best_params
    return (params=final_params, loss_history=losses)
end

end # module
