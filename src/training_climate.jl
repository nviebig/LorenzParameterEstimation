module TrainingClimate


using Enzyme
using Optimisers
using StatsBase
using Random
using StatsBase
using Printf: @printf, @sprintf

import ..LorenzParameterEstimation
using  ..LorenzParameterEstimation: L63Parameters, integrate


export train_climate, evaluate_statistics, ClimateConfig, train_statistics, soft_hist_3d, make_pdf_config, hard_hist_3d, soft_hist_3d_local

# ================================
# Configuration
# ================================

Base.@kwdef struct PdfConfig{T}
    centers::Vector{T}
    bandwidth::T
    loss_mode::Symbol
    sinkhorn_ε::T
    sinkhorn_iters::Int
end

# helper to build a PdfConfig from nbins/halfwidth (or explicit centers)
function make_pdf_config(::Type{T};
        nbins::Int=16,
        range_halfwidth::Real=30.0,
        centers::Union{Nothing,AbstractVector}=nothing,
        bandwidth::Union{Nothing,Real}=nothing,
        loss_mode::Symbol=:kl,
        sinkhorn_ε::Real=1e-2,
        sinkhorn_iters::Int=50) where {T}

    C = centers === nothing ?
        collect(range(T(-range_halfwidth), T(range_halfwidth); length=nbins)) :
        T.(centers)

    Δ = length(C) >= 2 ? (C[end]-C[1])/(length(C)-1) : T(1)
    h = bandwidth === nothing ? T(0.5*Δ) : T(bandwidth)

    return PdfConfig{T}(C, h, loss_mode, T(sinkhorn_ε), sinkhorn_iters)
end
 
# Configuration for training climate statistics

Base.@kwdef struct ClimateConfig{T}
    dt::T            = T(0.01)
    steps::Int       = 4000
    initial_conditions::Union{Nothing,AbstractMatrix{T}} = nothing
    samples_per_epoch::Int = 256
    rng::Random.AbstractRNG = Random.default_rng()
    loss_mode::Symbol = :sse            # NEW: choose :sse, :mse, :rmse, :mae
    pdf::PdfConfig{T} = make_pdf_config(T)   # <— NEW
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
    rel_delta::Float64 = 1e-3, 
    gradient_verbose::Bool=false,
    refresh::Int=10,
    gradient_clip_norm::Float64 = 1.0,
)
    # Type Promotion and Configuration 
    #(Promotes types for all relevant variables to ensure consistency, Creates a type-stable ClimateConfig object for the training run.)
    
    stats_tuple = stats isa Tuple ? stats : tuple(stats...)
    T = promote_type(typeof(initial_params.σ), eltype(base_u0), typeof(cfg.dt))

    pdfT = PdfConfig{T}(
        T.(cfg.pdf.centers),
        T(cfg.pdf.bandwidth),
        cfg.pdf.loss_mode,
        T(cfg.pdf.sinkhorn_ε),
        cfg.pdf.sinkhorn_iters
    )

    cfgT = ClimateConfig{T}(
        dt = T(cfg.dt),
        steps = cfg.steps,
        initial_conditions = (cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch = cfg.samples_per_epoch,
        rng = cfg.rng,
        loss_mode = cfg.loss_mode,
        pdf = pdfT,                                 # <-- preserve user PDF config (promoted)
    )

    #  Convert base_u0 to T
    base_u0T = T.(base_u0)

    # PDF mode detection (Checks if any statistic in stats is :pdf3d to determine if PDF training is needed.)
    is_pdf = any(s -> s == :pdf3d, stats_tuple)

    # Setup targets (Prepares target statistics based on whether PDF training is selected or not.)
    if is_pdf
        centers  = cfgT.pdf.centers
        hT       = cfgT.pdf.bandwidth
        sinkεT   = cfgT.pdf.sinkhorn_ε
        pdf_mode = cfgT.pdf.loss_mode

        p3d_tgt = T.(getfield(target, :p3d))
        s_p = sum(p3d_tgt); s_p > 0 && (p3d_tgt ./= s_p)

        # --- consistency check ---
        if hasproperty(target, :centers)
            tgt_centers = T.(getfield(target, :centers))
            length(tgt_centers) == length(centers) ||
                error("target centers length mismatch cfg.pdf.centers")
            if maximum(abs.(tgt_centers .- centers)) > eps(T)*10
                @warn "target centers differ from cfg.pdf.centers; training uses cfg.pdf grid; expect a loss floor"
            end
        end
        
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

    # Optimiser state (Initializes the optimizer state and parameter state for training.)
    pstate = _to_state(L63Parameters(T(initial_params.σ),T(initial_params.ρ),T(initial_params.β),
                                     T(initial_params.x_s),T(initial_params.y_s),T(initial_params.z_s),T(initial_params.θ)))
    ost = Optimisers.setup(optimizer, pstate)

    # Bookkeeping (Initializes arrays for tracking losses and statistics history, as well as variables for early stopping.)
    losses = Vector{T}(undef, epochs)
    stats_history = Vector{Any}(undef, epochs)
    param_history = Vector{Any}(undef, epochs)
    best_loss = Inf; best_params = nothing; patience_counter = 0; actual_epochs = 0

    # Initial stratified indices (Generates initial stratified time indices for sampling during training.)
    I = jittered_grid_burnin(cfgT.steps, min(cfgT.samples_per_epoch, cfgT.steps); burn=0.3, jitter=0.35)

    # Verbose output (Displays initial training configuration details if verbose mode is enabled.)
    if verbose
        numICs = (cfgT.initial_conditions === nothing) ? 1 : size(cfgT.initial_conditions,2)
        if is_pdf
            @printf("UnifiedTrain | PDF3D loss=%s | dt=%.4g steps=%d | ICs=%d | k=%d | bins=%d^3\n",
                    string(pdf_mode), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch, length(centers))
        else
            @printf("UnifiedTrain | stats=%s | mask=%s | dt=%.4g steps=%d | ICs=%d | k=%d\n",
                    string(stats_tuple), string(update_mask), T(cfgT.dt), cfgT.steps, numICs, cfgT.samples_per_epoch)
        end
        @printf("Init  | %s\n", _fmt_params(pstate))
    end

    # Main training loop
    for epoch in 1:epochs

        # --- prepare stratified time indices for this epoch ---
        #k = min(cfgT.samples_per_epoch, cfgT.steps); @assert k > 0
        k = min(cfgT.samples_per_epoch, cfgT.steps)
        if epoch == 1 || (epoch % refresh == 1)
            #k = min(cfgT.samples_per_epoch, cfgT.steps)
            I = jittered_grid_burnin(cfgT.steps, k; burn=0.3, jitter=0.35)
        end
        # --- extract current params and epoch settings ---
        σ,ρ,β = pstate.σ[1], pstate.ρ[1], pstate.β[1]
        x_s,y_s,z_s,θ = pstate.x_s[1], pstate.y_s[1], pstate.z_s[1], pstate.θ[1]
        U_epoch = (cfgT.initial_conditions === nothing) ? reshape(T.(base_u0T),3,1) : T.(cfgT.initial_conditions)
        dt_epoch, steps_epoch = cfgT.dt, cfgT.steps

        # ---- bandwidth annealing (smooth → sharp) ----
        Δ = (centers[end]-centers[1]) / (length(centers)-1)   # grid spacing
        h_start = T(3) * Δ                                    # very smooth at start
        h_end   = T(0.7) * Δ                                  # sharp(er) at end
        τ = T(800)                                            # decay time (epochs)
        h_curr = h_end + (h_start - h_end) * exp(-T(epoch)/τ) # exponential schedule

        # --- compute loss and gradients ---
        if is_pdf
            # Defines a closure loss_entry_pdf3d that calls _loss_pdf_core_3d with the current parameters and epoch settings.
            @noinline function loss_entry_pdf3d(σ, ρ, β, x_s, y_s, z_s, θ,
                                                dt::S, steps::Int,
                                                U::AbstractMatrix{S}, I::AbstractVector{Int},
                                                centers::AbstractVector{S}, h::S,
                                                p3d_t::AbstractVector{S},
                                                mode::Symbol, sinkε::S, sinkiters::Int)::S where {S<:Real}
                _loss_pdf_core_3d(σ, ρ, β, x_s, y_s, z_s, θ,
                                  dt, steps, U, I, centers, h, p3d_t, mode, sinkε, sinkiters)
            end
            # Computes the loss for the current epoch using the loss_entry_pdf3d function.
            L = loss_entry_pdf3d(σ,ρ,β,x_s,y_s,z_s,θ,
                     dt_epoch, steps_epoch, U_epoch, I,
                     centers, h_curr,                    # <— here
                     p3d_tgt,
                     pdf_mode, sinkεT, cfgT.pdf.sinkhorn_iters)

            # Stores the computed loss for the current epoch.
            losses[epoch] = L
            # Computes gradients using Enzyme's automatic differentiation for the loss_entry_pdf3d function.
            gtuple = Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                loss_entry_pdf3d,
                Enzyme.Active(σ), Enzyme.Active(ρ), Enzyme.Active(β),
                Enzyme.Active(x_s), Enzyme.Active(y_s), Enzyme.Active(z_s), Enzyme.Active(θ),
                Enzyme.Const(dt_epoch), Enzyme.Const(steps_epoch),
                Enzyme.Const(U_epoch), Enzyme.Const(I),
                Enzyme.Const(centers), Enzyme.Const(h_curr),  # <- was hT
                Enzyme.Const(p3d_tgt),
                Enzyme.Const(pdf_mode), Enzyme.Const(sinkεT), Enzyme.Const(cfgT.pdf.sinkhorn_iters),
            )
            # Flattens the gradient tuple returned by Enzyme into a plain tuple for easier manipulation.
            g = _flatten_grads(gtuple)
            # Constructs a gradient state NamedTuple from the flattened gradients.
            gstate_all = (σ=[T(g[1])], ρ=[T(g[2])], β=[T(g[3])],
                          x_s=[T(g[4])], y_s=[T(g[5])], z_s=[T(g[6])], θ=[T(g[7])])
            # Applies the update mask to the gradient state to zero out gradients for parameters not being updated.
            gstate = _mask_grads(gstate_all, update_mask)
            # Verbose output (Displays gradient information if verbose mode is enabled.)
            gradient_verbose && @show gstate

            # clip exploding gradients before optimizer update
            gnorm_before = _gradnorm(gstate)
            if gnorm_before > gradient_clip_norm
                verbose && @printf("  Gradients clipped (norm %.3e > %.2f)\n", gnorm_before, gradient_clip_norm)
            end
            gstate = _clip_grads(gstate, gradient_clip_norm)  # threshold = 1.0 (tune as needed)
            
            old = _from_state(pstate)  # copy before update
            ost, pstate = Optimisers.update!(ost, pstate, gstate)
            new = _from_state(pstate)

            verbose && @printf("Epoch %4d | σ=%.6f (Δ%.2e)  ρ=%.6f (Δ%.2e)  β=%.6f (Δ%.2e)  | loss=%.6e | g_norm=%.3e\n",epoch,new.σ, new.σ - old.σ,new.ρ, new.ρ - old.ρ,new.β, new.β - old.β, L, _gradnorm(gstate))
        else
            # Defines a closure loss_entry that calls _loss_subsample_core with the current parameters and epoch settings.
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
            # clip exploding gradients before optimizer update
            gnorm_before = _gradnorm(gstate)
            if gnorm_before > gradient_clip_norm
                verbose && @printf("  Gradients clipped (norm %.3e > %.2f)\n", gnorm_before, gradient_clip_norm)
            end
            gstate = _clip_grads(gstate, gradient_clip_norm)  # threshold = 1.0 (tune as needed)
            ost, pstate = Optimisers.update!(ost, pstate, gstate)
            verbose && @printf("Epoch %4d | loss = %.6e | g_norm = %.3e | %s\n",
                    epoch, L, _gradnorm(gstate), _fmt_params(pstate))
        end

        # --- record statistics for this epoch (full trajectory) ---
        cur_params = _from_state(pstate)
        if is_pdf
            stats_history[epoch] = (; pdf3d = _compute_pdf3d_statistic(cur_params, cfgT, base_u0T, cfgT.pdf.centers, h_curr, I))
            param_history[epoch] = cur_params
        else
            stats_history[epoch] = evaluate_statistics(cur_params; cfg=cfgT, base_u0=base_u0T,stats=stats_tuple, full_trajectory=true)
            param_history[epoch] = cur_params
        end

        # --- early stopping (absolute OR relative improvement) ---
        # treat the very first valid loss as an improvement
        is_valid = isfinite(L)

        if best_loss == Inf
            improved = is_valid
        else
            thresh = max(early_stopping_min_delta, rel_delta * abs(best_loss))
            improved = is_valid && (best_loss - L) > thresh
        end

        if improved
            best_loss = L
            best_params = deepcopy(cur_params)
            patience_counter = 0
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
    param_history = param_history[1:actual_epochs]
    final_params = best_params === nothing ? _from_state(pstate) : best_params
    if is_pdf
        Iall = collect(2:cfgT.steps+1)
        # at the end
        h_final = T(0.7) * ((centers[end]-centers[1]) / (length(centers)-1))
        Iall = collect(2:cfgT.steps+1)
        final_stats = (; pdf3d = _compute_pdf3d_statistic(
            final_params, cfgT, base_u0T, cfgT.pdf.centers, h_final, Iall))
    else
        final_stats  = evaluate_statistics(final_params; cfg=cfgT, base_u0=base_u0T, stats=stats_tuple, full_trajectory=true)
    end
    return (params=final_params, loss_history=losses, final_statistics=final_stats, stats_history=stats_history, param_history=param_history)
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

Makes sense depending on the inital condition: We choose the U_0 so that we are on the atttor.
We could sample only from the later part of the trajectory to avoid transients, but stratified sampling
helps ensure we get a good coverage of the attractor even for shorter trajectories
"""
function stratified_indices(rng::AbstractRNG, steps::Int, k::Int)
    # Ensure k is not greater than steps
    k = min(k, steps)
    # Create stratified bins and sample one index from each bin
    bins = range(2, steps + 1; length = k + 1)
    # Compute the bin centers
    I = Vector{Int}(undef, k)
    # Sample one index from each bin
    @inbounds for j in 1:k
        lo = ceil(Int, bins[j])
        hi = floor(Int, bins[j + 1] - eps())
        if lo > hi
            mid = clamp(round(Int, (bins[j] + bins[j + 1]) / 2), 2, steps + 1)
            lo = hi = mid
        end
        I[j] = rand(rng, lo:hi)
    end
    # Sort the indices to ensure they are in increasing order
    sort!(I)
    # Return the stratified indices
    return I
end

"""
    jittered_grid(steps::Int, k::Int; jitter=0.49)
Return `k` time indices roughly evenly spaced across the trajectory
range `2:steps+1`, one sample per segment, with jitter within each segment.

Potentially useful alternative to `stratified_indices` to avoid clustering and more stable?
"""
function jittered_grid_burnin(steps::Int, k::Int; burn=0.3, jitter=0.35)
    start = 2 + round(Int, burn*steps)
    bins = range(start, steps+1; length=k+1)
    I = Vector{Int}(undef,k)
    @inbounds for j in 1:k
        lo, hi = ceil(Int,bins[j]), floor(Int,bins[j+1]-eps())
        mid = (lo + hi) ÷ 2
        δ = max(0, round(Int, jitter*(hi-lo)/2))
        I[j] = clamp(mid + rand((-δ):δ), lo, hi)
    end
    sort!(I); I
end




# ================================
# Gradient clipping
# ================================
@inline function _clip_grads(g, max_norm::Real)
    n = _gradnorm(g)
    if n > max_norm && isfinite(n)
        scale = max_norm / n
        return (σ=g.σ .* scale, ρ=g.ρ .* scale, β=g.β .* scale,
                x_s=g.x_s .* scale, y_s=g.y_s .* scale, z_s=g.z_s .* scale, θ=g.θ .* scale)
    else
        return g
    end
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

        for _ in 1:cfg.steps
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

    pdfT = PdfConfig{T}(
        T.(cfg.pdf.centers),
        T(cfg.pdf.bandwidth),
        cfg.pdf.loss_mode,
        T(cfg.pdf.sinkhorn_ε),
        cfg.pdf.sinkhorn_iters
    )

    cfgT = ClimateConfig{T}(
        dt = T(cfg.dt),
        steps = cfg.steps,
        initial_conditions = (cfg.initial_conditions === nothing ? nothing : T.(cfg.initial_conditions)),
        samples_per_epoch = cfg.samples_per_epoch,
        rng = cfg.rng,
        loss_mode = cfg.loss_mode,
        pdf = pdfT,                                 # <-- preserve user PDF config (promoted)
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


# fast soft histogram: only visit bins within ±R of nearest center (uniform grid assumed)
function soft_hist_3d_local(xs::AbstractVector{T}, ys::AbstractVector{T}, zs::AbstractVector{T},
                            centers::AbstractVector{T}, h::T, out::AbstractVector{T};
                            R::Int = 3) where {T}
    m = length(centers); @assert length(out) == m*m*m
    fill!(out, zero(T))
    Δ = (centers[end]-centers[1])/(m-1)                  # uniform
    inv2h2 = one(T)/(2*h*h)
    invΔ   = one(T)/Δ

    @inbounds for s in eachindex(xs)
        # nearest bin indices
        ix0 = clamp( Int(round((xs[s]-centers[1])*invΔ)) + 1, 1, m )
        iy0 = clamp( Int(round((ys[s]-centers[1])*invΔ)) + 1, 1, m )
        iz0 = clamp( Int(round((zs[s]-centers[1])*invΔ)) + 1, 1, m )

        iL = max(1, ix0-R); iH = min(m, ix0+R)
        jL = max(1, iy0-R); jH = min(m, iy0+R)
        kL = max(1, iz0-R); kH = min(m, iz0+R)

        for i in iL:iH
            dx2 = (xs[s]-centers[i])^2
            ex = exp(-dx2*inv2h2)
            for j in jL:jH
                dy2 = (ys[s]-centers[j])^2
                exy = ex * exp(-dy2*inv2h2)
                base = (i-1)*m*m + (j-1)*m
                for k in kL:kH
                    dz2 = (zs[s]-centers[k])^2
                    out[base + k] += exy * exp(-dz2*inv2h2)
                end
            end
        end
    end
    s = sum(out); s > eps(T) ? (out ./= s) : fill!(out, one(T)/length(out))
    return nothing
end



# "Normal" binned 3D PDF (histogram) — no kernel smoothing
# Assumes 'centers' are uniformly spaced; 'out' is length nbins^3 and flattened
# as (i-1)*m*m + (j-1)*m + k with i,j,k in 1:m.
# Hard 3D histogram using bin edges from midpoints of centers
@inline function hard_hist_3d(xs::AbstractVector{T}, ys::AbstractVector{T}, zs::AbstractVector{T},
                              centers::AbstractVector{T}, out::AbstractVector{T}) where {T}
    m = length(centers)
    @assert length(out) == m*m*m

    # Build edges by midpoints; extrapolate half-step at both ends
    edges = Vector{T}(undef, m + 1)
    @inbounds begin
        # interior midpoints
        for i in 2:m
            edges[i] = (centers[i-1] + centers[i]) / 2
        end
        # end caps using local spacing (robust to mild non-uniformity)
        edges[1]   = centers[1]  - (centers[2]     - centers[1])     / 2
        edges[m+1] = centers[m]  + (centers[m]     - centers[m-1])   / 2
    end

    fill!(out, zero(T))
    ns = length(xs)

    @inbounds for s in 1:ns
        i = clamp(searchsortedlast(edges, xs[s]), 1, m)
        j = clamp(searchsortedlast(edges, ys[s]), 1, m)
        k = clamp(searchsortedlast(edges, zs[s]), 1, m)
        out[(i-1)*m*m + (j-1)*m + k] += one(T)
    end

    s = sum(out)
    if s > 0
        out ./= s
    else
        # degenerate case: no samples → fallback to uniform
        fill!(out, one(T) / (m*m*m))
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
function _loss_pdf_core_3d(σ, ρ, β, x_s, y_s, z_s, θ,         # 7 trainable scalars
                           dt::S, steps::Int,                 # integrator controls
                           U::AbstractMatrix{S},              # 3×B initial conditions (columns)
                           I::AbstractVector{Int},            # k time indices in 2:steps+1
                           centers::AbstractVector{S},        # length m, shared for x,y,z
                           h::S,                              # Gaussian bandwidth
                           p3d_tgt::AbstractVector{S},        # target pdf (length m^3), normalized
                           loss_mode::Symbol,                 # :kl, :cross_entropy, :wasserstein
                           sink_ε::S, sink_iters::Int)::S where {S<:Real}
    
    # integrate each IC once for 'steps', sample at I, build 3D pdf, compute loss vs target pdf
    p = L63Parameters(S(σ),S(ρ),S(β),S(x_s),S(y_s),S(z_s),S(θ))
    B = size(U,2)       # number of initial conditions
    k = length(I)       # number of sampled time indices
    nx = k*B            # total sampled points
    xs = Vector{S}(undef, nx); ys = similar(xs); zs = similar(xs)
    idx = 1
    
    # for each inital condition, integrate and sample at I
    @inbounds for b in 1:B
        # current state
        u   = copy(@view U[:,b])
        # intermediate steps
        k1  = similar(u); 
        k2 = similar(u); 
        k3 = similar(u); 
        k4 = similar(u)
        # temporary storage
        tmp = similar(u)
        # time row tracking
        trow = 1; idxp = 1; nextI = I[idxp]
        # integrate and sample
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
    # Compute the 3D PDF
    m = length(centers)
    n = m*m*m
    p3d = Vector{S}(undef, n)
    # Differentiable surrogate for training
    # soft_hist_3d(xs, ys, zs, centers, h, p3d)  # old
    soft_hist_3d_local(xs, ys, zs, centers, h, p3d; R=3)  # new (≈100–1000× faster)

    # Computes the loss between the predicted and target PDF using the selected mode
    if loss_mode === :cross_entropy
        return _cross_entropy_vec(p3d_tgt, p3d)
    elseif loss_mode === :kl
        return _kl_divergence_vec(p3d_tgt, p3d)
    elseif loss_mode === :wasserstein
        return _sinkhorn_wasserstein_nd(p3d_tgt, p3d, centers; ε=sink_ε, n_iter=sink_iters)
    else
        error("Unknown pdf loss mode: $loss_mode, must be :kl, :cross_entropy, or :wasserstein")
    end
end

function _compute_pdf3d_statistic(params::L63Parameters{S},
                                  cfg::ClimateConfig{S},
                                  base_u0::AbstractVector{S},
                                  centers::AbstractVector{S},
                                  h::S,
                                  I::AbstractVector{Int}) where {S<:Real}

    @assert length(I) > 0 "At least one time index must be provided"
    # initial conditions
    U = cfg.initial_conditions === nothing ? reshape(S.(base_u0), 3, 1) : S.(cfg.initial_conditions)
    _assert_ics(U)
    # parameters
    p = L63Parameters(S(params.σ), S(params.ρ), S(params.β),
                      S(params.x_s), S(params.y_s), S(params.z_s), S(params.θ))
    # prepare storage
    B = size(U,2)
    k = length(I)
    @assert k > 0 "PDF statistic requires at least one sampled time index"

    # extract sampled points
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
    hard_hist_3d(xs, ys, zs, centers, p3d)
    sum_p = sum(p3d)
    sum_p > 0 && (p3d ./= sum_p)
    return (; centers=centers, p3d=p3d)
end

end # module
