# ================================ Core Types ================================

import Optimisers
import Random

"""
    L63Parameters{T<:Real}

Parameters for the Lorenz-63 system.

# Fields
- `σ::T`: Prandtl number (buoyancy frequency) 
- `ρ::T`: Rayleigh number (density parameter)
- `β::T`: Geometric parameter (thermal expansion coefficient)

# Examples
```julia
# Classic chaotic parameters
params = L63Parameters(10.0, 28.0, 8.0/3.0)

# Create with keyword arguments
params = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
```
"""
struct L63Parameters{T<:Real}
    σ::T  # Prandtl number
    ρ::T  # Rayleigh number  
    β::T  # Geometric parameter
end


# Constructor with keyword arguments
L63Parameters(; σ::T, ρ::T, β::T) where {T<:Real} = L63Parameters{T}(σ, ρ, β)

# Promote constructor for mixed types
L63Parameters(σ, ρ, β) = (T = promote_type(typeof.((σ, ρ, β))...); L63Parameters{T}(T(σ), T(ρ), T(β)))

# Arithmetic operations for parameter updates
Base.:+(p1::L63Parameters, p2::L63Parameters) = L63Parameters(p1.σ + p2.σ, p1.ρ + p2.ρ, p1.β + p2.β)
Base.:-(p1::L63Parameters, p2::L63Parameters) = L63Parameters(p1.σ - p2.σ, p1.ρ - p2.ρ, p1.β - p2.β)
Base.:*(α::Real, p::L63Parameters) = L63Parameters(α * p.σ, α * p.ρ, α * p.β)
Base.:*(p::L63Parameters, α::Real) = α * p

# Array-like interface for gradient operations
Base.length(::L63Parameters) = 3
Base.iterate(p::L63Parameters, state=1) = state > 3 ? nothing : (getfield(p, state), state + 1)
Base.getindex(p::L63Parameters, i::Int) = getfield(p, i)
Base.isapprox(p1::L63Parameters, p2::L63Parameters; kwargs...) = 
    isapprox(p1.σ, p2.σ; kwargs...) && isapprox(p1.ρ, p2.ρ; kwargs...) && isapprox(p1.β, p2.β; kwargs...)

# Broadcasting support
Base.broadcastable(p::L63Parameters) = (p.σ, p.ρ, p.β)

# Norm for gradient clipping
LinearAlgebra.norm(p::L63Parameters) = sqrt(p.σ^2 + p.ρ^2 + p.β^2)

"""
    L63System{T<:Real}

Complete specification of a Lorenz-63 system including parameters and initial conditions.

# Fields
- `params::L63Parameters{T}`: System parameters
- `u0::AbstractVector{T}`: Initial conditions [x, y, z]
- `tspan::Tuple{T,T}`: Time span (t0, tf)
- `dt::T`: Time step size

# Examples
```julia
sys = L63System(
    params = L63Parameters(10.0, 28.0, 8.0/3.0),
    u0 = [1.0, 1.0, 1.0],
    tspan = (0.0, 10.0),
    dt = 0.01
)
```
"""
struct L63System{T<:Real,V<:AbstractVector{T}}
    params::L63Parameters{T}
    u0::V
    tspan::Tuple{T,T}
    dt::T
    
    function L63System{T,V}(params::L63Parameters{T}, u0::V, tspan::Tuple{T,T}, dt::T) where {T<:Real,V<:AbstractVector{T}}
        length(u0) == 3 || throw(ArgumentError("Initial condition must be 3-dimensional"))
        tspan[2] > tspan[1] || throw(ArgumentError("Final time must be greater than initial time"))
        dt > 0 || throw(ArgumentError("Time step must be positive"))
        new{T,V}(params, u0, tspan, dt)
    end
end

# Outer constructor with keyword arguments
function L63System(; params::L63Parameters{T}, u0::AbstractVector{S}, tspan::Tuple{U,V}, dt::W) where {T,S,U,V,W}
    R = promote_type(T, S, U, V, W)
    # Convert u0 to the promoted type while preserving array type
    u0_converted = similar(u0, R)
    u0_converted .= R.(u0)
    L63System{R,typeof(u0_converted)}(
        L63Parameters{R}(R(params.σ), R(params.ρ), R(params.β)),
        u0_converted,
        (R(tspan[1]), R(tspan[2])),
        R(dt)
    )
end

"""
    L63Solution{T<:Real}

Solution of a Lorenz-63 integration containing trajectory data and metadata.

# Fields
- `t::AbstractVector{T}`: Time points
- `u::AbstractMatrix{T}`: State trajectory (N×3 matrix)
- `system::L63System{T,V}`: Original system specification
- `final_state::AbstractVector{T}`: Final state
- `success::Bool`: Integration success flag
"""
struct L63Solution{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T},S}
    t::V
    u::M  # N×3 trajectory matrix
    system::S
    final_state::V
    success::Bool
    
    function L63Solution{T,V,M,S}(t::V, u::M, system::S) where {T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T},S}
        size(u, 2) == 3 || throw(ArgumentError("Trajectory must have 3 spatial dimensions"))
        length(t) == size(u, 1) || throw(ArgumentError("Time and trajectory dimensions must match"))
        final_state = size(u, 1) > 0 ? u[end, :] : similar(t, 3)
        if size(u, 1) == 0
            fill!(final_state, zero(T))
        end
        new{T,V,M,S}(t, u, system, final_state, true)
    end
end

# Convenience constructor
function L63Solution(t::V, u::M, system::S) where {T,V<:AbstractVector{T},M<:AbstractMatrix{T},S}
    L63Solution{T,V,M,S}(t, u, system)
end

# Convenience accessors
Base.length(sol::L63Solution) = size(sol.u, 1)
Base.getindex(sol::L63Solution, i::Int) = sol.u[i, :]
Base.getindex(sol::L63Solution, i::Int, j::Int) = sol.u[i, j]

"""
    L63TrainingConfig{T<:Real}

Configuration for parameter estimation training.

# Fields
- `epochs::Int`: Number of training epochs
- `η::T`: Learning rate
- `window_size::Int`: Length of training windows (steps)
- `clip_norm::T`: Gradient clipping threshold
- `update_mask::NamedTuple`: Which parameters to update (σ=true/false, ρ=true/false, β=true/false)
- `verbose::Bool`: Print training progress
"""
struct L63TrainingConfig{T<:Real,O,F,R<:Random.AbstractRNG}
    epochs::Int
    η::T
    window_size::Int
    stride::Int
    clip_norm::T
    update_mask::NamedTuple{(:σ, :ρ, :β), Tuple{Bool, Bool, Bool}}
    verbose::Bool
    batch_size::Int
    optimiser::O
    loss::F
    train_fraction::Float64
    shuffle::Bool
    rng::R
    eval_every::Int
    
    function L63TrainingConfig{T,O,F,R}(
        epochs::Int,
        η::T,
        window_size::Int,
        stride::Int,
        clip_norm::T,
        update_mask::NamedTuple{(:σ, :ρ, :β), Tuple{Bool, Bool, Bool}},
        verbose::Bool,
        batch_size::Int,
        optimiser::O,
        loss::F,
        train_fraction::Float64,
        shuffle::Bool,
        rng::R,
        eval_every::Int
    ) where {T<:Real,O,F,R<:Random.AbstractRNG}
        epochs > 0 || throw(ArgumentError("Number of epochs must be positive"))
        η > 0 || throw(ArgumentError("Learning rate must be positive"))
        window_size > 0 || throw(ArgumentError("Window size must be positive"))
        stride > 0 || throw(ArgumentError("Stride must be positive"))
        clip_norm > 0 || throw(ArgumentError("Clip norm must be positive"))
        batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
        0 < train_fraction <= 1 || throw(ArgumentError("Train fraction must be in (0, 1]"))
        eval_every > 0 || throw(ArgumentError("Evaluation interval must be positive"))
        new{T,O,F,R}(epochs, η, window_size, stride, clip_norm, update_mask, verbose,
                     batch_size, optimiser, loss, train_fraction, shuffle, rng, eval_every)
    end
end

# Constructor with keyword arguments and defaults
function L63TrainingConfig(; 
    epochs::Int = 100,
    η::Real = 5e-3, 
    window_size::Int = 300,
    stride::Union{Nothing, Int} = nothing,
    clip_norm::Real = 5.0,
    batch_size::Int = 16,
    optimiser = nothing,
    loss = nothing,
    train_fraction::Real = 0.8,
    shuffle::Bool = true,
    rng::Union{Nothing, Random.AbstractRNG} = nothing,
    eval_every::Int = 1,
    update_σ::Bool = true,
    update_ρ::Bool = true, 
    update_β::Bool = true,
    verbose::Bool = true
)
    T = promote_type(typeof.((η, clip_norm))...)
    stride_val = isnothing(stride) ? window_size : stride
    stride_val > 0 || throw(ArgumentError("Stride must be positive"))
    update_mask = (σ=update_σ, ρ=update_ρ, β=update_β)
    rng_val = isnothing(rng) ? Random.default_rng() : rng
    opt = isnothing(optimiser) ? Optimisers.OptimiserChain(
        Optimisers.ClipNorm(T(clip_norm)),
        Optimisers.Descent(T(η))
    ) : optimiser
    train_fraction_val = Float64(train_fraction)
    loss_fn = loss

    return L63TrainingConfig{T, typeof(opt), typeof(loss_fn), typeof(rng_val)}(
        epochs,
        T(η),
        window_size,
        stride_val,
        T(clip_norm),
        update_mask,
        verbose,
        batch_size,
        opt,
        loss_fn,
        train_fraction_val,
        shuffle,
        rng_val,
        eval_every
    )
end
