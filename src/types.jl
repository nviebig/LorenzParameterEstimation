# ================================ Core Types ================================

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

# Norm for gradient clipping
LinearAlgebra.norm(p::L63Parameters) = sqrt(p.σ^2 + p.ρ^2 + p.β^2)

"""
    L63System{T<:Real}

Complete specification of a Lorenz-63 system including parameters and initial conditions.

# Fields
- `params::L63Parameters{T}`: System parameters
- `u0::Vector{T}`: Initial conditions [x, y, z]
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
struct L63System{T<:Real}
    params::L63Parameters{T}
    u0::Vector{T}
    tspan::Tuple{T,T}
    dt::T
    
    function L63System{T}(params::L63Parameters{T}, u0::Vector{T}, tspan::Tuple{T,T}, dt::T) where {T<:Real}
        length(u0) == 3 || throw(ArgumentError("Initial condition must be 3-dimensional"))
        tspan[2] > tspan[1] || throw(ArgumentError("Final time must be greater than initial time"))
        dt > 0 || throw(ArgumentError("Time step must be positive"))
        new{T}(params, u0, tspan, dt)
    end
end

# Outer constructor with keyword arguments
function L63System(; params::L63Parameters{T}, u0::Vector{S}, tspan::Tuple{U,V}, dt::W) where {T,S,U,V,W}
    R = promote_type(T, S, U, V, W)
    L63System{R}(
        L63Parameters{R}(R(params.σ), R(params.ρ), R(params.β)),
        Vector{R}(u0),
        (R(tspan[1]), R(tspan[2])),
        R(dt)
    )
end

"""
    L63Solution{T<:Real}

Solution of a Lorenz-63 integration containing trajectory data and metadata.

# Fields
- `t::Vector{T}`: Time points
- `u::Matrix{T}`: State trajectory (N×3 matrix)
- `system::L63System{T}`: Original system specification
- `final_state::Vector{T}`: Final state
- `success::Bool`: Integration success flag
"""
struct L63Solution{T<:Real}
    t::Vector{T}
    u::Matrix{T}  # N×3 trajectory matrix
    system::L63System{T}
    final_state::Vector{T}
    success::Bool
    
    function L63Solution{T}(t::Vector{T}, u::Matrix{T}, system::L63System{T}) where {T<:Real}
        size(u, 2) == 3 || throw(ArgumentError("Trajectory must have 3 spatial dimensions"))
        length(t) == size(u, 1) || throw(ArgumentError("Time and trajectory dimensions must match"))
        final_state = size(u, 1) > 0 ? u[end, :] : zeros(T, 3)
        new{T}(t, u, system, final_state, true)
    end
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
struct L63TrainingConfig{T<:Real}
    epochs::Int
    η::T
    window_size::Int
    clip_norm::T
    update_mask::NamedTuple{(:σ, :ρ, :β), Tuple{Bool, Bool, Bool}}
    verbose::Bool
    
    function L63TrainingConfig{T}(epochs::Int, η::T, window_size::Int, clip_norm::T, 
                                  update_mask::NamedTuple, verbose::Bool) where {T<:Real}
        epochs > 0 || throw(ArgumentError("Number of epochs must be positive"))
        η > 0 || throw(ArgumentError("Learning rate must be positive"))
        window_size > 0 || throw(ArgumentError("Window size must be positive"))
        clip_norm > 0 || throw(ArgumentError("Clip norm must be positive"))
        new{T}(epochs, η, window_size, clip_norm, update_mask, verbose)
    end
end

# Constructor with keyword arguments and defaults
function L63TrainingConfig(; 
    epochs::Int = 100,
    η::Real = 5e-3, 
    window_size::Int = 300,
    clip_norm::Real = 5.0,
    update_σ::Bool = true,
    update_ρ::Bool = true, 
    update_β::Bool = true,
    verbose::Bool = true
)
    T = promote_type(typeof.((η, clip_norm))...)
    update_mask = (σ=update_σ, ρ=update_ρ, β=update_β)
    L63TrainingConfig{T}(epochs, T(η), window_size, T(clip_norm), update_mask, verbose)
end