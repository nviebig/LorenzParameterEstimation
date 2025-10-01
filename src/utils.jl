# ================================ Utilities ================================

# ================================ Array Backend Utilities ================================

"""
    similar_array(template::AbstractArray, ::Type{T}, dims...)

Create an array of type T with the same backend as the template array.
This ensures GPU/CPU compatibility by inferring the array type from existing arrays.
"""
similar_array(template::AbstractArray, ::Type{T}, dims...) where {T} = similar(template, T, dims...)

"""
    zeros_like(template::AbstractArray, ::Type{T}, dims...)

Create a zero-filled array of type T with the same backend as the template array.
"""
function zeros_like(template::AbstractArray, ::Type{T}, dims...) where {T}
    arr = similar(template, T, dims...)
    fill!(arr, zero(T))
    return arr
end

"""
    ones_like(template::AbstractArray, ::Type{T}, dims...)

Create a ones-filled array of type T with the same backend as the template array.
"""
function ones_like(template::AbstractArray, ::Type{T}, dims...) where {T}
    arr = similar(template, T, dims...)
    fill!(arr, one(T))
    return arr
end

"""
    adapt_array_type(target::AbstractArray, source::AbstractArray)

Convert source array to have the same backend type as target array.
"""
function adapt_array_type(target::AbstractArray, source::AbstractArray)
    # If they're already the same type, return source
    typeof(target) == typeof(source) && return source
    
    # Create a new array with the same backend as target
    result = similar(target, eltype(source), size(source))
    result .= source
    return result
end

"""
    infer_array_type(arrays::AbstractArray...)

Infer the common array backend type from multiple arrays.
Returns the type of the first non-Vector array, or Vector if all are Vector.
"""
function infer_array_type(arrays::AbstractArray...)
    isempty(arrays) && return Vector
    
    # Find the first array that's not a plain Vector/Matrix
    for arr in arrays
        if !(arr isa Vector || arr isa Matrix)
            return typeof(arr)
        end
    end
    
    # If all are plain arrays, return the type of the first one
    return typeof(first(arrays))
end

# ================================ Parameter and System Utilities ================================

"""
    classic_params(T=Float64)

Return classic chaotic Lorenz-63 parameters (σ=10, ρ=28, β=8/3) with default extended parameters.
"""
classic_params(::Type{T}=Float64) where {T} = L63Parameters(T(10), T(28), T(8)/T(3))

"""
    stable_params(T=Float64)

Return parameters for stable/periodic behavior (σ=10, ρ=15, β=8/3) with default extended parameters.
"""
stable_params(::Type{T}=Float64) where {T} = L63Parameters(T(10), T(15), T(8)/T(3))

"""
    periodic_params(T=Float64)

Return parameters for periodic orbits (σ=10, ρ=8, β=8/3) with default extended parameters.
"""
periodic_params(::Type{T}=Float64) where {T} = L63Parameters(T(10), T(8), T(8)/T(3))

"""
    standard_initial_condition(T=Float64)

Return standard initial condition [1, 1, 1].
"""
standard_initial_condition(::Type{T}=Float64) where {T} = T[1, 1, 1]

"""
    lyapunov_time(params::L63Parameters)

Estimate the Lyapunov time scale for the given parameters.
For classic parameters, this is approximately 1/0.9 ≈ 1.1 time units.
"""
function lyapunov_time(params::L63Parameters{T}) where {T}
    # Rough approximation based on classic parameters
    if abs(params.σ - 10) < 1 && abs(params.ρ - 28) < 5 && abs(params.β - 8/3) < 0.5
        return T(1.1)  # Classic case
    else
        return T(1.0)  # Conservative estimate
    end
end

"""
    optimal_window_size(params::L63Parameters, dt::Real)

Suggest optimal window size for training (1-2 Lyapunov times).
"""
function optimal_window_size(params::L63Parameters, dt::Real)
    τ_L = lyapunov_time(params)
    window_time = 1.5 * τ_L  # 1.5 Lyapunov times
    return max(50, Int(round(window_time / dt)))  # At least 50 steps
end

"""
    parameter_error(true_params::L63Parameters, estimated_params::L63Parameters)

Compute relative parameter estimation errors.
"""
function parameter_error(true_params::L63Parameters{T}, estimated_params::L63Parameters{T}) where {T}
    σ_error = abs(estimated_params.σ - true_params.σ) / abs(true_params.σ)
    ρ_error = abs(estimated_params.ρ - true_params.ρ) / abs(true_params.ρ) 
    β_error = abs(estimated_params.β - true_params.β) / abs(true_params.β)
    
    return (σ=σ_error, ρ=ρ_error, β=β_error, total=σ_error + ρ_error + β_error)
end

"""
    generate_noisy_observations(solution::L63Solution, noise_level::Real)

Add Gaussian noise to a clean trajectory to simulate observations.
"""
function generate_noisy_observations(solution::L63Solution{T}, noise_level::Real) where {T}
    σ_noise = T(noise_level)
    # Generate noise with the same array backend as the trajectory
    noise = similar_array(solution.u, T, size(solution.u))
    randn!(noise)
    noisy_trajectory = solution.u + σ_noise * noise
    
    return L63Solution(solution.t, noisy_trajectory, solution.system)
end

"""
    validation_split(solution::L63Solution, train_fraction::Real=0.8)

Split solution into training and validation sets.
"""
function validation_split(solution::L63Solution{T}, train_fraction::Real=0.8) where {T}
    n_total = length(solution)
    n_train = Int(round(train_fraction * n_total))
    
    # Training set
    train_t = solution.t[1:n_train]
    train_u = solution.u[1:n_train, :]
    train_system = L63System(
        params=solution.system.params,
        u0=solution.system.u0, 
        tspan=(solution.t[1], solution.t[n_train]),
        dt=solution.system.dt
    )
    train_solution = L63Solution(train_t, train_u, train_system)
    
    # Validation set  
    val_t = solution.t[n_train+1:end]
    val_u = solution.u[n_train+1:end, :]
    val_system = L63System(
        params=solution.system.params,
        u0=solution.u[n_train+1, :],  # New initial condition
        tspan=(solution.t[n_train+1], solution.t[end]),
        dt=solution.system.dt
    )
    val_solution = L63Solution(val_t, val_u, val_system)
    
    return train_solution, val_solution
end

# ================================ Extended Parameter Utilities ================================

"""
    with_coordinate_shifts(params::L63Parameters, x_s, y_s, z_s)

Create a new L63Parameters with specified coordinate shifts, keeping other parameters unchanged.
"""
function with_coordinate_shifts(params::L63Parameters{T}, x_s, y_s, z_s) where {T}
    return L63Parameters(params.σ, params.ρ, params.β, T(x_s), T(y_s), T(z_s), params.θ)
end

"""
    with_theta(params::L63Parameters, θ)

Create a new L63Parameters with specified theta parameter, keeping other parameters unchanged.
"""
function with_theta(params::L63Parameters{T}, θ) where {T}
    return L63Parameters(params.σ, params.ρ, params.β, params.x_s, params.y_s, params.z_s, T(θ))
end

"""
    classic_lorenz(params::L63Parameters)

Extract only the classic Lorenz parameters (σ, ρ, β) by setting extended parameters to defaults.
"""
function classic_lorenz(params::L63Parameters{T}) where {T}
    return L63Parameters(params.σ, params.ρ, params.β)  # Uses default values for extended params
end

"""
    has_coordinate_shifts(params::L63Parameters)

Check if the parameters have non-zero coordinate shifts.
"""
function has_coordinate_shifts(params::L63Parameters)
    return !(params.x_s ≈ 0 && params.y_s ≈ 0 && params.z_s ≈ 0)
end

"""
    has_theta_modification(params::L63Parameters)

Check if the parameters have a theta modification (θ ≠ 1).
"""
function has_theta_modification(params::L63Parameters)
    return !(params.θ ≈ 1)
end

"""
    parameter_summary(params::L63Parameters)

Print a summary of the parameter values and which extensions are active.
"""
function parameter_summary(params::L63Parameters)
    println("L63Parameters Summary:")
    println("  Core parameters: σ=$(params.σ), ρ=$(params.ρ), β=$(params.β)")
    if has_coordinate_shifts(params)
        println("  Coordinate shifts: x_s=$(params.x_s), y_s=$(params.y_s), z_s=$(params.z_s)")
    else
        println("  Coordinate shifts: None (all zero)")
    end
    if has_theta_modification(params)
        println("  Theta modification: θ=$(params.θ)")
    else
        println("  Theta modification: None (θ=1)")
    end
end