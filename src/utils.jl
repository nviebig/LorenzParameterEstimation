# ================================ Utilities ================================

"""
    classic_params(T=Float64)

Return classic chaotic Lorenz-63 parameters (σ=10, ρ=28, β=8/3).
"""
classic_params(::Type{T}=Float64) where {T} = L63Parameters{T}(T(10), T(28), T(8)/T(3))

"""
    stable_params(T=Float64)

Return parameters for stable/periodic behavior (σ=10, ρ=15, β=8/3).
"""
stable_params(::Type{T}=Float64) where {T} = L63Parameters{T}(T(10), T(15), T(8)/T(3))

"""
    periodic_params(T=Float64)

Return parameters for periodic orbits (σ=10, ρ=8, β=8/3).
"""
periodic_params(::Type{T}=Float64) where {T} = L63Parameters{T}(T(10), T(8), T(8)/T(3))

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
    noisy_trajectory = solution.u + σ_noise * randn(T, size(solution.u))
    
    return L63Solution{T}(solution.t, noisy_trajectory, solution.system)
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
    train_system = L63System{T}(
        solution.system.params,
        solution.system.u0, 
        (solution.t[1], solution.t[n_train]),
        solution.system.dt
    )
    train_solution = L63Solution{T}(train_t, train_u, train_system)
    
    # Validation set  
    val_t = solution.t[n_train+1:end]
    val_u = solution.u[n_train+1:end, :]
    val_system = L63System{T}(
        solution.system.params,
        solution.u[n_train+1, :],  # New initial condition
        (solution.t[n_train+1], solution.t[end]),
        solution.system.dt
    )
    val_solution = L63Solution{T}(val_t, val_u, val_system)
    
    return train_solution, val_solution
end