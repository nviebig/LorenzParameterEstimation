# ================================ Loss Functions ================================

using Statistics: mean
using Enzyme  # For compute_gradients() function

# NOTE: This file uses Enzyme.jl for precise gradient computation.
# All loss functions work with the modular_train!() function,
# while compute_gradients() is used by both train!() and modular_train!() functions.

"""
    window_rmse(predicted, target)

RMSE loss for trajectory windows. Uses Flux's mse under the hood.
"""
function window_rmse(predicted::AbstractMatrix, target::AbstractMatrix)
    mse_val = mean((predicted .- target).^2)
    return sqrt(mse_val)
end

"""
    window_mae(predicted, target) 

MAE loss for trajectory windows. Uses Flux's mae.
"""
function window_mae(predicted, target)
    return mean(abs.(predicted .- target))
end

"""
    window_mse(predicted, target)

MSE loss for trajectory windows. Direct Flux mse.
"""
function window_mse(predicted, target)
    return mean((predicted .- target).^2)
end

"""
    weighted_window_loss(base_loss_fn, weight_exponent=1.0)

Returns a weighted loss function that emphasizes later time steps.
Uses any base loss function (e.g., mse, mae, window_rmse).

# Example
```julia
loss_fn = weighted_window_loss(window_rmse, 1.5)
loss_value = loss_fn(predicted, target)
```
"""
function weighted_window_loss(base_loss_fn, weight_exponent::Real = 1.0)
    return function(predicted::AbstractMatrix, target::AbstractMatrix)
        window_length = size(predicted, 1)
        
        # Create exponential weights: later time steps get higher weight
        weights = [(i / window_length)^weight_exponent for i in 1:window_length]
        weights ./= sum(weights)  # Normalize
        
        total_loss = 0.0
        for i in 1:window_length
            window_pred = reshape(predicted[i, :], 1, :)
            window_target = reshape(target[i, :], 1, :)
            loss_val = base_loss_fn(window_pred, window_target)
            total_loss += weights[i] * loss_val
        end
        
        return total_loss
    end
end

"""
    probabilistic_loss(predicted, target; noise_std=0.1)

Negative log-likelihood loss assuming Gaussian noise.
Useful for Bayesian approaches.
"""
function probabilistic_loss(predicted::AbstractMatrix, target::AbstractMatrix; noise_std::Real = 0.1)
    diff = predicted .- target
    # Negative log-likelihood for Gaussian noise
    nll = 0.5 * sum(diff.^2) / (noise_std^2) + 0.5 * length(diff) * log(2π * noise_std^2)
    return nll
end

"""
    adaptive_loss(predicted, target; β=0.5)

Adaptive loss that interpolates between L1 and L2 based on residual magnitude.
Robust to outliers while maintaining efficiency.
"""
function adaptive_loss(predicted::AbstractMatrix, target::AbstractMatrix; β::Real = 0.5)
    residuals = abs.(predicted .- target)
    # Smooth L1 loss (Huber-like)
    mask = residuals .< β
    loss = sum(mask .* (0.5 .* residuals.^2 ./ β) .+ 
               (.!mask) .* (residuals .- 0.5 * β))
    return loss / length(residuals)
end

"""
    compute_loss(params::L63Parameters, target_solution::L63Solution, window_start::Int, window_length::Int)

Compute RMSE loss between predicted and target trajectories over a window.

Why windowed?
The Lorenz-63 system is chaotic: two trajectories with slightly different parameters diverge exponentially in time. 
If you compute the loss over a long trajectory, even tiny parameter errors cause huge mismatches after some time → the loss explodes and gradients become meaningless (dominated by chaotic drift).

A windowed loss fixes this by:

	•	Slicing the target trajectory into short segments of length window_length.
	•	Restarting each segment from the observed state (teacher forcing).
	•	Computing the loss only over this short horizon.

That way, the model stays close to the observations inside each window, and the gradients you backpropagate remain informative for parameter updates.

In practice:
	•	Long-horizon loss = unstable, useless gradients.
	•	Windowed loss = stable, local gradients that guide parameter recovery.


Uses teacher forcing: starts from the observed state at window_start and
integrates forward for window_length steps, comparing to target.

# Arguments
- `params`: Model parameters to evaluate
- `target_solution`: Target/observed trajectory 
- `window_start`: Starting index in target trajectory (1-based)
- `window_length`: Number of integration steps

# Returns
- RMSE loss over the window
"""


function compute_loss(
    params::L63Parameters{T},             # Parameters to evaluate
    target_solution::L63Solution{T},      # Target trajectory    
    window_start::Int, window_length::Int # Number of integration steps
    ) where {T}
    
    # Validate window bounds
    max_start = length(target_solution) - window_length
    (1 <= window_start <= max_start) || throw(BoundsError("Invalid window bounds"))
    
    # Teacher forcing: start from observed state
    u = copy(target_solution[window_start])
    dt = target_solution.system.dt
    
    # Accumulate squared errors
    se = zero(T)
    count = 0
    
    @inbounds for i in 1:window_length
        # Integrate one step
        u = rk4_step(u, params, dt)
        
        # Compare to target
        target = target_solution[window_start + i]
        for j in 1:3
            diff = u[j] - target[j]
            se += diff * diff
            count += 1
        end
    end
    
    return sqrt(se / count)  # RMSE
end

"""
    loss_function_enzyme(σ, ρ, β, u0, target_trajectory, window_length, dt)

Enzyme-compatible wrapper for loss computation with scalar parameters.
"""
@inline function loss_function_enzyme(
    σ::T,                           # Parameter σ
    ρ::T,                           # Parameter ρ
    β::T,                           # Parameter β
    u0::Vector{T},                  # Initial state (3-vector)
    target_trajectory::Matrix{T},   # Target trajectory (N×3 matrix)
    window_length::Int,             # Number of integration steps
    dt::T
    ) where {T}
    
    params = L63Parameters{T}(σ, ρ, β) # Construct parameters
    u = copy(u0) # Initial state
    
    se = zero(T) # Squared error accumulator
    count = 0    # Count of comparisons
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        
        # Compare to target at row i+1 (since target includes starting state)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            se += diff * diff
            count += 1
        end
    end
    
    return sqrt(se / count)
end

"""
    compute_gradients(params::L63Parameters, target_solution::L63Solution, 
                     window_start::Int, window_length::Int)

Compute gradients of loss function with respect to parameters using Enzyme AD.

# Returns
- `(loss_value, gradients)` where gradients is an L63Parameters object
"""
function compute_gradients(params::L63Parameters{T}, target_solution::L63Solution{T},
                          window_start::Int, window_length::Int) where {T}
    
    # Extract window data
    u0 = target_solution[window_start]
    window_end = window_start + window_length
    target_window = target_solution.u[window_start:window_end, :]
    dt = target_solution.system.dt
    
    # Convert to compatible types
    σ0, ρ0, β0 = T(params.σ), T(params.ρ), T(params.β)
    u0_vec = Vector{T}(u0)
    target_mat = Matrix{T}(target_window)
    
    # Compute gradients using Enzyme
    grads = Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(loss_function_enzyme),
        Enzyme.Active(σ0),
        Enzyme.Active(ρ0), 
        Enzyme.Active(β0),
        Enzyme.Const(u0_vec),
        Enzyme.Const(target_mat),
        Enzyme.Const(window_length),
        Enzyme.Const(dt)
    )
    
    # Extract gradients and compute loss
    G = grads[1]
    gσ, gρ, gβ = G[1], G[2], G[3]
    
    loss_val = loss_function_enzyme(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
    
    return loss_val, L63Parameters{T}(gσ, gρ, gβ)
end
