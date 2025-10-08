# ================================ Loss Functions ================================

using Statistics: mean
using Enzyme  # For compute_gradients() function

# NOTE: This file uses Enzyme.jl for precise gradient computation.
# All loss functions work with the modular_train!() function,
# while compute_gradients() is used by both train!() and modular_train!() functions.

# ================================ Tracking Structures ================================

"""
    TrainingMetrics

Structure to store detailed training metrics including individual window losses/gradients
and batch averages. Useful for analyzing chaotic behavior vs meaningful averaged information.
"""
mutable struct TrainingMetrics{T <: Real}
    # Loss tracking
    individual_losses::Vector{T}                   # Loss for each individual window
    batch_losses::Vector{T}                        # Average loss per batch
    epoch_losses::Vector{T}                        # Average loss per epoch
    
    # Gradient tracking for σ parameter
    individual_gradients_σ::Vector{T}              # σ gradient for each window
    batch_gradients_σ::Vector{T}                   # Average σ gradient per batch
    epoch_gradients_σ::Vector{T}                   # Average σ gradient per epoch
    
    # Gradient tracking for ρ parameter  
    individual_gradients_ρ::Vector{T}              # ρ gradient for each window
    batch_gradients_ρ::Vector{T}                   # Average ρ gradient per batch
    epoch_gradients_ρ::Vector{T}                   # Average ρ gradient per epoch
    
    # Gradient tracking for β parameter
    individual_gradients_β::Vector{T}              # β gradient for each window
    batch_gradients_β::Vector{T}                   # Average β gradient per batch
    epoch_gradients_β::Vector{T}                   # Average β gradient per epoch
    
    # Gradient tracking for x_s parameter (coordinate shift)
    individual_gradients_x_s::Vector{T}            # x_s gradient for each window
    batch_gradients_x_s::Vector{T}                 # Average x_s gradient per batch
    epoch_gradients_x_s::Vector{T}                 # Average x_s gradient per epoch
    
    # Gradient tracking for y_s parameter (coordinate shift)
    individual_gradients_y_s::Vector{T}            # y_s gradient for each window
    batch_gradients_y_s::Vector{T}                 # Average y_s gradient per batch
    epoch_gradients_y_s::Vector{T}                 # Average y_s gradient per epoch
    
    # Gradient tracking for z_s parameter (coordinate shift)
    individual_gradients_z_s::Vector{T}            # z_s gradient for each window
    batch_gradients_z_s::Vector{T}                 # Average z_s gradient per batch
    epoch_gradients_z_s::Vector{T}                 # Average z_s gradient per epoch
    
    # Gradient tracking for θ parameter 
    individual_gradients_θ::Vector{T}              # θ gradient for each window
    batch_gradients_θ::Vector{T}                   # Average θ gradient per batch
    epoch_gradients_θ::Vector{T}                   # Average θ gradient per epoch
    
    # Variance/chaos tracking (to quantify how chaotic individual gradients are)
    batch_gradient_std_σ::Vector{T}                # Standard deviation of σ gradients within each batch
    batch_gradient_std_ρ::Vector{T}                # Standard deviation of ρ gradients within each batch  
    batch_gradient_std_β::Vector{T}                # Standard deviation of β gradients within each batch
    batch_gradient_std_x_s::Vector{T}              # Standard deviation of x_s gradients within each batch
    batch_gradient_std_y_s::Vector{T}              # Standard deviation of y_s gradients within each batch
    batch_gradient_std_z_s::Vector{T}              # Standard deviation of z_s gradients within each batch
    batch_gradient_std_θ::Vector{T}                # Standard deviation of θ gradients within each batch
    
    # First gradient of each epoch (to track initial chaos)
    first_gradient_per_epoch_x_s::Vector{T}        # First individual x_s gradient of each epoch
    
    # Additional metadata
    batch_indices::Vector{Int}                     # Which batch each individual measurement belongs to
    window_indices::Vector{Int}                    # Window start index for each measurement
    
    function TrainingMetrics{T}() where {T <: Real}
        new{T}(T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], 
               T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], 
               T[], T[], T[], T[], T[], T[], T[], T[], Int[], Int[])
    end
end

TrainingMetrics() = TrainingMetrics{Float64}()

"""
    reset_metrics!(metrics::TrainingMetrics)

Clear all stored metrics, useful for starting fresh training runs.
"""
function reset_metrics!(metrics::TrainingMetrics)
    empty!(metrics.individual_losses)
    empty!(metrics.batch_losses)
    empty!(metrics.epoch_losses)
    empty!(metrics.individual_gradients_σ)
    empty!(metrics.batch_gradients_σ)
    empty!(metrics.epoch_gradients_σ)
    empty!(metrics.individual_gradients_ρ)
    empty!(metrics.batch_gradients_ρ)
    empty!(metrics.epoch_gradients_ρ)
    empty!(metrics.individual_gradients_β)
    empty!(metrics.batch_gradients_β)
    empty!(metrics.epoch_gradients_β)
    empty!(metrics.individual_gradients_x_s)
    empty!(metrics.batch_gradients_x_s)
    empty!(metrics.epoch_gradients_x_s)
    empty!(metrics.individual_gradients_y_s)
    empty!(metrics.batch_gradients_y_s)
    empty!(metrics.epoch_gradients_y_s)
    empty!(metrics.individual_gradients_z_s)
    empty!(metrics.batch_gradients_z_s)
    empty!(metrics.epoch_gradients_z_s)
    empty!(metrics.individual_gradients_θ)
    empty!(metrics.batch_gradients_θ)
    empty!(metrics.epoch_gradients_θ)
    empty!(metrics.batch_gradient_std_σ)
    empty!(metrics.batch_gradient_std_ρ)
    empty!(metrics.batch_gradient_std_β)
    empty!(metrics.batch_gradient_std_x_s)
    empty!(metrics.batch_gradient_std_y_s)
    empty!(metrics.batch_gradient_std_z_s)
    empty!(metrics.batch_gradient_std_θ)
    empty!(metrics.first_gradient_per_epoch_x_s)
    empty!(metrics.batch_indices)
    empty!(metrics.window_indices)
    return metrics
end

"""
    record_window_metrics!(metrics::TrainingMetrics, loss_val, gradients::L63Parameters, 
                          batch_idx::Int, window_start::Int)

Record loss and gradient values for an individual window.
"""
function record_window_metrics!(metrics::TrainingMetrics, loss_val, gradients::L63Parameters, 
                                batch_idx::Int, window_start::Int)
    push!(metrics.individual_losses, loss_val)
    push!(metrics.individual_gradients_σ, gradients.σ)
    push!(metrics.individual_gradients_ρ, gradients.ρ)
    push!(metrics.individual_gradients_β, gradients.β)
    push!(metrics.individual_gradients_x_s, gradients.x_s)
    push!(metrics.individual_gradients_y_s, gradients.y_s)
    push!(metrics.individual_gradients_z_s, gradients.z_s)
    push!(metrics.individual_gradients_θ, gradients.θ)
    push!(metrics.batch_indices, batch_idx)
    push!(metrics.window_indices, window_start)
    return metrics
end

"""
    record_batch_metrics!(metrics::TrainingMetrics, batch_loss, avg_gradients::L63Parameters, 
                         batch_individual_gradients::AbstractVector{<:L63Parameters})

Record averaged loss and gradients for a batch, plus compute variance to track chaos.
"""
function record_batch_metrics!(metrics::TrainingMetrics, batch_loss, avg_gradients::L63Parameters, 
                               batch_individual_gradients::AbstractVector{<:L63Parameters})
    push!(metrics.batch_losses, batch_loss)
    push!(metrics.batch_gradients_σ, avg_gradients.σ)
    push!(metrics.batch_gradients_ρ, avg_gradients.ρ)
    push!(metrics.batch_gradients_β, avg_gradients.β)
    push!(metrics.batch_gradients_x_s, avg_gradients.x_s)
    push!(metrics.batch_gradients_y_s, avg_gradients.y_s)
    push!(metrics.batch_gradients_z_s, avg_gradients.z_s)
    push!(metrics.batch_gradients_θ, avg_gradients.θ)
    
    # Compute standard deviations within the batch to quantify chaos
    if length(batch_individual_gradients) > 1
        σ_grads = [g.σ for g in batch_individual_gradients]
        ρ_grads = [g.ρ for g in batch_individual_gradients]
        β_grads = [g.β for g in batch_individual_gradients]
        x_s_grads = [g.x_s for g in batch_individual_gradients]
        y_s_grads = [g.y_s for g in batch_individual_gradients]
        z_s_grads = [g.z_s for g in batch_individual_gradients]
        θ_grads = [g.θ for g in batch_individual_gradients]
        
        push!(metrics.batch_gradient_std_σ, std(σ_grads))
        push!(metrics.batch_gradient_std_ρ, std(ρ_grads))
        push!(metrics.batch_gradient_std_β, std(β_grads))
        push!(metrics.batch_gradient_std_x_s, std(x_s_grads))
        push!(metrics.batch_gradient_std_y_s, std(y_s_grads))
        push!(metrics.batch_gradient_std_z_s, std(z_s_grads))
        push!(metrics.batch_gradient_std_θ, std(θ_grads))
    else
        # Single gradient in batch - no variance
        push!(metrics.batch_gradient_std_σ, 0.0)
        push!(metrics.batch_gradient_std_ρ, 0.0)
        push!(metrics.batch_gradient_std_β, 0.0)
        push!(metrics.batch_gradient_std_x_s, 0.0)
        push!(metrics.batch_gradient_std_y_s, 0.0)
        push!(metrics.batch_gradient_std_z_s, 0.0)
        push!(metrics.batch_gradient_std_θ, 0.0)
    end
    
    return metrics
end

# Backward compatibility: if no individual gradients provided, skip variance tracking
function record_batch_metrics!(metrics::TrainingMetrics, batch_loss, avg_gradients::L63Parameters)
    push!(metrics.batch_losses, batch_loss)
    push!(metrics.batch_gradients_σ, avg_gradients.σ)
    push!(metrics.batch_gradients_ρ, avg_gradients.ρ)
    push!(metrics.batch_gradients_β, avg_gradients.β)
    push!(metrics.batch_gradients_x_s, avg_gradients.x_s)
    push!(metrics.batch_gradients_y_s, avg_gradients.y_s)
    push!(metrics.batch_gradients_z_s, avg_gradients.z_s)
    push!(metrics.batch_gradients_θ, avg_gradients.θ)
    
    # Add zeros for variance (no individual gradients to compute variance from)
    push!(metrics.batch_gradient_std_σ, 0.0)
    push!(metrics.batch_gradient_std_ρ, 0.0)
    push!(metrics.batch_gradient_std_β, 0.0)
    push!(metrics.batch_gradient_std_x_s, 0.0)
    push!(metrics.batch_gradient_std_y_s, 0.0)
    push!(metrics.batch_gradient_std_z_s, 0.0)
    push!(metrics.batch_gradient_std_θ, 0.0)
    
    return metrics
end

"""
    record_epoch_metrics!(metrics::TrainingMetrics, epoch_loss, avg_gradients::L63Parameters)

Record averaged loss and gradients for an epoch.
"""
function record_epoch_metrics!(metrics::TrainingMetrics, epoch_loss, avg_gradients::L63Parameters)
    push!(metrics.epoch_losses, epoch_loss)
    push!(metrics.epoch_gradients_σ, avg_gradients.σ)
    push!(metrics.epoch_gradients_ρ, avg_gradients.ρ)
    push!(metrics.epoch_gradients_β, avg_gradients.β)
    push!(metrics.epoch_gradients_x_s, avg_gradients.x_s)
    push!(metrics.epoch_gradients_y_s, avg_gradients.y_s)
    push!(metrics.epoch_gradients_z_s, avg_gradients.z_s)
    push!(metrics.epoch_gradients_θ, avg_gradients.θ)
    return metrics
end

"""
    record_first_gradient_of_epoch!(metrics::TrainingMetrics, first_gradient::L63Parameters)

Record the first gradient computed in each epoch to track initial chaos level.
"""
function record_first_gradient_of_epoch!(metrics::TrainingMetrics, first_gradient::L63Parameters)
    push!(metrics.first_gradient_per_epoch_x_s, first_gradient.x_s)
    return metrics
end

"""
    compute_gradient_statistics(metrics::TrainingMetrics, batch_idx::Int)

Compute statistics about gradient variability within a batch to demonstrate
chaotic individual behavior vs meaningful averages.
"""
function compute_gradient_statistics(metrics::TrainingMetrics, batch_idx::Int)
    # Find indices for this batch
    batch_mask = metrics.batch_indices .== batch_idx
    
    if !any(batch_mask)
        return nothing
    end
    
    # Extract gradients for this batch
    σ_grads = metrics.individual_gradients_σ[batch_mask]
    ρ_grads = metrics.individual_gradients_ρ[batch_mask]
    β_grads = metrics.individual_gradients_β[batch_mask]
    x_s_grads = metrics.individual_gradients_x_s[batch_mask]
    y_s_grads = metrics.individual_gradients_y_s[batch_mask]
    z_s_grads = metrics.individual_gradients_z_s[batch_mask]
    θ_grads = metrics.individual_gradients_θ[batch_mask]
    
    # Compute statistics
    stats = (
        σ = (mean = mean(σ_grads), std = std(σ_grads), min = minimum(σ_grads), max = maximum(σ_grads)),
        ρ = (mean = mean(ρ_grads), std = std(ρ_grads), min = minimum(ρ_grads), max = maximum(ρ_grads)),
        β = (mean = mean(β_grads), std = std(β_grads), min = minimum(β_grads), max = maximum(β_grads)),
        x_s = (mean = mean(x_s_grads), std = std(x_s_grads), min = minimum(x_s_grads), max = maximum(x_s_grads)),
        y_s = (mean = mean(y_s_grads), std = std(y_s_grads), min = minimum(y_s_grads), max = maximum(y_s_grads)),
        z_s = (mean = mean(z_s_grads), std = std(z_s_grads), min = minimum(z_s_grads), max = maximum(z_s_grads)),
        θ = (mean = mean(θ_grads), std = std(θ_grads), min = minimum(θ_grads), max = maximum(θ_grads)),
        count = sum(batch_mask)
    )
    
    return stats
end

"""
    analyze_chaos_to_signal_conversion(metrics::TrainingMetrics; parameter=:x_s)

Analyze how individual chaotic gradients are converted to meaningful signal through averaging.
This demonstrates Milan's key insight about chaos in parameter estimation.

Returns a detailed analysis of:
- Individual gradient chaos (high variance)
- Batch averaging signal extraction (reduced variance)
- Epoch averaging optimization signal (clean signal)
- First gradients per epoch (tracking chaos level over training)
"""
function analyze_chaos_to_signal_conversion(metrics::TrainingMetrics; parameter=:x_s)
    # Select the gradient arrays based on parameter
    individual_grads, batch_grads, epoch_grads, batch_stds, first_grads = if parameter == :x_s
        (metrics.individual_gradients_x_s, metrics.batch_gradients_x_s, 
         metrics.epoch_gradients_x_s, metrics.batch_gradient_std_x_s, 
         metrics.first_gradient_per_epoch_x_s)
    elseif parameter == :σ
        (metrics.individual_gradients_σ, metrics.batch_gradients_σ, 
         metrics.epoch_gradients_σ, metrics.batch_gradient_std_σ, Float64[])
    elseif parameter == :ρ
        (metrics.individual_gradients_ρ, metrics.batch_gradients_ρ, 
         metrics.epoch_gradients_ρ, metrics.batch_gradient_std_ρ, Float64[])
    elseif parameter == :β
        (metrics.individual_gradients_β, metrics.batch_gradients_β, 
         metrics.epoch_gradients_β, metrics.batch_gradient_std_β, Float64[])
    else
        error("Parameter $parameter not supported. Use :x_s, :σ, :ρ, or :β")
    end
    
    if isempty(individual_grads)
        return nothing
    end
    
    # Compute chaos metrics
    ind_mean = mean(individual_grads)
    ind_std = std(individual_grads)
    ind_cv = abs(ind_std / ind_mean) * 100  # Coefficient of variation as chaos measure
    
    # Batch averaging metrics
    batch_mean = isempty(batch_grads) ? NaN : mean(batch_grads)
    batch_std = isempty(batch_grads) ? NaN : std(batch_grads)
    batch_cv = isempty(batch_grads) ? NaN : abs(batch_std / batch_mean) * 100
    
    # Epoch averaging metrics  
    epoch_mean = isempty(epoch_grads) ? NaN : mean(epoch_grads)
    epoch_std = isempty(epoch_grads) ? NaN : std(epoch_grads)
    epoch_cv = isempty(epoch_grads) ? NaN : abs(epoch_std / epoch_mean) * 100
    
    # Average batch-level standard deviation (measure of within-batch chaos)
    avg_batch_std = isempty(batch_stds) ? NaN : mean(batch_stds)
    
    # First gradient analysis (chaos evolution over training)
    first_grad_std = isempty(first_grads) ? NaN : std(first_grads)
    first_grad_trend = if length(first_grads) > 5
        # Simple linear trend: is chaos decreasing over training?
        epochs = 1:length(first_grads)
        corr_coef = cor(epochs, abs.(first_grads))
        corr_coef
    else
        NaN
    end
    
    # Signal-to-noise improvements
    chaos_reduction_batch = isfinite(batch_cv) ? ind_cv - batch_cv : NaN
    chaos_reduction_epoch = isfinite(epoch_cv) ? ind_cv - epoch_cv : NaN
    snr_improvement_batch = isfinite(batch_cv) && batch_cv > 0 ? ind_cv / batch_cv : NaN
    snr_improvement_epoch = isfinite(epoch_cv) && epoch_cv > 0 ? ind_cv / epoch_cv : NaN
    
    return (
        parameter = parameter,
        
        # Raw chaos metrics
        individual = (mean = ind_mean, std = ind_std, cv = ind_cv, count = length(individual_grads)),
        batch = (mean = batch_mean, std = batch_std, cv = batch_cv, count = length(batch_grads)),
        epoch = (mean = epoch_mean, std = epoch_std, cv = epoch_cv, count = length(epoch_grads)),
        
        # Within-batch chaos
        avg_within_batch_std = avg_batch_std,
        
        # Training evolution
        first_gradients = (std = first_grad_std, trend_correlation = first_grad_trend, count = length(first_grads)),
        
        # Key insights for Milan's research
        chaos_to_signal = (
            batch_chaos_reduction_percent = chaos_reduction_batch,
            epoch_chaos_reduction_percent = chaos_reduction_epoch, 
            batch_snr_improvement = snr_improvement_batch,
            epoch_snr_improvement = snr_improvement_epoch,
            averaging_effectiveness = !isnan(snr_improvement_batch) && snr_improvement_batch > 1.5
        )
    )
end

# ================================ Enzyme-compatible helper functions ================================
# These functions are defined at module level to avoid scoping issues with Enzyme


"""
    mae_enzyme_function(σ, ρ, β, u0, target_trajectory, window_length, dt)

Compute the **mean absolute error (MAE)** between a predicted Lorenz-63 trajectory 
(obtained by integrating with RK4 for `window_length` steps) and a given target trajectory. 

- Parameters (`σ`, `ρ`, `β`) define the Lorenz-63 system.  
- `u0` is the initial 3D state vector.  
- `target_trajectory` is an N×3 matrix of reference states.  
- `window_length` is the number of time steps to integrate.  
- `dt` is the integration step size.  

The absolute error is smoothed with `sqrt(x^2 + ε)` to ensure differentiability for Enzyme AD.  
Returns the average per-component error over the window.
"""

function mae_enzyme_function(
    σ::T,                                                   # Parameter σ
    ρ::T,                                                   # Parameter ρ
    β::T,                                                   # Parameter β
    u0::AbstractVector{T},                                  # Initial state (3-vector)    
    target_trajectory::AbstractMatrix{T},                   # Target trajectory (N×3 matrix)
    window_length::Int,                                     # Number of integration steps
    dt::T                                                   # Time step
    ) where {T} 

    # Construct parameters and initial state
    params = L63Parameters(σ, ρ, β)                      # Construct parameters
    u = similar(u0)                                         # Initial state with same backend
    u .= u0                                                 # Copy initial state

    ae = zero(T)                                            # Absolute error accumulator
    count = 0                                               # Count of comparisons
    epsilon = T(1e-6)                                       # Small value for smooth approximation
    
    @inbounds for i in 1:window_length                      # Integrate forward
        u = rk4_step(u, params, dt)                         # One RK4 step
        for j in 1:3                                        # Compare to target at row i+1
            diff = u[j] - target_trajectory[i+1, j]         # Difference
            # Use smooth approximation: sqrt(x^2 + ε) ≈ |x| but differentiable everywhere
            ae += sqrt(diff * diff + epsilon)               # Smooth absolute value
            count += 1                                      # Increment count
        end
    end
    
    return ae / count
end

"""
    adaptive_enzyme_function(σ, ρ, β, u0, target_trajectory, window_length, dt)

Compute an **adaptive robust loss (Huber-like)** between a predicted Lorenz-63 trajectory 
and a given target trajectory.  

- For small errors (|diff| ≤ δ): behaves like scaled MSE (0.5 * diff² / δ).  
- For large errors (|diff| > δ): behaves like MAE (|diff| − 0.5δ).  
- Uses a smooth approximation of |diff| to remain differentiable for Enzyme AD.  

Returns the mean loss across all time steps and state components.
"""

function adaptive_enzyme_function(σ::T, ρ::T, β::T, u0::AbstractVector{T}, 
                                 target_trajectory::AbstractMatrix{T}, 
                                 window_length::Int, dt::T) where {T}
    params = L63Parameters(σ, ρ, β)
    u = similar(u0)
    u .= u0
    
    total_loss = zero(T)
    count = 0
    δ = T(1.0)  # Huber loss threshold
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            
            # Smooth Huber loss that's differentiable everywhere
            # For |diff| ≤ δ: 0.5 * diff^2 / δ
            # For |diff| > δ: |diff| - 0.5 * δ
            # But use smooth approximation for |diff|
            squared_diff = diff * diff
            sqrt_term = sqrt(squared_diff + T(1e-8))  # Smooth approximation of |diff|
            
            if squared_diff <= δ * δ
                loss_contribution = T(0.5) * squared_diff / δ
            else
                loss_contribution = sqrt_term - T(0.5) * δ
            end
            
            total_loss += loss_contribution
            count += 1
        end
    end
    
    return total_loss / count
end

"""
    mse_enzyme_function(σ, ρ, β, u0, target_trajectory, window_length, dt)

Compute the **mean squared error (MSE)** between a predicted Lorenz-63 trajectory 
(obtained by RK4 integration) and a given target trajectory.  

- Parameters (`σ`, `ρ`, `β`) define the Lorenz-63 system.  
- `u0` is the initial 3D state vector.  
- `target_trajectory` is an N×3 matrix of reference states.  
- `window_length` is the number of integration steps.  
- `dt` is the integration step size.  

Returns the average squared error per component over the integration window.
"""

function mse_enzyme_function(σ::T, ρ::T, β::T, u0::AbstractVector{T}, 
                            target_trajectory::AbstractMatrix{T}, 
                            window_length::Int, dt::T) where {T}
    params = L63Parameters(σ, ρ, β)
    u = similar(u0)
    u .= u0
    
    se = zero(T)
    count = 0
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            se += diff * diff
            count += 1
        end
    end
    
    return se / count
end

# ================================ User-facing loss functions ================================

# These functions operate on full trajectory matrices and can be passed to modular_train!()
# They can also be used directly by users for custom training loops.
# All take (predicted::Matrix, target::Matrix) -> Real

# User specifies: loss_function = window_rmse
#                 ↓
# Training creates: predicted_matrix, target_matrix  
#                 ↓
# Forward pass: window_rmse(predicted_matrix, target_matrix) → scalar
#                 ↓
# Gradient pass: Uses loss_function_enzyme(σ,ρ,β,...) for Enzyme.autodiff
#                 ↓
# Result: Gradients w.r.t. parameters

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
    u = similar(target_solution.u0)
    u .= target_solution[window_start]
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
    u0::AbstractVector{T},          # Initial state (3-vector)
    target_trajectory::AbstractMatrix{T},   # Target trajectory (N×3 matrix)
    window_length::Int,             # Number of integration steps
    dt::T
    ) where {T}
    
    params = L63Parameters(σ, ρ, β) # Construct parameters with defaults for extended fields
    u = similar(u0) # Initial state with same backend
    u .= u0
    
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
    compute_gradients_with_tracking(params::L63Parameters, target_solution::L63Solution, 
                                   window_start::Int, window_length::Int, 
                                   loss_function::Function = window_rmse;
                                   metrics::Union{TrainingMetrics, Nothing} = nothing,
                                   batch_idx::Int = 0)

Compute gradients with optional tracking for detailed analysis.
If metrics is provided, records individual window loss and gradients.
"""
function compute_gradients_with_tracking(params::L63Parameters{T}, target_solution::L63Solution{T},
                                        window_start::Int, window_length::Int, 
                                        loss_function::Function = window_rmse;
                                        metrics::Union{TrainingMetrics, Nothing} = nothing,
                                        batch_idx::Int = 0) where {T}
    
    # Use the extended gradients function to handle all 7 parameters
    loss_val, gradients = compute_gradients_extended(params, target_solution, window_start, window_length, loss_function)
    
    # Record metrics if requested
    if metrics !== nothing
        record_window_metrics!(metrics, loss_val, gradients, batch_idx, window_start)
    end
    
    return loss_val, gradients
end

"""
    compute_gradients(params::L63Parameters, target_solution::L63Solution, 
                     window_start::Int, window_length::Int, loss_function::Function = window_rmse)

Compute gradients of any loss function with respect to parameters using Enzyme AD.

# Arguments
- `params`: Parameters to evaluate
- `target_solution`: Target trajectory
- `window_start`: Starting index in target trajectory
- `window_length`: Number of integration steps  
- `loss_function`: Loss function that takes (predicted::Matrix, target::Matrix) -> Real

# Returns
- `(loss_value, gradients)` where gradients is an L63Parameters object
"""
function compute_gradients(params::L63Parameters{T}, target_solution::L63Solution{T},
                          window_start::Int, window_length::Int, 
                          loss_function::Function = window_rmse) where {T}
    
    # Use the modular version which already handles custom loss functions correctly
    return compute_gradients_modular(params, target_solution, window_start, window_length, loss_function)
end

"""
    compute_gradients_modular(params::L63Parameters, target_solution::L63Solution, 
                             window_start::Int, window_length::Int, loss_function::Function)

Compute gradients of any loss function with respect to parameters using Enzyme AD.
This is a modular version that can work with different loss functions.

# Arguments
- `params`: Parameters to evaluate
- `target_solution`: Target trajectory
- `window_start`: Starting index in target trajectory
- `window_length`: Number of integration steps  
- `loss_function`: Loss function that takes (predicted::Matrix, target::Matrix) -> Real

# Returns
- `(loss_value, gradients)` where gradients is an L63Parameters object
"""
function compute_gradients_modular(
    params::L63Parameters{T},                   # Parameters to evaluate
    target_solution::L63Solution{T},            # Target trajectory
    window_start::Int, window_length::Int,      # Number of integration steps
    loss_function::Function                     # Loss function to use
    ) where {T}
    
    # Extract window data
    u0 = target_solution[window_start]         # Initial state at window start
    window_end = window_start + window_length       # End index (exclusive)
    target_window = target_solution.u[window_start:window_end, :] # Target states for the window
    dt = target_solution.system.dt # Time step
    
    # Convert to compatible types
    σ0, ρ0, β0 = T(params.σ), T(params.ρ), T(params.β)  # Ensure parameters are of type T
    u0_vec = u0 isa Vector ? u0 : Vector{T}(u0) # Ensure u0 is a Vector{T}
    target_mat = target_window isa Matrix ? target_window : Matrix{T}(target_window) # Ensure target is Matrix{T}
    
    # For now, use specific loss function implementations that Enzyme can handle
    if loss_function === window_rmse
        # Use the proven working enzyme function for RMSE
        grads = Enzyme.autodiff(
            Enzyme.Reverse,                         # Reverse mode AD
            Enzyme.Const(loss_function_enzyme),     # Function to differentiate
            Enzyme.Active(σ0),                      # Active variable
            Enzyme.Active(ρ0),                      # Active variable
            Enzyme.Active(β0),                      # Active variable
            Enzyme.Const(u0_vec),                   # Constant input
            Enzyme.Const(target_mat),               # Constant input    
            Enzyme.Const(window_length),            # Constant input
            Enzyme.Const(dt)                        # Constant input
        )
        
        G = grads[1]
        gσ, gρ, gβ = G[1], G[2], G[3]
        loss_val = loss_function_enzyme(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters(gσ, gρ, gβ)
        
    elseif loss_function === window_mae
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(mae_enzyme_function),
            Enzyme.Active(σ0),
            Enzyme.Active(ρ0), 
            Enzyme.Active(β0),
            Enzyme.Const(u0_vec),
            Enzyme.Const(target_mat),
            Enzyme.Const(window_length),
            Enzyme.Const(dt)
        )
        
        # Extract gradients correctly - grads[1] is the tuple of derivatives
        gσ, gρ, gβ = grads[1][1], grads[1][2], grads[1][3]
        loss_val = mae_enzyme_function(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters(gσ, gρ, gβ)
        
    elseif loss_function === window_mse
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(mse_enzyme_function),
            Enzyme.Active(σ0),
            Enzyme.Active(ρ0), 
            Enzyme.Active(β0),
            Enzyme.Const(u0_vec),
            Enzyme.Const(target_mat),
            Enzyme.Const(window_length),
            Enzyme.Const(dt)
        )
        
        # Extract gradients correctly - grads[1] is the tuple of derivatives
        gσ, gρ, gβ = grads[1][1], grads[1][2], grads[1][3]
        loss_val = mse_enzyme_function(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters(gσ, gρ, gβ)
        
    elseif loss_function === adaptive_loss
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(adaptive_enzyme_function),
            Enzyme.Active(σ0),
            Enzyme.Active(ρ0), 
            Enzyme.Active(β0),
            Enzyme.Const(u0_vec),
            Enzyme.Const(target_mat),
            Enzyme.Const(window_length),
            Enzyme.Const(dt)
        )
        
        # Extract gradients correctly - grads[1] is the tuple of derivatives
        gσ, gρ, gβ = grads[1][1], grads[1][2], grads[1][3]
        loss_val = adaptive_enzyme_function(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters(gσ, gρ, gβ)
        
    else
        # For other loss functions, fall back to the old approach for now
        # TODO: Add more loss function implementations as needed
        @warn "Loss function $(nameof(loss_function)) not yet optimized for Enzyme. Using fallback (may have zero gradients)."
        
        # Create enzyme-compatible wrapper for the given loss function
        function enzyme_loss_wrapper(σ::T, ρ::T, β::T, u0::AbstractVector{T}, 
                                   target_trajectory::AbstractMatrix{T}, 
                                   window_length::Int, dt::T) where {T}
            
            params = L63Parameters(σ, ρ, β)
            u = similar(u0)
            u .= u0
            
            # Integrate forward and collect predicted trajectory
            predicted_trajectory = Matrix{T}(undef, window_length, 3)
            
            @inbounds for i in 1:window_length 
                u = rk4_step(u, params, dt)
                predicted_trajectory[i, :] .= u
            end
            
            # Apply the user-provided loss function inline for Enzyme compatibility
            # Note: target_trajectory includes initial state, so we skip the first row
            target_for_loss = @view target_trajectory[2:end, :]  # Skip initial state
            
            # Call the loss function with proper views to ensure Enzyme can track derivatives
            return loss_function(predicted_trajectory, target_for_loss)
        end
        
        # Compute gradients using Enzyme
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(enzyme_loss_wrapper),
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
        
        loss_val = enzyme_loss_wrapper(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters(gσ, gρ, gβ)
    end
end

# ================================ Extended Enzyme Functions for 7 Parameters ================================

"""
    loss_function_enzyme_extended(σ, ρ, β, x_s, y_s, z_s, θ, u0, target_trajectory, window_length, dt)

Extended enzyme-compatible wrapper for loss computation with all 7 parameters.
"""
@inline function loss_function_enzyme_extended(
    σ::T, ρ::T, β::T, x_s::T, y_s::T, z_s::T, θ::T,    # All 7 parameters
    u0::AbstractVector{T},                               # Initial state (3-vector)
    target_trajectory::AbstractMatrix{T},                # Target trajectory (N×3 matrix)
    window_length::Int,                                  # Number of integration steps
    dt::T
    ) where {T}
    
    params = L63Parameters{T}(σ, ρ, β, x_s, y_s, z_s, θ)  # Construct extended parameters
    u = similar(u0)                                       # Initial state with same backend
    u .= u0
    
    se = zero(T)                                          # Squared error accumulator
    count = 0                                             # Count of comparisons
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        
        # Compare to target at row i+1 (since target includes starting state)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            se += diff * diff
            count += 1
        end
    end
    
    return sqrt(se / count)  # RMSE
end

"""
    mae_enzyme_function_extended(σ, ρ, β, x_s, y_s, z_s, θ, u0, target_trajectory, window_length, dt)

Extended MAE enzyme function with all 7 parameters.
"""
@inline function mae_enzyme_function_extended(
    σ::T, ρ::T, β::T, x_s::T, y_s::T, z_s::T, θ::T,    # All 7 parameters
    u0::AbstractVector{T},                               # Initial state (3-vector)
    target_trajectory::AbstractMatrix{T},                # Target trajectory (N×3 matrix)
    window_length::Int,                                  # Number of integration steps
    dt::T
    ) where {T}
    
    params = L63Parameters{T}(σ, ρ, β, x_s, y_s, z_s, θ)  # Construct extended parameters
    u = similar(u0)                                       # Initial state with same backend
    u .= u0
    
    ae = zero(T)                                          # Absolute error accumulator
    count = 0                                             # Count of comparisons
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        
        # Compare to target at row i+1 (since target includes starting state)
        for j in 1:3
            diff = abs(u[j] - target_trajectory[i+1, j])
            ae += diff
            count += 1
        end
    end
    
    return ae / count  # MAE
end

"""
    mse_enzyme_function_extended(σ, ρ, β, x_s, y_s, z_s, θ, u0, target_trajectory, window_length, dt)

Extended MSE enzyme function with all 7 parameters.
"""
@inline function mse_enzyme_function_extended(
    σ::T, ρ::T, β::T, x_s::T, y_s::T, z_s::T, θ::T,    # All 7 parameters
    u0::AbstractVector{T},                               # Initial state (3-vector)
    target_trajectory::AbstractMatrix{T},                # Target trajectory (N×3 matrix)
    window_length::Int,                                  # Number of integration steps
    dt::T
    ) where {T}
    
    params = L63Parameters{T}(σ, ρ, β, x_s, y_s, z_s, θ)  # Construct extended parameters
    u = similar(u0)                                       # Initial state with same backend
    u .= u0
    
    se = zero(T)                                          # Squared error accumulator
    count = 0                                             # Count of comparisons
    
    @inbounds for i in 1:window_length 
        u = rk4_step(u, params, dt)
        
        # Compare to target at row i+1 (since target includes starting state)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            se += diff * diff
            count += 1
        end
    end
    
    return se / count  # MSE
end

"""
    compute_gradients_extended(params::L63Parameters, target_solution::L63Solution, 
                              window_start::Int, window_length::Int, loss_function::Function)

Compute gradients of any loss function with respect to all 7 extended parameters using Enzyme AD.

# Arguments
- `params`: Extended parameters to evaluate (σ, ρ, β, x_s, y_s, z_s, θ)
- `target_solution`: Target trajectory
- `window_start`: Starting index in target trajectory
- `window_length`: Number of integration steps  
- `loss_function`: Loss function that takes (predicted::Matrix, target::Matrix) -> Real

# Returns
- `(loss_value, gradients)` where gradients is an L63Parameters object with all 7 gradients
"""
function compute_gradients_extended(
    params::L63Parameters{T},                   # Extended parameters to evaluate
    target_solution::L63Solution{T},            # Target trajectory
    window_start::Int, window_length::Int,      # Number of integration steps
    loss_function::Function                     # Loss function to use
    ) where {T}
    
    # Extract window data
    u0 = target_solution[window_start]                          # Initial state at window start
    window_end = window_start + window_length                   # End index (exclusive)
    target_window = target_solution.u[window_start:window_end, :] # Target states for the window
    dt = target_solution.system.dt                             # Time step
    
    # Convert to compatible types
    σ0, ρ0, β0 = T(params.σ), T(params.ρ), T(params.β)         # Classic parameters
    x_s0, y_s0, z_s0, θ0 = T(params.x_s), T(params.y_s), T(params.z_s), T(params.θ)  # Extended parameters
    u0_vec = u0 isa Vector ? u0 : Vector{T}(u0)                # Ensure u0 is a Vector{T}
    target_mat = target_window isa Matrix ? target_window : Matrix{T}(target_window) # Ensure target is Matrix{T}
    
    # For now, use specific loss function implementations that Enzyme can handle
    if loss_function === window_rmse
        # Use the extended enzyme function for RMSE
        grads = Enzyme.autodiff(
            Enzyme.Reverse,                                     # Reverse mode AD
            Enzyme.Const(loss_function_enzyme_extended),        # Function to differentiate
            Enzyme.Active(σ0),                                  # Active variable
            Enzyme.Active(ρ0),                                  # Active variable
            Enzyme.Active(β0),                                  # Active variable
            Enzyme.Active(x_s0),                                # Active variable
            Enzyme.Active(y_s0),                                # Active variable
            Enzyme.Active(z_s0),                                # Active variable
            Enzyme.Active(θ0),                                  # Active variable
            Enzyme.Const(u0_vec),                               # Constant input
            Enzyme.Const(target_mat),                           # Constant input    
            Enzyme.Const(window_length),                        # Constant input
            Enzyme.Const(dt)                                    # Constant input
        )
        
        G = grads[1]
        gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ = G[1], G[2], G[3], G[4], G[5], G[6], G[7]
        loss_val = loss_function_enzyme_extended(σ0, ρ0, β0, x_s0, y_s0, z_s0, θ0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters{T}(gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ)
        
    elseif loss_function === window_mae
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(mae_enzyme_function_extended),
            Enzyme.Active(σ0), Enzyme.Active(ρ0), Enzyme.Active(β0),
            Enzyme.Active(x_s0), Enzyme.Active(y_s0), Enzyme.Active(z_s0), Enzyme.Active(θ0),
            Enzyme.Const(u0_vec), Enzyme.Const(target_mat),
            Enzyme.Const(window_length), Enzyme.Const(dt)
        )
        
        G = grads[1]
        gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ = G[1], G[2], G[3], G[4], G[5], G[6], G[7]
        loss_val = mae_enzyme_function_extended(σ0, ρ0, β0, x_s0, y_s0, z_s0, θ0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters{T}(gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ)
        
    elseif loss_function === window_mse
        grads = Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(mse_enzyme_function_extended),
            Enzyme.Active(σ0), Enzyme.Active(ρ0), Enzyme.Active(β0),
            Enzyme.Active(x_s0), Enzyme.Active(y_s0), Enzyme.Active(z_s0), Enzyme.Active(θ0),
            Enzyme.Const(u0_vec), Enzyme.Const(target_mat),
            Enzyme.Const(window_length), Enzyme.Const(dt)
        )
        
        G = grads[1]
        gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ = G[1], G[2], G[3], G[4], G[5], G[6], G[7]
        loss_val = mse_enzyme_function_extended(σ0, ρ0, β0, x_s0, y_s0, z_s0, θ0, u0_vec, target_mat, window_length, dt)
        
        return loss_val, L63Parameters{T}(gσ, gρ, gβ, gx_s, gy_s, gz_s, gθ)
        
    else
        # For other loss functions, use a dedicated wrapper that works with Enzyme
        # We need to handle this case more carefully for Enzyme compatibility
        error("Loss function $(nameof(loss_function)) is not yet supported with extended parameters. " *
              "Please use window_rmse, window_mae, or window_mse.")
    end
end
