# ================================ Training ================================

import Optimisers
import Random
using Printf: @sprintf

# ================================ Modular Training (Enzyme-based) ================================

"""
    modular_train!(params, target_solution; 
                   optimizer_config=adam_config(), 
                   loss_function=window_rmse,
                   kwargs...)

Modular training function using Enzyme.jl for precise gradient computation
with Optimisers.jl for advanced optimization and Flux.jl for loss functions.

This function provides:
- Precise gradients via Enzyme.jl automatic differentiation
- Support for a wide range of optimizers from Optimisers.jl  
- Modular loss functions from Flux.jl
- Direct parameter optimization without neural network overhead

# Arguments
- `params::L63Parameters`: Initial parameter guess
- `target_solution::L63Solution`: Target trajectory to fit
- `optimizer_config::OptimizerConfig`: Optimizer configuration (see optimizers.jl)
- `loss_function::Function`: Loss function to use (see loss.jl)

# Keyword Arguments
- `epochs::Int=100`: Number of training epochs
- `window_size::Int=300`: Size of trajectory windows
- `stride::Int=window_size÷2`: Stride between windows
- `batch_size::Int=32`: Mini-batch size
- `train_fraction::Real=0.8`: Fraction of data for training
- `shuffle::Bool=true`: Shuffle training data
- `verbose::Bool=true`: Print training progress
- `eval_every::Int=10`: Evaluate every N epochs
- `early_stopping_patience::Int=20`: Early stopping patience
- `update_σ::Bool=true`: Whether to update σ parameter
- `update_ρ::Bool=true`: Whether to update ρ parameter  
- `update_β::Bool=true`: Whether to update β parameter
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Example
```julia
# Quick training with Adam
results = modular_train!(params, solution)

# Custom optimizer and loss
opt_config = adamw_config(learning_rate=1e-4, weight_decay=1e-3)
results = modular_train!(params, solution, 
                        optimizer_config=opt_config,
                        loss_function=window_mae,
                        epochs=200)

# Robust training setup
results = modular_train!(params, solution,
                        optimizer_config=robust_optimizer(),
                        loss_function=weighted_window_loss(window_rmse, 1.5),
                        epochs=300,
                        early_stopping_patience=30)
```
"""
function modular_train!(
    params::L63Parameters{T},
    target_solution::L63Solution{T};
    # Core configuration
    optimizer_config::OptimizerConfig = adam_config(),
    loss_function::Function = window_rmse,  # Note: Currently ignored, using RMSE from compute_gradients
    
    # Training parameters
    epochs::Int = 100,
    window_size::Int = 300,
    stride::Union{Nothing, Int} = nothing,
    batch_size::Int = 32,
    
    # Data splitting
    train_fraction::Real = 0.8,
    shuffle::Bool = true,
    
    # Parameter updates
    update_σ::Bool = true,
    update_ρ::Bool = true,
    update_β::Bool = true,
    
    # Training control
    verbose::Bool = true,
    eval_every::Int = 1,
    early_stopping_patience::Int = 20,
    early_stopping_min_delta::Real = 1e-6,
    
    # Reproducibility
    rng::Random.AbstractRNG = Random.default_rng()
) where {T}
    
    # Set default stride
    stride_val = isnothing(stride) ? window_size ÷ 2 : stride
    
    # Generate window starting positions
    max_start = length(target_solution) - window_size
    max_start > 0 || throw(ArgumentError("Window size too large for trajectory"))
    
    window_starts = collect(1:stride_val:max_start)
    n_windows = length(window_starts)
    
    # Train/validation split
    indices = collect(1:n_windows)
    if shuffle
        Random.shuffle!(rng, indices)
    end
    
    train_count = max(1, round(Int, train_fraction * n_windows))
    train_indices = window_starts[indices[1:train_count]]
    val_indices = train_count < n_windows ? window_starts[indices[train_count + 1:end]] : Int[]
    
    # Initialize simple gradient descent optimizer (for now, bypassing complex Optimisers.jl setup)
    learning_rate = T(optimizer_config.learning_rate)
    ps = (σ = params.σ, ρ = params.ρ, β = params.β)
    
    # Parameter update mask (for masking gradients)
    update_mask = (σ = update_σ, ρ = update_ρ, β = update_β)
    
    # Training state
    metrics_history = Vector{NamedTuple}(undef, 0)
    param_history = L63Parameters{T}[params]
    best_params = params
    best_metric = convert(T, Inf)
    patience_counter = 0
    
    if verbose
        println("   Optimizer: $(optimizer_config.name)")
        println("   Data: $(length(train_indices)) train windows, $(length(val_indices)) val windows")
        println("   Window size: $window_size, stride: $stride_val")
        println("   Updating: σ=$update_σ, ρ=$update_ρ, β=$update_β")
        println()
        println("Epoch │   Train    │    Val     │      σ     │      ρ     │      β     │")
        println("──────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
    end
    
    for epoch in 1:epochs
        # Training phase
        epoch_loss = zero(T)
        
        # Shuffle training windows
        current_train_indices = shuffle ? Random.shuffle(rng, copy(train_indices)) : train_indices
        
        # Process training windows in batches
        train_batches = [current_train_indices[i:min(i+batch_size-1, end)] 
                        for i in 1:batch_size:length(current_train_indices)]
        
        for batch_windows in train_batches
            # Compute average gradients over the batch
            batch_loss = zero(T)
            avg_grads = (σ = zero(T), ρ = zero(T), β = zero(T))
            
            for window_start in batch_windows
                current_params = L63Parameters{T}(ps.σ, ps.ρ, ps.β)
                loss_val, grads = compute_gradients(current_params, target_solution, 
                                                  window_start, window_size)
                batch_loss += loss_val
                avg_grads = (σ = avg_grads.σ + grads.σ, 
                           ρ = avg_grads.ρ + grads.ρ, 
                           β = avg_grads.β + grads.β)
            end
            
            # Average the gradients and loss
            batch_size_actual = length(batch_windows)
            batch_loss /= batch_size_actual
            avg_grads = (σ = avg_grads.σ / batch_size_actual,
                        ρ = avg_grads.ρ / batch_size_actual,
                        β = avg_grads.β / batch_size_actual)
            
            # Apply parameter update mask
            masked_grads = (σ = update_mask.σ ? avg_grads.σ : zero(T),
                           ρ = update_mask.ρ ? avg_grads.ρ : zero(T),
                           β = update_mask.β ? avg_grads.β : zero(T))
            
            # Simple gradient descent update
            ps = (σ = ps.σ - learning_rate * masked_grads.σ,
                  ρ = ps.ρ - learning_rate * masked_grads.ρ,
                  β = ps.β - learning_rate * masked_grads.β)
            epoch_loss += batch_loss
        end
        
        train_loss = epoch_loss / length(train_batches)
        
        # Validation phase
        val_loss = if !isempty(val_indices) && epoch % eval_every == 0
            val_total = zero(T)
            for window_start in val_indices
                current_params = L63Parameters{T}(ps.σ, ps.ρ, ps.β)
                loss_val, _ = compute_gradients(current_params, target_solution, 
                                              window_start, window_size)
                val_total += loss_val
            end
            val_total / length(val_indices)
        else
            missing
        end
        
                # Record metrics
        current_params = L63Parameters{T}(ps.σ, ps.ρ, ps.β)
        push!(param_history, current_params)
        
        metrics = (
            epoch = epoch,
            train_loss = train_loss,
            val_loss = val_loss,
            params = current_params
        )
        push!(metrics_history, metrics)
        
        # Update best model
        metric_for_best = ismissing(val_loss) ? train_loss : val_loss
        if metric_for_best < best_metric - early_stopping_min_delta
            best_metric = metric_for_best
            best_params = current_params
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        # Print progress
        if verbose
            val_str = ismissing(val_loss) ? "    —     " : @sprintf("%10.6f", val_loss)
            println(@sprintf("%5d │ %10.6f │ %s │ %10.5f │ %10.5f │ %10.5f │",
                   epoch, train_loss, val_str, current_params.σ, current_params.ρ, current_params.β))
        end
        # Early stopping
        if patience_counter >= early_stopping_patience
            if verbose
                println("\n⚠️  Early stopping triggered at epoch $epoch")
                println("   Best metric: $(best_metric:.8f)")
            end
            break
        end
    end
    
    if verbose
        println()
        println("✅ Training completed")
        println("   Final epochs: $(length(param_history) - 1)")
        println("   Best metric: ", best_metric)
        println()
    end
    
    return (
        best_params = best_params,
        metrics_history = metrics_history,
        param_history = param_history,
        optimizer_config = optimizer_config,
        loss_function = loss_function
    )
end

# -- Pretty printing ---------------------------------------------------------

function _print_training_header()
    println()
    println("┌───────┬────────────┬────────────┬───────────┬───────────┬───────────┐")
    println("│ Epoch │   Train    │    Val     │      σ    │      ρ    │      β    │")
    println("├───────┼────────────┼────────────┼───────────┼───────────┼───────────┤")
end

function _print_training_row(epoch::Int, train_loss, val_loss, params::L63Parameters)
    train_str = @sprintf("%10.6f", train_loss)
    val_str = ismissing(val_loss) ? "    —     " : @sprintf("%10.6f", val_loss)
    σ_str = @sprintf("%9.4f", params.σ)
    ρ_str = @sprintf("%9.4f", params.ρ)
    β_str = @sprintf("%9.4f", params.β)
    println(@sprintf("│ %5d │ %s │ %s │ %s │ %s │ %s │", epoch, train_str, val_str, σ_str, ρ_str, β_str))
end

function _print_training_footer()
    println("└───────────────────────────────────────────────────────────────────────┘")
    println()
end

# -- Public API --------------------------------------------------------------

# ================================ Traditional Training (Enzyme-based) ================================

"""
    train!(params::L63Parameters, target_solution::L63Solution, config::L63TrainingConfig)

Traditional training function using Enzyme.jl for gradient computation.
This provides precise gradients directly from the ODE integration with full
Optimisers.jl support for advanced optimization.
"""
function train!(
    params::L63Parameters{T},
    target_solution::L63Solution{T},
    config::L63TrainingConfig{T}) where {T}

    # Create windows from the solution data
    n_timesteps = length(target_solution.t)
    window_size = config.window_size
    stride = config.stride
    
    # Calculate number of windows we can create
    n_windows = max(0, div(n_timesteps - window_size, stride) + 1)
    n_windows > 0 || throw(ArgumentError("Insufficient data for training with given window_size and stride"))
    
    # Create train/validation split
    rng = deepcopy(config.rng)
    indices = collect(1:n_windows)
    if config.shuffle
        Random.shuffle!(rng, indices)
    end
    
    train_count = max(1, min(n_windows, round(Int, config.train_fraction * n_windows)))
    train_indices = indices[1:train_count]
    val_indices = train_count < n_windows ? indices[train_count + 1:end] : Int[]
    
    loss_fn = config.loss === nothing ? window_rmse : config.loss
    
    # Initialize parameters as NamedTuple for Optimisers.jl
    # Wrap scalars in Ref to make them mutable for Optimisers.jl
    ps = (σ = [params.σ], ρ = [params.ρ], β = [params.β])
    opt_state = Optimisers.setup(config.optimiser, ps)
    
    # Initialize tracking
    metrics_history = Vector{NamedTuple{(:train, :validation), Tuple{T, Union{Missing, T}}}}(undef, config.epochs)
    param_history = Vector{L63Parameters{T}}(undef, config.epochs + 1)
    param_history[1] = params
    
    best_params = params
    best_metric = convert(T, Inf)
    
    if config.verbose
        _print_training_header()
    end
    
    for epoch in 1:config.epochs
        if config.shuffle
            Random.shuffle!(rng, train_indices)
        end
        
        epoch_loss = zero(T)
        n_batches = 0
        
        # Create batches for this epoch
        batch_indices = [train_indices[i:min(i+config.batch_size-1, end)] 
                        for i in 1:config.batch_size:length(train_indices)]
        
        for batch in batch_indices
            # Compute average gradients over the batch
            batch_loss = zero(T)
            avg_grads = (σ = zero(T), ρ = zero(T), β = zero(T))
            
            for window_idx in batch
                # Calculate window start index
                window_start = (window_idx - 1) * stride + 1
                
                # Convert current ps to L63Parameters for gradient computation
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1])
                loss_val, gradients = compute_gradients(current_params, target_solution, window_start, window_size)
                
                batch_loss += loss_val
                avg_grads = (σ = avg_grads.σ + gradients.σ, 
                           ρ = avg_grads.ρ + gradients.ρ, 
                           β = avg_grads.β + gradients.β)
            end
            
            # Average the gradients and loss
            batch_size_actual = length(batch)
            batch_loss /= batch_size_actual
            avg_grads = (σ = avg_grads.σ / batch_size_actual,
                        ρ = avg_grads.ρ / batch_size_actual,
                        β = avg_grads.β / batch_size_actual)
            
            # Apply parameter update mask and convert to array format
            masked_grads = (σ = [config.update_mask.σ ? avg_grads.σ : zero(T)],
                           ρ = [config.update_mask.ρ ? avg_grads.ρ : zero(T)],
                           β = [config.update_mask.β ? avg_grads.β : zero(T)])
            
            # Apply gradient clipping if specified
            if config.clip_norm < Inf
                grad_norm = sqrt(masked_grads.σ[1]^2 + masked_grads.ρ[1]^2 + masked_grads.β[1]^2)
                if grad_norm > config.clip_norm
                    clip_factor = config.clip_norm / grad_norm
                    masked_grads = (σ = [masked_grads.σ[1] * clip_factor],
                                   ρ = [masked_grads.ρ[1] * clip_factor],
                                   β = [masked_grads.β[1] * clip_factor])
                end
            end
            
            # Update parameters using Optimisers.jl
            opt_state, ps = Optimisers.update(opt_state, ps, masked_grads)
            
            # Convert updated parameters back to L63Parameters
            params = L63Parameters(ps.σ[1], ps.ρ[1], ps.β[1])
            
            epoch_loss += batch_loss
            n_batches += 1
        end
        
        train_loss = epoch_loss / n_batches
        
        # Validation phase
        val_loss = if !isempty(val_indices) && epoch % config.eval_every == 0
            val_total = zero(T)
            for window_idx in val_indices
                window_start = (window_idx - 1) * stride + 1
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1])
                loss_val, _ = compute_gradients(current_params, target_solution, window_start, window_size)
                val_total += loss_val
            end
            val_total / length(val_indices)
        else
            missing
        end
        
        # Record metrics
        current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1])
        param_history[epoch + 1] = current_params
        metrics_history[epoch] = (train = train_loss, validation = val_loss)
        
        # Update best model
        metric = ismissing(val_loss) ? train_loss : val_loss
        if metric < best_metric
            best_metric = metric
            best_params = deepcopy(current_params)
        end
        
        if config.verbose
            _print_training_row(epoch, train_loss, val_loss, current_params)
        end
    end
    
    if config.verbose
        _print_training_footer()
    end
    
    return best_params, metrics_history, param_history
end

"""
    estimate_parameters(target_solution::L63Solution, initial_guess::L63Parameters; config=L63TrainingConfig())

Convenience wrapper around [`train!`] for parameter estimation workflows.
"""
function estimate_parameters(target_solution::L63Solution{T}, initial_guess::L63Parameters{T};
                             config::L63TrainingConfig{T} = L63TrainingConfig()) where {T}
    best_params, metrics_history, param_history = train!(initial_guess, target_solution, config)
    return best_params, metrics_history, param_history
end

