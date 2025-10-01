# ================================ Training ================================

import Optimisers
import Random
using Printf: @sprintf

# ================================ Modular Training  ================================

"""
    modular_train!(params, 
                target_solution; 
                optimizer_config=adam_config(), 
                loss_function=window_rmse,
                kwargs...)

Unified modular training function for all Lorenz-63 parameters using Enzyme.jl for gradient computation
with Optimisers.jl for advanced optimization.

This function automatically handles both classic (σ, ρ, β) and extended (x_s, y_s, z_s, θ) parameters.
It provides:
- Gradients via Enzyme.jl automatic differentiation for all 7 parameters
- Support for a wide range of optimizers from Optimisers.jl
- Fully modular loss functions - any function that takes (predicted::Matrix, target::Matrix) -> Real
- Direct parameter optimization without neural network overhead
- Selective parameter training (choose which parameters to update)
- Full backward compatibility

# Arguments
- `params::L63Parameters`: Initial parameter guess (works with both classic and extended parameters)
- `target_solution::L63Solution`: Target trajectory to fit
- `optimizer_config::OptimizerConfig`: Optimizer configuration (see optimizers.jl)
- `loss_function::Function`: Loss function to use - any function that takes (predicted, target) matrices and returns a scalar loss

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
- `update_x_s::Bool=false`: Whether to update x_s coordinate shift
- `update_y_s::Bool=false`: Whether to update y_s coordinate shift
- `update_z_s::Bool=false`: Whether to update z_s coordinate shift
- `update_θ::Bool=false`: Whether to update θ parameter
- `rng::AbstractRNG=Random.default_rng()`: Random number generator

# Examples
```julia
# Classic usage (unchanged - trains σ, ρ, β)
results = modular_train!(classic_params, solution)

# Train only coordinate shifts
results = modular_train!(extended_params, solution, 
                        update_σ=false, update_ρ=false, update_β=false,
                        update_x_s=true, update_y_s=true, update_z_s=true)

# Train only x_s shift
results = modular_train!(extended_params, solution, 
                        update_σ=false, update_ρ=false, update_β=false,
                        update_x_s=true)

# Train all parameters
results = modular_train!(extended_params, solution,
                        update_σ=true, update_ρ=true, update_β=true,
                        update_x_s=true, update_y_s=true, update_z_s=true, update_θ=true)

# Using different loss functions
results = modular_train!(params, solution, 
                        loss_function=window_mae,
                        epochs=200)

# Custom optimizer and weighted loss
opt_config = adamw_config(learning_rate=1e-4, weight_decay=1e-3)
results = modular_train!(params, solution, 
                        optimizer_config=opt_config,
                        loss_function=weighted_window_loss(window_rmse, 1.5),
                        epochs=200)
```
"""
function modular_train!(
    params::L63Parameters{T},                           # Initial parameter guess
    target_solution::L63Solution{T};                    # Target trajectory to fit
    # Core configuration
    optimizer_config::OptimizerConfig = adam_config(),  # Optimizer configuration (which optimizer, learning rate, etc. - see optimizers.jl)
    loss_function::Function = window_rmse,              # Loss function to use - any function that takes (predicted, target) matrices and returns a scalar loss
    
    # Training parameters
    epochs::Int = 100,                                  # Number of training epochs
    window_size::Int = 300,                             # Size of trajectory windows
    stride::Union{Nothing, Int} = nothing,              # Stride between windows (default: window_size ÷ 2)
    batch_size::Int = 32,                               # Mini-batch size
    
    # Data splitting
    train_fraction::Real = 0.8,                         # Fraction of data for training (remaining for validation)
    shuffle::Bool = true,                               # Shuffle training data
    
    # Parameter updates (classic parameters)
    update_σ::Bool = true,                              # Whether to update σ parameter
    update_ρ::Bool = true,                              # Whether to update ρ parameter
    update_β::Bool = true,                              # Whether to update β parameter
    
    # Parameter updates (extended parameters - auto-detected)
    update_x_s::Bool = false,                           # Whether to update x_s coordinate shift
    update_y_s::Bool = false,                           # Whether to update y_s coordinate shift
    update_z_s::Bool = false,                           # Whether to update z_s coordinate shift
    update_θ::Bool = false,                             # Whether to update θ parameter

    # Training control
    verbose::Bool = true,                               # Print training progress
    eval_every::Int = 1,                                # Evaluate every N epochs (on validation set)
    early_stopping_patience::Int = 20,                  # Early stopping patience (in epochs)
    early_stopping_min_delta::Real = 1e-6,              # Minimum change to qualify as improvement for early stopping
    
    # Reproducibility
    rng::Random.AbstractRNG = Random.default_rng()      # Random number generator
) where {T}
    
    # Set default stride
    stride_val = isnothing(stride) ? window_size ÷ 2 : stride  # Default to half window size
    
    # Generate window starting positions
    max_start = length(target_solution) - window_size    # Last valid start index
    max_start > 0 || throw(ArgumentError("Window size too large for trajectory"))   # Ensure we can create at least one window
    
    window_starts = collect(1:stride_val:max_start)      # Starting indices of windows
    n_windows = length(window_starts)                    # Number of windows 
    
    # Train/validation split
    indices = collect(1:n_windows)                      # Indices for shuffling
    if shuffle                                          # Shuffle if specified
        Random.shuffle!(rng, indices)                   # Shuffle indices
    end
    
    train_count = max(1, round(Int, train_fraction * n_windows)) # Ensure at least one training window
    train_indices = window_starts[indices[1:train_count]]  # Training window start indices
    val_indices = train_count < n_windows ? window_starts[indices[train_count + 1:end]] : Int[] # Validation window start indices
    
    # Initialize parameters as NamedTuple for Optimisers.jl
    # Wrap scalars in arrays to make them mutable for Optimisers.jl
    ps = (σ = [params.σ], ρ = [params.ρ], β = [params.β],
          x_s = [params.x_s], y_s = [params.y_s], z_s = [params.z_s], θ = [params.θ])  # All 7 parameters
    opt_state = Optimisers.setup(optimizer_config.optimizer, ps)    # Initialize optimizer state
    
    # Parameter update mask (for masking gradients) - all 7 parameters
    update_mask = (σ = update_σ, ρ = update_ρ, β = update_β,
                   x_s = update_x_s, y_s = update_y_s, z_s = update_z_s, θ = update_θ)
    
    # Training state
    metrics_history = NamedTuple[]                                 # To store (epoch, train_loss, val_loss, params)
    param_history = L63Parameters{T}[params]                       # To store parameter history
    best_params = params                                           # Best parameters found                
    best_metric = convert(T, Inf)                                  # Best metric (lower is better)
    patience_counter = 0                                           # Early stopping counter
    
    if verbose
        active_params = String[]
        update_σ && push!(active_params, "σ")
        update_ρ && push!(active_params, "ρ")
        update_β && push!(active_params, "β")
        update_x_s && push!(active_params, "x_s")
        update_y_s && push!(active_params, "y_s")
        update_z_s && push!(active_params, "z_s")
        update_θ && push!(active_params, "θ")
        
        println("   Optimizer: $(optimizer_config.name)")
        println("   Data: $(length(train_indices)) train windows, $(length(val_indices)) val windows")
        println("   Window size: $window_size, stride: $stride_val")
        println("   Updating: $(join(active_params, ", "))")
        println()
        println("Epoch │   Train    │    Val     │ Parameters")
        println("──────┼────────────┼────────────┼────────────────────────────")
    end
    # Training loop
    for epoch in 1:epochs         
        # Training phase
        epoch_loss = zero(T)    # Initialize epoch loss
        
        # Shuffle training windows
        current_train_indices = shuffle ? Random.shuffle(rng, copy(train_indices)) : train_indices  # Shuffle if specified
        
        # Process training windows in batches
        train_batches = [current_train_indices[i:min(i+batch_size-1, end)] 
                        for i in 1:batch_size:length(current_train_indices)]
        
        for batch_windows in train_batches
            # Compute average gradients over the batch
            batch_loss = zero(T)  # Initialize batch loss
            avg_grads = (σ = zero(T), ρ = zero(T), β = zero(T), 
                        x_s = zero(T), y_s = zero(T), z_s = zero(T), θ = zero(T))
            
            for window_start in batch_windows                                       # Process each window in the batch
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                                 ps.x_s[1], ps.y_s[1], ps.z_s[1], ps.θ[1])  # All 7 parameters
                loss_val, grads = compute_gradients_extended(
                    current_params, 
                    target_solution, 
                    window_start, 
                    window_size, 
                    loss_function
                )  # Compute loss and gradients via Enzyme for all 7 parameters
                batch_loss += loss_val # Accumulate batch loss
                avg_grads = (σ = avg_grads.σ + grads.σ, 
                           ρ = avg_grads.ρ + grads.ρ, 
                           β = avg_grads.β + grads.β,
                           x_s = avg_grads.x_s + grads.x_s,
                           y_s = avg_grads.y_s + grads.y_s,
                           z_s = avg_grads.z_s + grads.z_s,
                           θ = avg_grads.θ + grads.θ)
            end
            
            # Average the gradients and loss
            batch_size_actual = length(batch_windows)
            batch_loss /= batch_size_actual
            avg_grads = (σ = avg_grads.σ / batch_size_actual,
                        ρ = avg_grads.ρ / batch_size_actual,
                        β = avg_grads.β / batch_size_actual,
                        x_s = avg_grads.x_s / batch_size_actual,
                        y_s = avg_grads.y_s / batch_size_actual,
                        z_s = avg_grads.z_s / batch_size_actual,
                        θ = avg_grads.θ / batch_size_actual)
            
            # Apply parameter update mask and convert to array format
            masked_grads = (σ = [update_mask.σ ? avg_grads.σ : zero(T)],
                           ρ = [update_mask.ρ ? avg_grads.ρ : zero(T)],
                           β = [update_mask.β ? avg_grads.β : zero(T)],
                           x_s = [update_mask.x_s ? avg_grads.x_s : zero(T)],
                           y_s = [update_mask.y_s ? avg_grads.y_s : zero(T)],
                           z_s = [update_mask.z_s ? avg_grads.z_s : zero(T)],
                           θ = [update_mask.θ ? avg_grads.θ : zero(T)])
            
            # Update parameters using Optimisers.jl
            opt_state, ps = Optimisers.update(opt_state, ps, masked_grads)  # Update parameters using Optimisers.jl
            epoch_loss += batch_loss
        end

        train_loss = epoch_loss / length(train_batches)  # Average training loss over epoch

        # Validation phase
        val_loss = if !isempty(val_indices) && epoch % eval_every == 0          # Only evaluate on validation set every eval_every epochs
            val_total = zero(T)                                                 # Initialize validation loss                        
            for window_start in val_indices                                     # Process each validation window               
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                                 ps.x_s[1], ps.y_s[1], ps.z_s[1], ps.θ[1])  # All 7 parameters
                loss_val, _ = compute_gradients_extended(                       # Compute loss (ignore gradients)
                    current_params,                                             # Current parameters
                    target_solution,                                            # Target solution
                    window_start,                                               # Window start
                    window_size,                                                # Window size
                    loss_function                                               # Loss function
                )
                val_total += loss_val                                           # Accumulate validation loss               
            end
            val_total / length(val_indices)                                     # Average validation loss
        else
            missing                                                             # No validation this epoch                          
        end
        
        # Record metrics
        current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                         ps.x_s[1], ps.y_s[1], ps.z_s[1], ps.θ[1])  # All 7 parameters
        push!(param_history, current_params)                                    # Store parameter history

        metrics = (                                                             # Record metrics
            epoch = epoch,                                                      # Current epoch
            train_loss = train_loss,                                            # Training loss
            val_loss = val_loss,                                                # Validation loss (or missing)     
            params = current_params                                             # Current parameters      
        )
        push!(metrics_history, metrics)                                         # Store metrics history
        
        # Update best model
        metric_for_best = ismissing(val_loss) ? train_loss : val_loss           # Use validation loss if available, else training loss
        if metric_for_best < best_metric - early_stopping_min_delta             # Improvement found
            best_metric = metric_for_best                                       # Update best metric
            best_params = current_params                                        # Update best parameters
            patience_counter = 0                                                # Reset patience counter
        else
            patience_counter += 1                                               # No improvement, increment counter 
        end
        
        # Print progress
        if verbose
            val_str = ismissing(val_loss) ? "    —     " : @sprintf("%10.6f", val_loss)
            param_str = ""
            if update_σ || update_ρ || update_β
                param_str *= @sprintf("σ=%.3f,ρ=%.3f,β=%.3f", current_params.σ, current_params.ρ, current_params.β)
            end
            if update_x_s || update_y_s || update_z_s
                if length(param_str) > 0; param_str *= " "; end
                param_str *= @sprintf("x_s=%.3f,y_s=%.3f,z_s=%.3f", current_params.x_s, current_params.y_s, current_params.z_s)
            end
            if update_θ
                if length(param_str) > 0; param_str *= " "; end
                param_str *= @sprintf("θ=%.3f", current_params.θ)
            end
            
            println(@sprintf("%5d │ %10.6f │ %s │ %s",
                   epoch, train_loss, val_str, param_str))
        end
        # Early stopping
        if patience_counter >= early_stopping_patience
            if verbose
                println("\n Early stopping triggered at epoch $epoch")
                println("   Best metric: $(best_metric:.8f)")
            end
            break
        end
    end
    
    if verbose
        println()
        println("Training completed")
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
# ================================ Traditional Training (Enzyme-based),  ================================

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
    metrics_history = NamedTuple{(:train, :validation), Tuple{T, Union{Missing, T}}}[]
    param_history = L63Parameters{T}[]
    sizehint!(metrics_history, config.epochs)
    sizehint!(param_history, config.epochs + 1)
    push!(param_history, params)
    
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
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                                 params.x_s, params.y_s, params.z_s, params.θ)
                loss_val, gradients = compute_gradients(current_params, target_solution, window_start, window_size, loss_fn)
                
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
            params = L63Parameters(ps.σ[1], ps.ρ[1], ps.β[1], params.x_s, params.y_s, params.z_s, params.θ)
            
            epoch_loss += batch_loss
            n_batches += 1
        end
        
        train_loss = epoch_loss / n_batches
        
        # Validation phase
        val_loss = if !isempty(val_indices) && epoch % config.eval_every == 0
            val_total = zero(T)
            for window_idx in val_indices
                window_start = (window_idx - 1) * stride + 1
                current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                                 params.x_s, params.y_s, params.z_s, params.θ)
                loss_val, _ = compute_gradients(current_params, target_solution, window_start, window_size, loss_fn)
                val_total += loss_val
            end
            val_total / length(val_indices)
        else
            missing
        end
        
        # Record metrics
        current_params = L63Parameters{T}(ps.σ[1], ps.ρ[1], ps.β[1], 
                                         params.x_s, params.y_s, params.z_s, params.θ)
        push!(param_history, current_params)
        push!(metrics_history, (train = train_loss, validation = val_loss))
        
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

