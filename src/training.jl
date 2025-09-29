# ================================ Training ================================

"""
    train!(params::L63Parameters, target_solution::L63Solution, config::L63TrainingConfig)

Train Lorenz-63 parameters using windowed gradient descent with Enzyme AD.

# Arguments
- `params`: Initial parameter guess (will be modified in-place)
- `target_solution`: Target trajectory for parameter estimation
- `config`: Training configuration

# Returns
- `(best_params, loss_history, param_history)`: Best parameters, per-epoch loss values, 
  and the parameter state (including epoch 0) tracked across training.
"""
function train!(
    params::L63Parameters{T},        # Initial parameter guess (modified in-place)
    target_solution::L63Solution{T}, # Target trajectory
    config::L63TrainingConfig{T}     # Training configuration
    ) where {T}
    
    # Validate inputs
    N = length(target_solution)
    N > config.window_size || throw(ArgumentError("Target trajectory too short for window size"))
    
    # Initialize tracking
    loss_history = T[]
    param_history = L63Parameters{T}[]
    push!(param_history, L63Parameters{T}(params.σ, params.ρ, params.β))
    best_params = L63Parameters{T}(params.σ, params.ρ, params.β)
    best_loss = T(Inf)
    
    # Window starts (non-overlapping for simplicity)
    window_starts = collect(1:config.window_size:(N - config.window_size))
    
    # Training header
    if config.verbose
        _print_training_header(config.epochs)
        t0 = time()
        prev_loss = T(NaN)
    end
    
    # Training loop
    for epoch in 1:config.epochs
        epoch_start = config.verbose ? time() : 0.0
        epoch_loss = zero(T)
        epoch_grad_norm = zero(T)
        n_windows = 0
        
        # Process each window
        for window_start in window_starts
            # Compute loss and gradients
            loss, grads = compute_gradients(params, target_solution, window_start, config.window_size)
            
            # Gradient clipping
            grad_norm = norm(grads)
            scale = grad_norm > config.clip_norm ? config.clip_norm / grad_norm : one(T)
            
            # Parameter updates (respecting update mask)
            if config.update_mask.σ
                params = L63Parameters{T}(params.σ - config.η * scale * grads.σ, params.ρ, params.β)
            end
            if config.update_mask.ρ  
                params = L63Parameters{T}(params.σ, params.ρ - config.η * scale * grads.ρ, params.β)
            end
            if config.update_mask.β
                params = L63Parameters{T}(params.σ, params.ρ, params.β - config.η * scale * grads.β)
            end
            
            epoch_loss += loss
            epoch_grad_norm += grad_norm
            n_windows += 1
        end
        
        # Average loss over windows
        avg_loss = epoch_loss / n_windows
        avg_grad_norm = epoch_grad_norm / n_windows
        push!(loss_history, avg_loss)
        push!(param_history, L63Parameters{T}(params.σ, params.ρ, params.β))
        
        # Track best parameters
        if avg_loss < best_loss
            best_loss = avg_loss
            best_params = L63Parameters{T}(params.σ, params.ρ, params.β)
        end
        
        # Progress logging
        if config.verbose
            epoch_time = time() - epoch_start
            total_time = time() - t0
            Δloss = isnan(prev_loss) ? zero(T) : avg_loss - prev_loss
            
            _print_training_row(epoch, avg_loss, Δloss, best_loss, avg_grad_norm, 
                              config.η, params, epoch_time, total_time)
            prev_loss = avg_loss
        end
    end
    
    if config.verbose
        _print_training_footer()
    end
    
    return best_params, loss_history, param_history
end

# Private helper functions for pretty printing
function _print_training_header(total_epochs::Int)
    println()
    println("┌────────┬───────────┬────────────┬───────────┬───────────┬─────────┬─────────┬─────────┬─────────┬────────┬─────────┐")
    println("│  Epoch │      Loss │      ΔLoss │      Best │  GradNorm │      LR │       σ │       ρ │       β │   t/ep │ Elapsed │")
    println("├────────┼───────────┼────────────┼───────────┼───────────┼─────────┼─────────┼─────────┼─────────┼────────┼─────────┤")
    flush(stdout)  # Ensure proper output ordering
end

function _print_training_row(epoch, loss, Δloss, best_loss, grad_norm, η, params, t_epoch, t_total)
    loss_str = @sprintf("%9.6f", loss)
    Δloss_str = @sprintf("%+9.6f", Δloss)  
    best_str = @sprintf("%9.6f", best_loss)
    grad_str = @sprintf("%9.6f", grad_norm)
    lr_str = @sprintf("%7.5f", η)
    σ_str = @sprintf("%7.4f", params.σ)
    ρ_str = @sprintf("%7.4f", params.ρ)  
    β_str = @sprintf("%7.4f", params.β)
    t_ep_str = t_epoch < 1.0 ? @sprintf("%5.0fms", t_epoch * 1000) : @sprintf("%6.1fs", t_epoch)
    t_tot_str = @sprintf("%7.1fs", t_total)
    
    Δcolor = Δloss < 0 ? "\e[32m" : (Δloss > 0 ? "\e[31m" : "\e[2m")
    reset = "\e[0m"
    
    @printf("│ %6d  │ %s │ %s%s%s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │\n",
            epoch, loss_str, Δcolor, Δloss_str, reset, best_str, grad_str, 
            lr_str, σ_str, ρ_str, β_str, t_ep_str, t_tot_str)
    flush(stdout)
end

function _print_training_footer()
    println("└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    println()
    flush(stdout)
end

"""
    estimate_parameters(target_solution::L63Solution, initial_guess::L63Parameters; 
                       config::L63TrainingConfig = L63TrainingConfig())

High-level function for parameter estimation.

# Arguments
- `target_solution`: Observed trajectory
- `initial_guess`: Initial parameter guess
- `config`: Training configuration (optional)

# Returns
- `(estimated_params, loss_history, param_history)`
"""
function estimate_parameters(target_solution::L63Solution{T}, 
                           initial_guess::L63Parameters{T};
                           config::L63TrainingConfig{T} = L63TrainingConfig()) where {T}
    
    # Copy initial guess (avoid mutation)
    params = L63Parameters{T}(initial_guess.σ, initial_guess.ρ, initial_guess.β)
    
    # Train parameters
    best_params, loss_history, param_history = train!(params, target_solution, config)
    
    return best_params, loss_history, param_history
end
