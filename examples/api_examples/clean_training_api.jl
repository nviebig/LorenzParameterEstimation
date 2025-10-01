# ===== Clean Training API Examples =====
# This demonstrates the clean, TensorFlow-like API for parameter estimation

using LorenzParameterEstimation

# Example 1: Basic training with defaults
function example_basic()
    # Generate some test data
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(true_params)
    solution = integrate(system, [1.0, 1.0, 1.0], 10.0)
    
    # Initial guess (could be random or from prior knowledge)
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Simple config - uses defaults (Adam optimizer, RMSE loss, 80% train split)
    config = L63TrainingConfig()
    
    # Train!
    best_params, metrics, history = train!(initial_guess, solution, config)
    
    println("True params: σ=$(true_params.σ), ρ=$(true_params.ρ), β=$(true_params.β)")
    println("Estimated:   σ=$(best_params.σ), ρ=$(best_params.ρ), β=$(best_params.β)")
end

# Example 2: Custom optimizer and loss function
function example_custom()
    # Generate test data
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(true_params)
    solution = integrate(system, [1.0, 1.0, 1.0], 10.0)
    
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Custom configuration
    config = L63TrainingConfig(
        # Optimizer choice
        optimiser = adamw_config(learning_rate=1e-3, weight_decay=1e-4).optimizer,
        
        # Loss function choice  
        loss = window_mae,  # Use MAE instead of RMSE
        
        # Training parameters
        epochs = 200,
        batch_size = 64,
        
        # Data split (70% train, 30% validation)
        train_fraction = 0.7,
        
        # Window settings
        window_size = 150,
        
        # Early stopping
        eval_every = 5
    )
    
    best_params, metrics, history = train!(initial_guess, solution, config)
    return best_params, metrics
end

# Example 3: Robust training with adaptive loss
function example_robust()
    # Generate noisy test data
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(true_params)
    solution = integrate(system, [1.0, 1.0, 1.0], 10.0)
    
    # Add some noise to make it challenging
    # solution.u .+= 0.1 * randn(size(solution.u))
    
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    config = L63TrainingConfig(
        optimiser = rmsprop_config(learning_rate=5e-4).optimizer,
        loss = adaptive_loss,  # Robust to outliers
        epochs = 300,
        train_fraction = 0.8
    )
    
    best_params, metrics, history = train!(initial_guess, solution, config)
    return best_params, metrics
end

# Example 4: Modular training (alternative API)
function example_modular()
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(true_params)
    solution = integrate(system, [1.0, 1.0, 1.0], 10.0)
    
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Direct modular API - more flexible
    results = modular_train!(
        initial_guess, 
        solution,
        optimizer_config = adam_config(learning_rate=2e-3),
        loss_function = weighted_window_loss(window_rmse, 1.2),  # Emphasize later times
        epochs = 150,
        batch_size = 32,
        train_fraction = 0.75,
        verbose = true
    )
    
    return results.best_params, results.metrics_history
end

# Example 5: Parameter subset training
function example_partial_parameters()
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(true_params)
    solution = integrate(system, [1.0, 1.0, 1.0], 10.0)
    
    # Maybe we know β accurately but want to estimate σ and ρ
    initial_guess = L63Parameters(8.0, 25.0, 8/3)  # β is correct
    
    config = L63TrainingConfig(
        optimiser = adam_config().optimizer,
        loss = window_rmse,
        epochs = 100,
        
        # Only update σ and ρ, keep β fixed
        update_σ = true,
        update_ρ = true, 
        update_β = false  # Keep β fixed at its initial value
    )
    
    best_params, metrics, history = train!(initial_guess, solution, config)
    return best_params
end

# Summary of where train/test split is defined:
println("""
🎯 TRAIN/TEST SPLIT CONFIGURATION:

The train/validation split is controlled by the `train_fraction` parameter in L63TrainingConfig:

```julia
config = L63TrainingConfig(
    train_fraction = 0.8,  # 80% for training, 20% for validation
    shuffle = true         # Shuffle the data before splitting
)
```

The splitting happens automatically in the train! function:
1. Creates windows from your trajectory data
2. Randomly shuffles the windows (if shuffle=true)  
3. Takes first `train_fraction` for training
4. Uses remaining windows for validation

You can also control:
- `window_size`: Length of each training segment
- `stride`: Overlap between windows (default: window_size÷2)
- `shuffle`: Whether to randomize the order
- `batch_size`: Mini-batch size within training data
""")