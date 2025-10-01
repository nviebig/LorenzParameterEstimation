# Noise Robustness Examples for Lorenz Parameter Estimation

using LorenzParameterEstimation
using Random
using Printf

"""
    add_noise!(solution, noise_level; rng=Random.default_rng())

Add Gaussian noise to a solution trajectory.

# Arguments
- `solution`: L63Solution to modify
- `noise_level`: Standard deviation of noise relative to signal magnitude
- `rng`: Random number generator
"""
function add_noise!(solution::L63Solution, noise_level::Real; rng=Random.default_rng())
    # Calculate signal magnitude for relative noise
    signal_std = std(solution.u)
    noise_std = noise_level * signal_std
    
    # Add noise to trajectory
    noise = randn(rng, size(solution.u)) * noise_std
    solution.u .+= noise
    
    return solution
end

"""
    generate_noisy_data(true_params; noise_level=0.05, tspan=(0.0, 10.0), dt=0.01)

Generate a noisy Lorenz trajectory for testing parameter estimation robustness.
"""
function generate_noisy_data(true_params::L63Parameters; 
                           noise_level::Real = 0.05,
                           tspan::Tuple = (0.0, 10.0),
                           dt::Real = 0.01,
                           seed::Int = 42)
    
    # Generate clean trajectory
    rng = Random.MersenneTwister(seed)
    system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=tspan, dt=dt)
    solution = integrate(system)
    
    # Add noise
    add_noise!(solution, noise_level, rng=rng)
    
    return solution
end

"""
Example 1: Basic noise robustness test
"""
function example_basic_noise()
    println("🔊 Example 1: Basic Noise Robustness")
    println("="^50)
    
    true_params = L63Parameters(10.0, 28.0, 8/3)
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Test different noise levels
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    println("Noise Level │    σ     │    ρ     │    β     │  Loss   │ σ Error ")
    println("────────────┼──────────┼──────────┼──────────┼─────────┼─────────")
    
    for noise_level in noise_levels
        # Generate noisy data
        noisy_solution = generate_noisy_data(true_params, noise_level=noise_level)
        
        # Train with RMSE loss (standard)
        config = L63TrainingConfig(
            optimiser = adam_config(learning_rate=5e-3).optimizer,
            loss = window_rmse,
            epochs = 50,
            window_size = 150,
            verbose = false
        )
        
        best_params, _, _ = train!(deepcopy(initial_guess), noisy_solution, config)
        
        # Calculate error
        σ_error = abs(best_params.σ - true_params.σ)
        final_loss = compute_loss(best_params, noisy_solution, 1, 100)
        
        println(@sprintf("    %4.2f    │ %8.3f │ %8.3f │ %8.3f │ %7.3f │ %7.3f", 
                noise_level, best_params.σ, best_params.ρ, best_params.β, final_loss, σ_error))
    end
    
    println("\n📊 Observation: As noise increases, estimation error should increase.")
    println("   If RMSE loss becomes unstable with high noise, try adaptive_loss instead.")
end

"""
Example 2: Comparing loss functions for noise robustness
"""
function example_robust_loss_functions()
    println("\n🛡️  Example 2: Robust Loss Functions")
    println("="^50)
    
    true_params = L63Parameters(10.0, 28.0, 8/3)
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Generate moderately noisy data
    noisy_solution = generate_noisy_data(true_params, noise_level=0.08)
    
    # Test different loss functions
    loss_functions = [
        ("RMSE", window_rmse),
        ("MAE", window_mae),
        ("Adaptive", adaptive_loss),
    ]
    
    println("Loss Function │    σ     │    ρ     │    β     │ σ Error │  Status")
    println("──────────────┼──────────┼──────────┼──────────┼─────────┼─────────")
    
    for (name, loss_fn) in loss_functions
        try
            results = modular_train!(
                deepcopy(initial_guess), 
                noisy_solution,
                optimizer_config = adam_config(learning_rate=5e-3),
                loss_function = loss_fn,
                epochs = 50,
                window_size = 150,
                verbose = false
            )
            
            σ_error = abs(results.best_params.σ - true_params.σ)
            
            println(@sprintf("%-13s │ %8.3f │ %8.3f │ %8.3f │ %7.3f │    ✅", 
                    name, results.best_params.σ, results.best_params.ρ, 
                    results.best_params.β, σ_error))
        catch e
            println(@sprintf("%-13s │    ---   │    ---   │    ---   │   ---   │    ❌", name))
        end
    end
    
    println("\n📊 Adaptive loss should be more robust to outliers in noisy data.")
end

"""
Example 3: Noise level sensitivity analysis
"""
function example_noise_sensitivity()
    println("\n📈 Example 3: Noise Sensitivity Analysis")
    println("="^50)
    
    true_params = L63Parameters(10.0, 28.0, 8/3)
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Test range of noise levels
    noise_levels = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    
    println("Testing parameter estimation accuracy vs noise level...")
    println()
    println("Noise │ RMSE Loss      │ Adaptive Loss   │ Improvement")
    println("Level │  σ     Error   │  σ     Error    │")
    println("──────┼────────────────┼─────────────────┼────────────")
    
    for noise_level in noise_levels
        noisy_solution = generate_noisy_data(true_params, noise_level=noise_level, seed=123)
        
        # Test RMSE loss
        results_rmse = modular_train!(
            deepcopy(initial_guess), noisy_solution,
            optimizer_config = adam_config(learning_rate=5e-3),
            loss_function = window_rmse,
            epochs = 40, window_size = 120, verbose = false
        )
        
        # Test Adaptive loss
        results_adaptive = modular_train!(
            deepcopy(initial_guess), noisy_solution,
            optimizer_config = adam_config(learning_rate=5e-3),
            loss_function = adaptive_loss,
            epochs = 40, window_size = 120, verbose = false
        )
        
        rmse_error = abs(results_rmse.best_params.σ - true_params.σ)
        adaptive_error = abs(results_adaptive.best_params.σ - true_params.σ)
        improvement = rmse_error > adaptive_error ? "✅ Better" : "❌ Worse"
        
        println(@sprintf("%5.2f │ %6.3f %6.3f │ %6.3f %6.3f  │ %s", 
                noise_level, results_rmse.best_params.σ, rmse_error,
                results_adaptive.best_params.σ, adaptive_error, improvement))
    end
    
    println("\n📊 Adaptive loss should perform better at higher noise levels.")
end

"""
Example 4: Custom noise-robust training configuration
"""
function example_noise_robust_config()
    println("\n⚙️  Example 4: Noise-Robust Training Configuration")
    println("="^50)
    
    true_params = L63Parameters(10.0, 28.0, 8/3)
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    # Generate very noisy data
    very_noisy_solution = generate_noisy_data(true_params, noise_level=0.15)
    
    println("Testing robust configuration for very noisy data (noise_level=0.15)...")
    
    # Robust configuration
    robust_config = L63TrainingConfig(
        # Use AdamW with weight decay for regularization
        optimiser = adamw_config(learning_rate=1e-3, weight_decay=1e-3).optimizer,
        
        # Use adaptive loss for outlier robustness
        loss = adaptive_loss,
        
        # More epochs for convergence with noise
        epochs = 100,
        
        # Smaller windows to limit error accumulation
        window_size = 100,
        
        # Larger batch size for stable gradients
        batch_size = 64,
        
        # More validation to catch overfitting
        eval_every = 5,
        
        verbose = false
    )
    
    best_params, metrics, _ = train!(initial_guess, very_noisy_solution, robust_config)
    
    println("Results with robust configuration:")
    println(@sprintf("True:      σ=%6.3f, ρ=%6.3f, β=%6.3f", true_params.σ, true_params.ρ, true_params.β))
    println(@sprintf("Estimated: σ=%6.3f, ρ=%6.3f, β=%6.3f", best_params.σ, best_params.ρ, best_params.β))
    println(@sprintf("Errors:    σ=%6.3f, ρ=%6.3f, β=%6.3f", 
            abs(best_params.σ - true_params.σ),
            abs(best_params.ρ - true_params.ρ),
            abs(best_params.β - true_params.β)))
    
    println("\n🎯 This configuration should handle very noisy data better than defaults.")
end

# Run all examples
function run_all_noise_examples()
    println("🎯 Lorenz Parameter Estimation - Noise Robustness Examples")
    println("="^60)
    
    example_basic_noise()
    example_robust_loss_functions()
    example_noise_sensitivity()
    example_noise_robust_config()
    
    println("\n✅ All noise robustness examples completed!")
    println("\n💡 Key takeaways:")
    println("   • adaptive_loss is more robust to noisy data than RMSE")
    println("   • AdamW with weight decay provides regularization")
    println("   • Smaller window sizes limit error accumulation")
    println("   • Higher noise requires more careful hyperparameter tuning")
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_noise_examples()
end