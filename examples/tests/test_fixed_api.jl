# Quick test of the fixed training API
using LorenzParameterEstimation

# Test both traditional and modular APIs
function test_training_apis()
    println("🧪 Testing Fixed Training APIs")
    println("=" ^ 50)
    
    # Generate test data
    true_params = L63Parameters(10.0, 28.0, 8/3)
    system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
    solution = integrate(system)  # Use the system's built-in initial conditions and time span
    
    initial_guess = L63Parameters(8.0, 25.0, 2.5)
    
    println("True parameters: σ=$(true_params.σ), ρ=$(true_params.ρ), β=$(true_params.β)")
    println("Initial guess:   σ=$(initial_guess.σ), ρ=$(initial_guess.ρ), β=$(initial_guess.β)")
    println()
    
    # Test 1: Traditional train! with custom loss and optimizer
    println("1️⃣  Testing traditional train! with custom loss + optimizer")
    config1 = L63TrainingConfig(
        optimiser = adam_config(learning_rate=1e-2).optimizer,
        loss = window_mae,  # MAE instead of RMSE
        epochs = 10,  # Quick test
        window_size = 100,
        train_fraction = 0.8,
        verbose = false
    )
    
    try
        best_params1, metrics1, history1 = train!(deepcopy(initial_guess), solution, config1)
        println("   ✅ Success! Final: σ=$(round(best_params1.σ, digits=2)), ρ=$(round(best_params1.ρ, digits=2)), β=$(round(best_params1.β, digits=2))")
        println("   📊 Final loss: $(round(metrics1[end].train, digits=4))")
    catch e
        println("   ❌ Error: $e")
    end
    
    println()
    
    # Test 2: Modular train! with different optimizer and loss
    println("2️⃣  Testing modular_train! with different optimizer + loss")
    
    try
        results2 = modular_train!(
            deepcopy(initial_guess), 
            solution,
            optimizer_config = adamw_config(learning_rate=5e-3, weight_decay=1e-4),
            loss_function = window_rmse,
            epochs = 10,
            window_size = 100,
            train_fraction = 0.8,
            verbose = false
        )
        
        best_params2 = results2.best_params
        println("   ✅ Success! Final: σ=$(round(best_params2.σ, digits=2)), ρ=$(round(best_params2.ρ, digits=2)), β=$(round(best_params2.β, digits=2))")
        println("   📊 Final loss: $(round(results2.metrics_history[end].train_loss, digits=4))")
    catch e
        println("   ❌ Error: $e")
    end
    
    println()
    
    # Test 3: Different loss functions
    println("3️⃣  Testing different loss functions")
    loss_functions = [
        ("RMSE", window_rmse),
        ("MAE", window_mae), 
        ("MSE", window_mse),
        ("Adaptive", adaptive_loss)
    ]
    
    for (name, loss_fn) in loss_functions
        config = L63TrainingConfig(
            optimiser = adam_config().optimizer,
            loss = loss_fn,
            epochs = 5,
            window_size = 50,
            verbose = false
        )
        
        try
            best_params, metrics, _ = train!(deepcopy(initial_guess), solution, config)
            println("   ✅ $name loss: Final σ=$(round(best_params.σ, digits=1))")
        catch e
            println("   ❌ $name loss failed: $e")
        end
    end
    
    println()
    println("🎉 Testing complete! Both APIs should now work with custom optimizers and loss functions.")
end

# Run the test
test_training_apis()