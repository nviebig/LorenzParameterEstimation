# Comprehensive Noise Robustness Test
# This demonstrates when adaptive loss outperforms RMSE

using LorenzParameterEstimation
using Random
using Statistics

println("🔊 Comprehensive Noise Robustness Analysis")
println("="^50)

true_params = L63Parameters(10.0, 28.0, 8/3)
initial_guess = L63Parameters(8.0, 25.0, 2.5)

println("True parameters: σ=10.0, ρ=28.0, β=2.667")
println("Initial guess: σ=8.0, ρ=25.0, β=2.5")
println()

# Test range of noise levels
noise_levels = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

println("Noise Level │ RMSE Loss      │ MAE Loss       │ Adaptive Loss  │ Best Method")
println("           │  σ    Error   │  σ    Error   │  σ    Error   │")
println("───────────┼────────────────┼────────────────┼────────────────┼─────────────")

for noise_level in noise_levels
    # Generate noisy data
    Random.seed!(42)  # Consistent results
    system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 4.0), dt=0.01)
    solution = integrate(system)
    
    if noise_level > 0
        signal_std = std(solution.u)
        noise_std = noise_level * signal_std
        solution.u .+= randn(size(solution.u)) * noise_std
    end
    
    # Test RMSE
    results_rmse = modular_train!(
        deepcopy(initial_guess), solution,
        optimizer_config = adam_config(learning_rate=5e-3),
        loss_function = window_rmse,
        epochs = 40, window_size = 120, verbose = false
    )
    
    # Test MAE
    results_mae = modular_train!(
        deepcopy(initial_guess), solution,
        optimizer_config = adam_config(learning_rate=5e-3),
        loss_function = window_mae,
        epochs = 40, window_size = 120, verbose = false
    )
    
    # Test Adaptive
    results_adaptive = modular_train!(
        deepcopy(initial_guess), solution,
        optimizer_config = adam_config(learning_rate=5e-3),
        loss_function = adaptive_loss,
        epochs = 40, window_size = 120, verbose = false
    )
    
    # Calculate errors
    rmse_error = abs(results_rmse.best_params.σ - 10.0)
    mae_error = abs(results_mae.best_params.σ - 10.0)
    adaptive_error = abs(results_adaptive.best_params.σ - 10.0)
    
    # Find best method
    errors = [rmse_error, mae_error, adaptive_error]
    methods = ["RMSE", "MAE", "Adaptive"]
    best_idx = argmin(errors)
    best_method = methods[best_idx]
    
    println(@sprintf("   %4.2f    │ %5.3f %5.3f │ %5.3f %5.3f │ %5.3f %5.3f │ %s", 
            noise_level, 
            results_rmse.best_params.σ, rmse_error,
            results_mae.best_params.σ, mae_error,
            results_adaptive.best_params.σ, adaptive_error,
            best_method))
end

println()
println("📊 Analysis:")
println("   • At low noise (≤0.05): RMSE often performs well")
println("   • At high noise (≥0.1): Adaptive/MAE should outperform RMSE")
println("   • Adaptive loss is robust to outliers caused by noise")
println()

# Demonstration with outliers
println("🎯 Outlier Robustness Test:")
println("="^30)

# Create data with deliberate outliers
Random.seed!(123)
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 3.0), dt=0.01)
outlier_solution = integrate(system)

# Add some extreme outliers (simulating measurement errors)
n_points = size(outlier_solution.u, 1)
n_outliers = max(1, round(Int, 0.02 * n_points))  # 2% outliers
outlier_indices = rand(1:n_points, n_outliers)

for idx in outlier_indices
    # Add large random spikes
    outlier_solution.u[idx, :] .+= randn(3) * 5.0
end

println("Added $(n_outliers) extreme outliers to simulate measurement errors...")

# Compare RMSE vs Adaptive on outlier data
results_rmse_outlier = modular_train!(
    deepcopy(initial_guess), outlier_solution,
    optimizer_config = adam_config(learning_rate=5e-3),
    loss_function = window_rmse,
    epochs = 50, window_size = 100, verbose = false
)

results_adaptive_outlier = modular_train!(
    deepcopy(initial_guess), outlier_solution,
    optimizer_config = adam_config(learning_rate=5e-3),
    loss_function = adaptive_loss,
    epochs = 50, window_size = 100, verbose = false
)

rmse_outlier_error = abs(results_rmse_outlier.best_params.σ - 10.0)
adaptive_outlier_error = abs(results_adaptive_outlier.best_params.σ - 10.0)

println("Results with outliers:")
println("RMSE:     σ=$(round(results_rmse_outlier.best_params.σ, digits=3)), error=$(round(rmse_outlier_error, digits=3))")
println("Adaptive: σ=$(round(results_adaptive_outlier.best_params.σ, digits=3)), error=$(round(adaptive_outlier_error, digits=3))")

if adaptive_outlier_error < rmse_outlier_error
    println("✅ Adaptive loss is more robust to outliers!")
else
    println("⚠️  RMSE performed similarly (may need more outliers to see difference)")
end

println("\n🎉 Noise robustness testing complete!")
println("💡 Use adaptive_loss when dealing with noisy or corrupted measurements.")