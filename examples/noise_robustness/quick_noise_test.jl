# Quick Noise Test - Simple example for immediate testing

using LorenzParameterEstimation
using Random
using Statistics

println("🔊 Quick Noise Robustness Test")
println("="^40)

# Generate clean data
true_params = L63Parameters(10.0, 28.0, 8/3)
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 3.0), dt=0.01)
clean_solution = integrate(system)

# Add noise (5% of signal magnitude)
Random.seed!(42)
signal_std = std(clean_solution.u)
noise_std = 0.05 * signal_std
noisy_solution = deepcopy(clean_solution)
noisy_solution.u .+= randn(size(clean_solution.u)) * noise_std

initial_guess = L63Parameters(8.0, 25.0, 2.5)

println("True parameters: σ=10.0, ρ=28.0, β=2.667")
println("Noise level: 5% of signal")
println()

# Test 1: Standard RMSE loss
println("1️⃣  Testing RMSE loss with noisy data...")
results_rmse = modular_train!(
    deepcopy(initial_guess), 
    noisy_solution,
    optimizer_config = adam_config(learning_rate=5e-3),
    loss_function = window_rmse,
    epochs = 30,
    window_size = 100,
    verbose = false
)

# Test 2: Robust adaptive loss
println("2️⃣  Testing adaptive loss with noisy data...")
results_adaptive = modular_train!(
    deepcopy(initial_guess), 
    noisy_solution,
    optimizer_config = adam_config(learning_rate=5e-3),
    loss_function = adaptive_loss,
    epochs = 30,
    window_size = 100,
    verbose = false
)

println("\n📊 Results:")
println("Method    │    σ     │    ρ     │    β     │ σ Error")
println("──────────┼──────────┼──────────┼──────────┼────────")
println("RMSE      │ $(rpad(round(results_rmse.best_params.σ, digits=3), 8)) │ $(rpad(round(results_rmse.best_params.ρ, digits=3), 8)) │ $(rpad(round(results_rmse.best_params.β, digits=3), 8)) │ $(round(abs(results_rmse.best_params.σ - 10.0), digits=3))")
println("Adaptive  │ $(rpad(round(results_adaptive.best_params.σ, digits=3), 8)) │ $(rpad(round(results_adaptive.best_params.ρ, digits=3), 8)) │ $(rpad(round(results_adaptive.best_params.β, digits=3), 8)) │ $(round(abs(results_adaptive.best_params.σ - 10.0), digits=3))")

rmse_error = abs(results_rmse.best_params.σ - 10.0)
adaptive_error = abs(results_adaptive.best_params.σ - 10.0)

if adaptive_error < rmse_error
    println("\n✅ Adaptive loss performed better with noisy data!")
else
    println("\n⚠️  RMSE performed similarly or better (low noise level)")
end