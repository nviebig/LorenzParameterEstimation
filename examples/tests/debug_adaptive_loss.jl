# Debug adaptive loss behavior
using LorenzParameterEstimation
using Random

println("🔍 Debugging Adaptive Loss Behavior")
println("="^40)

true_params = L63Parameters(10.0, 28.0, 8/3)
initial_guess = L63Parameters(9.5, 27.0, 2.8)

# Generate clean data
Random.seed!(42)
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 6.0), dt=0.01)
solution = integrate(system)

println("Testing different loss functions with verbose output...")
println()

# Test RMSE with verbose
println("1️⃣ RMSE Training (first 20 epochs):")
results_rmse = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adam_config(learning_rate=2e-2),
    loss_function = window_rmse,
    epochs = 20, window_size = 120, verbose = true
)

println("\n" * "="^50)
println("2️⃣ Adaptive Loss Training (first 20 epochs):")
results_adaptive = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adam_config(learning_rate=2e-2),
    loss_function = adaptive_loss,
    epochs = 20, window_size = 120, verbose = true
)

println("\n" * "="^50)
println("📊 Comparison after 20 epochs:")
println("RMSE final:     σ=$(round(results_rmse.best_params.σ, digits=4))")
println("Adaptive final: σ=$(round(results_adaptive.best_params.σ, digits=4))")

# Let's also test with different adaptive loss delta parameter
println("\n" * "="^50)
println("3️⃣ Testing different adaptive loss configurations...")

# Unfortunately, our current adaptive_loss might be hardcoded
# Let's check what happens with MAE instead
println("Testing MAE loss (should be more robust than RMSE):")
results_mae = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adam_config(learning_rate=2e-2),
    loss_function = window_mae,
    epochs = 20, window_size = 120, verbose = true
)

println("\nMAE final: σ=$(round(results_mae.best_params.σ, digits=4))")

println("\n💡 Analysis:")
if abs(results_adaptive.best_params.σ - initial_guess.σ) < 0.01
    println("   ⚠️  Adaptive loss is NOT updating parameters - possible gradient issue")
else
    println("   ✅ Adaptive loss is updating parameters normally")
end

if abs(results_mae.best_params.σ - initial_guess.σ) < 0.01
    println("   ⚠️  MAE loss is NOT updating parameters - possible gradient issue")
else
    println("   ✅ MAE loss is updating parameters normally")
end