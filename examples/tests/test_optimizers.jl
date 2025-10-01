# Test to verify we're using actual optimizers, not just gradient descent
using LorenzParameterEstimation

# Generate longer trajectory 
true_params = L63Parameters(10.0, 28.0, 8/3)
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 5.0), dt=0.01)
solution = integrate(system)

initial_guess = L63Parameters(8.0, 25.0, 2.5)

println("🔍 Testing if we're actually using different optimizers...")
println("True params: σ=10.0, ρ=28.0, β=2.667")
println("Initial guess: σ=8.0, ρ=25.0, β=2.5")
println()

# Test 1: SGD with no momentum (pure gradient descent)
println("1️⃣  Testing SGD (no momentum)...")
results_sgd = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = sgd_config(learning_rate=5e-3, momentum=0.0),
    loss_function = window_rmse,
    epochs = 30,
    window_size = 100,
    verbose = false
)

# Test 2: Adam (adaptive learning rates)
println("2️⃣  Testing Adam...")
results_adam = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adam_config(learning_rate=5e-3),
    loss_function = window_rmse,
    epochs = 30,
    window_size = 100,
    verbose = false
)

# Test 3: AdamW (with weight decay)
println("3️⃣  Testing AdamW...")
results_adamw = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adamw_config(learning_rate=5e-3, weight_decay=1e-3),
    loss_function = window_rmse,
    epochs = 30,
    window_size = 100,
    verbose = false
)

println("\n📊 Results after 30 epochs:")
println("="^50)
println("Optimizer │    σ     │    ρ     │    β     │  Loss  ")
println("──────────┼──────────┼──────────┼──────────┼────────")
println("SGD       │ $(rpad(round(results_sgd.best_params.σ, digits=3), 8)) │ $(rpad(round(results_sgd.best_params.ρ, digits=3), 8)) │ $(rpad(round(results_sgd.best_params.β, digits=3), 8)) │ $(round(results_sgd.metrics_history[end].train_loss, digits=3))")
println("Adam      │ $(rpad(round(results_adam.best_params.σ, digits=3), 8)) │ $(rpad(round(results_adam.best_params.ρ, digits=3), 8)) │ $(rpad(round(results_adam.best_params.β, digits=3), 8)) │ $(round(results_adam.metrics_history[end].train_loss, digits=3))")
println("AdamW     │ $(rpad(round(results_adamw.best_params.σ, digits=3), 8)) │ $(rpad(round(results_adamw.best_params.ρ, digits=3), 8)) │ $(rpad(round(results_adamw.best_params.β, digits=3), 8)) │ $(round(results_adamw.metrics_history[end].train_loss, digits=3))")

println("\n📈 Parameter changes from initial:")
println("SGD:   Δσ=$(round(results_sgd.best_params.σ - 8.0, digits=3)), Δρ=$(round(results_sgd.best_params.ρ - 25.0, digits=3)), Δβ=$(round(results_sgd.best_params.β - 2.5, digits=3))")
println("Adam:  Δσ=$(round(results_adam.best_params.σ - 8.0, digits=3)), Δρ=$(round(results_adam.best_params.ρ - 25.0, digits=3)), Δβ=$(round(results_adam.best_params.β - 2.5, digits=3))")
println("AdamW: Δσ=$(round(results_adamw.best_params.σ - 8.0, digits=3)), Δρ=$(round(results_adamw.best_params.ρ - 25.0, digits=3)), Δβ=$(round(results_adamw.best_params.β - 2.5, digits=3))")

# Check if results are actually different
σ_diff_sgd_adam = abs(results_sgd.best_params.σ - results_adam.best_params.σ)
σ_diff_adam_adamw = abs(results_adam.best_params.σ - results_adamw.best_params.σ)

println("\n🔬 Analysis:")
if σ_diff_sgd_adam > 0.01 || σ_diff_adam_adamw > 0.01
    println("✅ Optimizers are DIFFERENT! SGD vs Adam difference: $(round(σ_diff_sgd_adam, digits=4))")
    println("✅ This proves we're using actual optimizers, not just gradient descent!")
else
    println("⚠️  Optimizers seem similar. Difference: $(round(σ_diff_sgd_adam, digits=4))")
    println("   This might indicate we're still doing vanilla gradient descent...")
end