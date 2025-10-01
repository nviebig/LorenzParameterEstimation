# Quick convergence test to understand the training behavior
using LorenzParameterEstimation
using Random

println("🔍 Training Convergence Investigation")
println("="^40)

true_params = L63Parameters(10.0, 28.0, 8/3)
initial_guess = L63Parameters(8.0, 25.0, 2.5)

println("True: σ=10.0, ρ=28.0, β=2.667")
println("Guess: σ=8.0, ρ=25.0, β=2.5")
println()

# Generate clean data first
Random.seed!(42)
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 10.0), dt=0.01)
solution = integrate(system)

println("Testing different learning rates and epochs...")
println()

configs = [
    (learning_rate=1e-2, epochs=200, name="High LR, More Epochs"),
    (learning_rate=5e-3, epochs=300, name="Medium LR, More Epochs"),
    (learning_rate=1e-3, epochs=150, name="Low LR, Medium Epochs"),
    (learning_rate=1e-4, epochs=500, name="Very Low LR, Many Epochs")
]

for config in configs
    println("Testing: $(config.name)")
    
    results = modular_train!(
        deepcopy(initial_guess), solution,
        optimizer_config = adam_config(learning_rate=config.learning_rate),
        loss_function = window_rmse,
        epochs = config.epochs, 
        window_size = 150, 
        verbose = false
    )
    
    println("  Final: σ=$(round(results.best_params.σ, digits=3)), " *
            "ρ=$(round(results.best_params.ρ, digits=3)), " *
            "β=$(round(results.best_params.β, digits=3))")
    println("  Error: σ_err=$(round(abs(results.best_params.σ - 10.0), digits=3))")
    println()
end

println("🎯 Let's also check the last few loss values to see convergence:")

# Detailed training with verbose output
results_verbose = modular_train!(
    deepcopy(initial_guess), solution,
    optimizer_config = adam_config(learning_rate=1e-2),
    loss_function = window_rmse,
    epochs = 100, 
    window_size = 150, 
    verbose = true
)

println("\nFinal parameters:")
println("σ = $(results_verbose.best_params.σ) (target: 10.0)")
println("ρ = $(results_verbose.best_params.ρ) (target: 28.0)")
println("β = $(results_verbose.best_params.β) (target: 2.667)")