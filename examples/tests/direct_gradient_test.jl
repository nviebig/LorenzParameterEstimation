# Test gradients directly
using LorenzParameterEstimation
using Enzyme

println("🔬 Direct Gradient Testing")
println("="^30)

# Test parameters
true_params = L63Parameters(10.0, 28.0, 8/3)
test_params = L63Parameters(9.5, 27.0, 2.8)

# Generate test data
system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
solution = integrate(system)

println("Testing gradient computation directly...")

# Test the compute_gradients function
try
    println("\n1️⃣ Testing RMSE gradients:")
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.window_rmse
    )
    println("   Loss: $(round(loss_val, digits=6))")
    println("   Gradients: σ=$(round(grads.σ, digits=6)), ρ=$(round(grads.ρ, digits=6)), β=$(round(grads.β, digits=6))")
    println("   Gradient magnitudes: $(round(abs(grads.σ), digits=6)), $(round(abs(grads.ρ), digits=6)), $(round(abs(grads.β), digits=6))")
catch e
    println("   ERROR: $e")
end

try
    println("\n2️⃣ Testing Adaptive gradients:")
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.adaptive_loss
    )
    println("   Loss: $(round(loss_val, digits=6))")
    println("   Gradients: σ=$(round(grads.σ, digits=6)), ρ=$(round(grads.ρ, digits=6)), β=$(round(grads.β, digits=6))")
    println("   Gradient magnitudes: $(round(abs(grads.σ), digits=6)), $(round(abs(grads.ρ), digits=6)), $(round(abs(grads.β), digits=6))")
catch e
    println("   ERROR: $e")
end

try
    println("\n3️⃣ Testing MAE gradients:")
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.window_mae
    )
    println("   Loss: $(round(loss_val, digits=6))")
    println("   Gradients: σ=$(round(grads.σ, digits=6)), ρ=$(round(grads.ρ, digits=6)), β=$(round(grads.β, digits=6))")
    println("   Gradient magnitudes: $(round(abs(grads.σ), digits=6)), $(round(abs(grads.ρ), digits=6)), $(round(abs(grads.β), digits=6))")
catch e
    println("   ERROR: $e")
end

println("\n💡 Analysis:")
println("   If RMSE has non-zero gradients but others don't, the enzyme implementations are the issue")
println("   If all have zero gradients, there might be a deeper problem with the setup")