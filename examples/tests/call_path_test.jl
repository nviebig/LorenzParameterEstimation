# Test the exact compute_gradients_modular call path
using LorenzParameterEstimation

println("🎯 Testing Exact Call Path")
println("="^30)

true_params = L63Parameters(10.0, 28.0, 8/3)
test_params = L63Parameters(9.5, 27.0, 2.8)

system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
solution = integrate(system)

println("Testing compute_gradients_modular with MAE...")

# This is the exact call that happens in training
try
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.window_mae
    )
    
    println("✅ Success!")
    println("Loss value: $loss_val")
    println("Gradients: σ=$(grads.σ), ρ=$(grads.ρ), β=$(grads.β)")
    
    # Check if these are the same as the direct call
    if abs(grads.σ) > 1e-10
        println("🎉 MAE gradients are working through compute_gradients_modular!")
    else
        println("⚠️  MAE gradients are still zero through compute_gradients_modular")
    end
    
catch e
    println("❌ Error in compute_gradients_modular: $e")
end

println("\nTesting with adaptive loss...")

try
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.adaptive_loss
    )
    
    println("✅ Success!")
    println("Loss value: $loss_val")
    println("Gradients: σ=$(grads.σ), ρ=$(grads.ρ), β=$(grads.β)")
    
    if abs(grads.σ) > 1e-10
        println("🎉 Adaptive gradients are working through compute_gradients_modular!")
    else
        println("⚠️  Adaptive gradients are still zero through compute_gradients_modular")
    end
    
catch e
    println("❌ Error in compute_gradients_modular: $e")
end

println("\nFor comparison, testing RMSE:")

try
    loss_val, grads = LorenzParameterEstimation.compute_gradients_modular(
        test_params, solution, 1, 50, LorenzParameterEstimation.window_rmse
    )
    
    println("✅ Success!")
    println("Loss value: $loss_val")
    println("Gradients: σ=$(grads.σ), ρ=$(grads.ρ), β=$(grads.β)")
    
catch e
    println("❌ Error in compute_gradients_modular: $e")
end

println("\n💡 This will tell us if the issue is in compute_gradients_modular or elsewhere")