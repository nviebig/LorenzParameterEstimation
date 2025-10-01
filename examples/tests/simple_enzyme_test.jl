# Test simple enzyme functions directly
using LorenzParameterEstimation
using Enzyme

println("🧪 Testing Simple Enzyme Functions")
println("="^35)

# Create some test data
test_params = L63Parameters(9.5, 27.0, 2.8)
u0 = [1.0, 1.0, 1.0]
target_mat = [1.1 1.1 1.1; 1.2 1.2 1.2; 1.3 1.3 1.3]  # Simple 3x3 target
window_length = 2
dt = 0.01

println("Testing simple MAE enzyme function...")

# Test the MAE enzyme function directly
function simple_mae_test(σ, ρ, β, u0, target_trajectory, window_length, dt)
    T = typeof(σ)
    params = L63Parameters{T}(σ, ρ, β)
    u = similar(u0)
    u .= u0
    
    total_error = zero(T)
    count = 0
    epsilon = T(1e-6)
    
    for i in 1:window_length 
        u = LorenzParameterEstimation.rk4_step(u, params, dt)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            # Simple smooth approximation
            smooth_abs = sqrt(diff * diff + epsilon)
            total_error += smooth_abs
            count += 1
        end
    end
    
    return total_error / count
end

# Test gradients
try
    grads = Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(simple_mae_test),
        Enzyme.Active(9.5),
        Enzyme.Active(27.0), 
        Enzyme.Active(2.8),
        Enzyme.Const(u0),
        Enzyme.Const(target_mat),
        Enzyme.Const(window_length),
        Enzyme.Const(dt)
    )
    
    println("Simple MAE gradients: $(grads[1])")
    
    # Also compute the function value
    loss_val = simple_mae_test(9.5, 27.0, 2.8, u0, target_mat, window_length, dt)
    println("Simple MAE loss: $loss_val")
catch e
    println("ERROR with simple MAE: $e")
end

println("\nTesting even simpler function...")

# Test the absolute simplest case
function ultra_simple_test(σ, ρ, β)
    return σ^2 + ρ^2 + β^2
end

try
    grads = Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(ultra_simple_test),
        Enzyme.Active(9.5),
        Enzyme.Active(27.0), 
        Enzyme.Active(2.8)
    )
    
    println("Ultra simple gradients: $(grads[1])")
catch e
    println("ERROR with ultra simple: $e")
end

println("\n💡 This will help us identify where the issue lies")