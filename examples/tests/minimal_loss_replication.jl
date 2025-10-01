# Create a minimal test that exactly replicates the loss.jl logic
using LorenzParameterEstimation
using Enzyme

println("🔬 Minimal Loss.jl Replication Test")
println("="^40)

true_params = L63Parameters(10.0, 28.0, 8/3)
test_params = L63Parameters(9.5, 27.0, 2.8)

system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
solution = integrate(system)

# Replicate the exact logic from compute_gradients_modular
window_start = 1
window_length = 50

# Extract window data (exactly like loss.jl)
u0 = solution[window_start]
window_end = window_start + window_length
target_window = solution.u[window_start:window_end, :]
dt = solution.system.dt

# Convert to compatible types (exactly like loss.jl)
T = Float64
σ0, ρ0, β0 = T(test_params.σ), T(test_params.ρ), T(test_params.β)
u0_vec = u0 isa Vector ? u0 : Vector{T}(u0)
target_mat = target_window isa Matrix ? target_window : Matrix{T}(target_window)

println("Data setup complete...")
println("σ0=$σ0, ρ0=$ρ0, β0=$β0")

# Create exact MAE function from loss.jl  
function mae_enzyme_exact(σ::T, ρ::T, β::T, u0::AbstractVector{T}, 
                         target_trajectory::AbstractMatrix{T}, 
                         window_length::Int, dt::T) where {T}
    params = L63Parameters{T}(σ, ρ, β)
    u = similar(u0)
    u .= u0
    
    ae = zero(T) # Absolute error accumulator
    count = 0
    epsilon = T(1e-6)  # Small value for smooth approximation
    
    @inbounds for i in 1:window_length 
        u = LorenzParameterEstimation.rk4_step(u, params, dt)
        for j in 1:3
            diff = u[j] - target_trajectory[i+1, j]
            # Use smooth approximation: sqrt(x^2 + ε) ≈ |x| but differentiable everywhere
            ae += sqrt(diff * diff + epsilon)
            count += 1
        end
    end
    
    return ae / count
end

println("Testing enzyme call exactly like loss.jl...")

try
    grads = Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(mae_enzyme_exact),
        Enzyme.Active(σ0),
        Enzyme.Active(ρ0), 
        Enzyme.Active(β0),
        Enzyme.Const(u0_vec),
        Enzyme.Const(target_mat),
        Enzyme.Const(window_length),
        Enzyme.Const(dt)
    )
    
    println("Raw grads: $grads")
    println("grads[1]: $(grads[1])")
    
    # Try both extraction methods
    println("\nMethod 1 (G = grads[1]; G[1], G[2], G[3]):")
    G = grads[1]
    gσ1, gρ1, gβ1 = G[1], G[2], G[3]
    println("  σ=$gσ1, ρ=$gρ1, β=$gβ1")
    
    println("\nMethod 2 (grads[1][1], grads[1][2], grads[1][3]):")
    gσ2, gρ2, gβ2 = grads[1][1], grads[1][2], grads[1][3]
    println("  σ=$gσ2, ρ=$gρ2, β=$gβ2")
    
    # They should be the same
    if gσ1 == gσ2 && gρ1 == gρ2 && gβ1 == gβ2
        println("✅ Both methods give the same result")
    else
        println("❌ Methods give different results!")
    end
    
    # Check what's creating the L63Parameters
    final_grads = L63Parameters{T}(gσ1, gρ1, gβ1)
    println("\nFinal gradient object: $final_grads")
    println("Individual values: σ=$(final_grads.σ), ρ=$(final_grads.ρ), β=$(final_grads.β)")
    
catch e
    println("ERROR: $e")
end