# Debug gradient extraction specifically
using LorenzParameterEstimation
using Enzyme

println("🔍 Debugging Gradient Extraction")
println("="^35)

# Use real data from the package
true_params = L63Parameters(10.0, 28.0, 8/3)
test_params = L63Parameters(9.5, 27.0, 2.8)

system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
solution = integrate(system)

# Extract the same data that the real function uses
window_start = 1
window_length = 50
u0 = solution[window_start]
window_end = window_start + window_length
target_window = solution.u[window_start:window_end, :]
dt = solution.system.dt

# Convert to compatible types
T = Float64
σ0, ρ0, β0 = T(test_params.σ), T(test_params.ρ), T(test_params.β)
u0_vec = u0 isa Vector ? u0 : Vector{T}(u0)
target_mat = target_window isa Matrix ? target_window : Matrix{T}(target_window)

println("Data prepared...")
println("σ0=$σ0, ρ0=$ρ0, β0=$β0")
println("u0_vec=$(u0_vec)")
println("target_mat size: $(size(target_mat))")
println("window_length=$window_length, dt=$dt")

# Create the exact MAE function from loss.jl
function mae_enzyme_debug(σ::T, ρ::T, β::T, u0::AbstractVector{T}, 
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

println("\nTesting enzyme autodiff...")

try
    grads = Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(mae_enzyme_debug),
        Enzyme.Active(σ0),
        Enzyme.Active(ρ0), 
        Enzyme.Active(β0),
        Enzyme.Const(u0_vec),
        Enzyme.Const(target_mat),
        Enzyme.Const(window_length),
        Enzyme.Const(dt)
    )
    
    println("Raw gradient result: $grads")
    println("grads[1] = $(grads[1])")
    
    G = grads[1]
    if isa(G, Tuple) && length(G) >= 3
        gσ, gρ, gβ = G[1], G[2], G[3]
        println("Extracted gradients: σ=$gσ, ρ=$gρ, β=$gβ")
        
        # Compute loss value too
        loss_val = mae_enzyme_debug(σ0, ρ0, β0, u0_vec, target_mat, window_length, dt)
        println("Loss value: $loss_val")
        
        final_grads = L63Parameters{T}(gσ, gρ, gβ)
        println("Final gradient object: $final_grads")
    else
        println("Unexpected gradient structure: type=$(typeof(G)), value=$G")
    end
    
catch e
    println("ERROR: $e")
    println("Error type: $(typeof(e))")
end

println("\n💡 This should show us exactly what's happening with gradient extraction")