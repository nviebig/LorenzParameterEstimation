using LorenzParameterEstimation
using Plots

println("Testing visualization functionality...")

# Create test data
params = classic_params()
system = L63System(params=params, u0=[1.0,1.0,1.0], tspan=(0.0,2.0), dt=0.01)
sol = integrate(system)

println("Testing plot_trajectory...")
try
    p = plot_trajectory(sol)
    println("✅ plot_trajectory works!")
catch e
    println("❌ plot_trajectory failed: $e")
end

println("Testing plot_phase_portrait...")
try
    p = plot_phase_portrait(sol)
    println("✅ plot_phase_portrait works!")
catch e
    println("❌ plot_phase_portrait failed: $e")
end

println("Visualization test completed!")