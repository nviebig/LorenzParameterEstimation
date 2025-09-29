# Simple test script to verify package functionality
using Pkg
Pkg.activate("/Users/niklasviebig/master_thesis/LorenzParameterEstimation")

println("Testing LorenzParameterEstimation.jl package...")

# Test basic import
using LorenzParameterEstimation
println("✅ Package imported successfully")

# Test basic types and functions
params = classic_params()
println("✅ Classic parameters: $params")

system = L63System(params=params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 1.0), dt=0.01)
println("✅ System created: $(typeof(system))")

solution = integrate(system)
println("✅ Integration completed: $(length(solution)) points")

# Test visualization extensions
try
    using Plots
    println("✅ Plots.jl loaded")
    
    # Check if visualization functions are available
    if isdefined(Main, :plot_trajectory)
        println("✅ Visualization extensions loaded successfully")
    else
        println("❌ Visualization extensions not loaded")
    end
catch e
    println("⚠️  Plots.jl not available: $e")
end

println("\nPackage test completed!")