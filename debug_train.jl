#!/usr/bin/env julia
# Debugging script to find where the 3-parameter constructor error occurs

using LorenzParameterEstimation
using Optimisers

println("Testing exact train! call to isolate error...")

# This is exactly what the test was doing
true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
println("✓ True params created: ", true_params)

guess_params = L63Parameters(10.0, 17.0, 8.0/3.0) 
println("✓ Guess params created: ", guess_params)

sol = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
println("✓ Integration successful")

config = L63TrainingConfig(
    optimiser = Adam(0.01),
    epochs = 1,
    window_size = 50,
    verbose = true
)
println("✓ Config created")

# Now test the actual train! call with detailed error catching
try
    println("Calling train!...")
    result = train!(guess_params, sol, config)
    println("✓ Training successful")
catch e
    println("✗ Error in train!: ", e)
    
    # Print the full stack trace
    println("\nFull backtrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end