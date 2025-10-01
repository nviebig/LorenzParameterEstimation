# ========================================
# LorenzParameterEstimation.jl Test Suite
# ========================================
# 
# Comprehensive test suite following Julia best practices
# Run with: julia --project=. -e "using Pkg; Pkg.test()"
# Or: julia --project=. test/runtests.jl

using Test
using LorenzParameterEstimation
using Random
using LinearAlgebra
using Statistics

# Set random seed for reproducible tests
Random.seed!(12345)

println("🧪 Starting LorenzParameterEstimation.jl Test Suite")
println("=" ^ 60)

# Test suite structure
const TEST_MODULES = [
    ("Core Types", "test_types.jl"),
    ("Integration", "test_integration.jl"), 
    ("Loss Functions", "test_loss.jl"),
    ("Optimizers", "test_optimizers.jl"),
    ("Training APIs", "test_training.jl"),
    ("Utilities", "test_utils.jl"),
    ("End-to-End", "test_integration_e2e.jl"),
    ("Code Quality", "test_code_quality.jl")
]

# Run all test modules
@testset "LorenzParameterEstimation.jl" begin
    for (description, test_file) in TEST_MODULES
        @testset "$description" begin
            println("🔍 Testing: $description")
            include(test_file)
        end
    end
end

println("\n✅ All tests completed!")
println("📊 Test coverage and performance metrics available in logs")