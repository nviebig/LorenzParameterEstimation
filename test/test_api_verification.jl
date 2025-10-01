# ========================================
# Quick Test Suite for API Verification
# ========================================

using LorenzParameterEstimation
using Test

@testset "API Verification" begin
    # Test core functionality
    @testset "Basic Components" begin
        # Types
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        @test params isa L63Parameters
        
        # Integration
        solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        @test solution isa L63Solution
        
        # Optimizer configs
        config = adam_config()
        @test config isa OptimizerConfig
        
        # Loss functions (matrix-based)
        pred = rand(10, 3)
        target = rand(10, 3)
        @test window_rmse(pred, target) ≥ 0
        @test window_mae(pred, target) ≥ 0
        @test window_mse(pred, target) ≥ 0
        
        println("✅ Basic components working")
    end
    
    @testset "Training APIs" begin
        # Create test data for training
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 5.0), 0.01)  # Longer trajectory for training
        initial_guess = L63Parameters(8.0, 25.0, 2.5)
        
        # Test train! API (primary API)
        @test_nowarn begin
            config = L63TrainingConfig(epochs=2, window_size=10, verbose=false)
            result = train!(initial_guess, solution, config)
            @test result isa Tuple  # Should return (best_params, metrics, history)
        end
        
        # TODO: Fix modular_train! API - has optimizer state compatibility issues
        # Currently has errors with optimizer state management for Float64 parameters
        
        println("✅ Training APIs verified (train! working)")
    end
end