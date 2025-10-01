# Loss Function Tests
@testset "User-Facing Loss Functions" begin
    @testset "window_rmse" begin
        predicted = rand(50, 3)
        target = predicted .+ 0.01 * randn(50, 3)
        
        loss = window_rmse(predicted, target)
        @test loss ≥ 0
        @test isa(loss, Float64)
        
        # Perfect prediction
        @test window_rmse(predicted, predicted) ≈ 0 atol=1e-10
    end
    
    @testset "window_mae" begin
        predicted = rand(20, 3)
        target = predicted .+ 0.02 * randn(20, 3)
        
        mae_loss = window_mae(predicted, target)
        @test mae_loss ≥ 0
        @test isa(mae_loss, Float64)
        @test window_mae(predicted, predicted) ≈ 0 atol=1e-10
    end
    
    @testset "window_mse" begin
        predicted = rand(30, 3)
        target = predicted .+ 0.05 * randn(30, 3)
        
        mse_loss = window_mse(predicted, target)
        @test mse_loss ≥ 0
        @test isa(mse_loss, Float64)
        @test window_mse(predicted, predicted) ≈ 0 atol=1e-10
    end
end

@testset "Gradient Computation" begin
    @testset "Basic Functionality" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        target_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        
        loss_val, gradients = compute_gradients(true_params, target_solution, 1, 20)
        
        @test isa(loss_val, Float64)
        @test loss_val ≥ 0
        @test isa(gradients, L63Parameters)
        @test all(isfinite.([gradients.σ, gradients.ρ, gradients.β]))
        @test loss_val < 1e-10  # Should be very small with exact parameters
    end
    
    @testset "Different Parameters" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        target_solution = integrate(true_params, [1.0, 1.0, 1.1], (0.0, 0.3), 0.01)
        
        test_params = L63Parameters(9.8, 27.5, 2.6)
        loss_val, gradients = compute_gradients(test_params, target_solution, 1, 15)
        
        @test loss_val > 0
        @test isa(gradients, L63Parameters)
        @test all(isfinite.([gradients.σ, gradients.ρ, gradients.β]))
        
        grad_norm = sqrt(gradients.σ^2 + gradients.ρ^2 + gradients.β^2)
        @test grad_norm > 1e-8
    end
end

@testset "Edge Cases" begin
    @testset "Stability Tests" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.3), 0.01)
        
        test_cases = [
            L63Parameters(10.0, 28.0, 8.0/3.0),
            L63Parameters(9.5, 27.5, 2.5),
            L63Parameters(11.0, 29.0, 3.0),
        ]
        
        for params in test_cases
            loss_val, gradients = compute_gradients_modular(params, solution, 1, 15, window_rmse)
            @test isfinite(loss_val)
            @test all(isfinite.([gradients.σ, gradients.ρ, gradients.β]))
            @test loss_val ≥ 0
        end
    end
end
