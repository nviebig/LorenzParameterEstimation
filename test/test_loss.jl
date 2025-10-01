# ========================================
# Loss Function Tests
# ========================================

@testset "User-Facing Loss Functions" begin
    @testset "window_rmse" begin
        # Generate test data
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        
        # Test with same parameters (should give very low loss)
        loss = window_rmse(true_params, solution, 1, 50)
        @test loss ≥ 0
        @test loss < 0.1  # Should be very low for correct parameters
        
        # Test with different parameters (should give higher loss)
        wrong_params = L63Parameters(8.0, 25.0, 2.0)
        loss_wrong = window_rmse(wrong_params, solution, 1, 50)
        @test loss_wrong > loss
        @test loss_wrong > 0.1
        
        # Test different window positions
        loss1 = window_rmse(true_params, solution, 1, 50)
        loss2 = window_rmse(true_params, solution, 51, 50)
        @test loss1 ≥ 0
        @test loss2 ≥ 0
        # Losses can be different due to chaotic dynamics
    end
    
    @testset "window_mae" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        loss = window_mae(true_params, solution, 1, 30)
        @test loss ≥ 0
        @test isfinite(loss)
        
        # MAE should be different from RMSE for same data
        rmse_loss = window_rmse(true_params, solution, 1, 30)
        @test loss != rmse_loss
    end
    
    @testset "window_mse" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        mse_loss = window_mse(true_params, solution, 1, 30)
        rmse_loss = window_rmse(true_params, solution, 1, 30)
        
        @test mse_loss ≥ 0
        @test rmse_loss ≥ 0
        @test mse_loss ≈ rmse_loss^2 atol=1e-10
    end
    
    @testset "adaptive_loss" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        loss = adaptive_loss(true_params, solution, 1, 30)
        @test loss ≥ 0
        @test isfinite(loss)
        
        # Should be different from MSE/MAE (Huber loss)
        mse_loss = window_mse(true_params, solution, 1, 30)
        @test loss != mse_loss
    end
    
    @testset "weighted_window_loss" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        # Equal weights should be similar to regular MSE
        weights = [1.0, 1.0, 1.0]
        weighted_loss = weighted_window_loss(true_params, solution, 1, 30, weights)
        mse_loss = window_mse(true_params, solution, 1, 30)
        
        @test weighted_loss ≥ 0
        @test abs(weighted_loss - mse_loss) < 0.1 * mse_loss
        
        # Different weights should give different results
        weights_unequal = [2.0, 1.0, 0.5]
        weighted_loss2 = weighted_window_loss(true_params, solution, 1, 30, weights_unequal)
        @test weighted_loss2 != weighted_loss
    end
end

@testset "Gradient Computation" begin
    @testset "compute_gradients_modular Basic" begin
        # Setup test data
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        # Test parameters (slightly different)
        test_params = L63Parameters(9.5, 27.0, 2.8)
        
        loss_val, gradients = compute_gradients_modular(
            test_params, solution, 1, 30, window_rmse
        )
        
        @test loss_val ≥ 0
        @test isfinite(loss_val)
        @test length(gradients) == 3  # σ, ρ, β gradients
        @test all(isfinite.(gradients))
        
        # Gradients should not all be zero (unless at exact minimum)
        @test any(abs.(gradients) .> 1e-6)
    end
    
    @testset "Gradient Consistency" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        test_params = L63Parameters(9.8, 27.5, 2.7)
        
        # Test same computation multiple times
        loss1, grad1 = compute_gradients_modular(test_params, solution, 1, 20, window_rmse)
        loss2, grad2 = compute_gradients_modular(test_params, solution, 1, 20, window_rmse)
        
        @test loss1 ≈ loss2 atol=1e-12
        @test grad1 ≈ grad2 atol=1e-12
    end
    
    @testset "Different Loss Functions Gradients" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        test_params = L63Parameters(9.5, 27.0, 2.8)
        
        loss_fns = [window_rmse, window_mae, window_mse, adaptive_loss]
        results = []
        
        for loss_fn in loss_fns
            loss_val, grad = compute_gradients_modular(test_params, solution, 1, 20, loss_fn)
            push!(results, (loss_val, grad))
            
            @test loss_val ≥ 0
            @test isfinite(loss_val)
            @test length(grad) == 3
            @test all(isfinite.(grad))
        end
        
        # Different loss functions should give different gradients
        @test results[1][2] != results[2][2]  # RMSE vs MAE
        @test results[1][2] != results[3][2]  # RMSE vs MSE
    end
    
    @testset "Finite Difference Validation" begin
        # Validate gradients using finite differences
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.3), 0.01)
        test_params = L63Parameters(9.8, 27.5, 2.7)
        
        loss_val, analytical_grad = compute_gradients_modular(
            test_params, solution, 1, 15, window_rmse
        )
        
        # Finite difference approximation
        h = 1e-6
        finite_diff_grad = zeros(3)
        
        # σ gradient
        params_plus = L63Parameters(test_params.σ + h, test_params.ρ, test_params.β)
        params_minus = L63Parameters(test_params.σ - h, test_params.ρ, test_params.β)
        loss_plus = window_rmse(params_plus, solution, 1, 15)
        loss_minus = window_rmse(params_minus, solution, 1, 15)
        finite_diff_grad[1] = (loss_plus - loss_minus) / (2h)
        
        # ρ gradient
        params_plus = L63Parameters(test_params.σ, test_params.ρ + h, test_params.β)
        params_minus = L63Parameters(test_params.σ, test_params.ρ - h, test_params.β)
        loss_plus = window_rmse(params_plus, solution, 1, 15)
        loss_minus = window_rmse(params_minus, solution, 1, 15)
        finite_diff_grad[2] = (loss_plus - loss_minus) / (2h)
        
        # β gradient
        params_plus = L63Parameters(test_params.σ, test_params.ρ, test_params.β + h)
        params_minus = L63Parameters(test_params.σ, test_params.ρ, test_params.β - h)
        loss_plus = window_rmse(params_plus, solution, 1, 15)
        loss_minus = window_rmse(params_minus, solution, 1, 15)
        finite_diff_grad[3] = (loss_plus - loss_minus) / (2h)
        
        # Compare analytical and finite difference gradients
        rel_error = abs.(analytical_grad - finite_diff_grad) ./ (abs.(finite_diff_grad) .+ 1e-8)
        @test all(rel_error .< 1e-3)  # 0.1% relative error tolerance
    end
end

@testset "Edge Cases and Robustness" begin
    @testset "Very Short Windows" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        
        # Very short window
        @test_nowarn window_rmse(true_params, solution, 1, 1)
        @test_nowarn window_rmse(true_params, solution, 1, 2)
    end
    
    @testset "Window at End of Trajectory" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        n_points = length(solution.times)
        window_size = 10
        
        # Should work near the end
        @test_nowarn window_rmse(true_params, solution, n_points - window_size, window_size)
        
        # Should handle boundary correctly
        loss = window_rmse(true_params, solution, n_points - window_size + 1, window_size)
        @test isfinite(loss)
        @test loss ≥ 0
    end
    
    @testset "Different Parameter Ranges" begin
        solution = integrate(L63Parameters(10.0, 28.0, 8.0/3.0), [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        
        # Very small parameters
        small_params = L63Parameters(0.1, 0.1, 0.1)
        @test_nowarn window_rmse(small_params, solution, 1, 20)
        
        # Large parameters
        large_params = L63Parameters(100.0, 100.0, 100.0)
        @test_nowarn window_rmse(large_params, solution, 1, 20)
        
        # Mixed scale parameters
        mixed_params = L63Parameters(1e-3, 1e3, 1.0)
        @test_nowarn window_rmse(mixed_params, solution, 1, 20)
    end
    
    @testset "Numerical Stability" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.3), 0.01)
        
        # Parameters that might cause numerical issues
        test_cases = [
            L63Parameters(10.0, 28.0, 8.0/3.0),  # Exact match
            L63Parameters(1e-8, 1e-8, 1e-8),     # Very small
            L63Parameters(1e8, 1e8, 1e8),        # Very large
            L63Parameters(10.0, -28.0, 8.0/3.0), # Negative ρ
            L63Parameters(-10.0, 28.0, 8.0/3.0), # Negative σ
        ]
        
        for params in test_cases
            loss_val, gradients = compute_gradients_modular(params, solution, 1, 15, window_rmse)
            @test isfinite(loss_val)
            @test all(isfinite.(gradients))
            @test loss_val ≥ 0
        end
    end
end