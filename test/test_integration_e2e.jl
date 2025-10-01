# ========================================
# End-to-End Integration Tests
# ========================================

@testset "Complete Workflow Tests" begin
    @testset "Basic Parameter Estimation Workflow" begin
        println("   Running complete parameter estimation workflow...")
        
        # 1. Generate "truth" data
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        true_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 3.0), 0.01)
        
        # 2. Create initial guess
        initial_guess = L63Parameters(8.0, 25.0, 2.5)
        
        # 3. Train using legacy API
        config = L63TrainingConfig(
            epochs = 50,
            window_size = 100,
            train_fraction = 0.8,
            verbose = false
        )
        
        best_params, metrics, history = train!(deepcopy(initial_guess), true_solution, config)
        
        # 4. Validate results
        @test best_params isa L63Parameters
        @test metrics.final_loss < metrics.initial_loss
        
        # Should be reasonably close to true parameters
        param_error = parameter_error(best_params, true_params)
        @test param_error < 2.0  # Allow some estimation error
        
        # 5. Test trained parameters by forward simulation
        estimated_solution = integrate(best_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        @test estimated_solution isa L63Solution
        @test all(isfinite.(estimated_solution.trajectory))
        
        println("   ✅ Basic workflow completed successfully")
    end
    
    @testset "Modern API Complete Workflow" begin
        println("   Running modern API workflow...")
        
        # 1. Setup true system
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        true_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.5), 0.01)
        
        # 2. Initial guess
        initial_guess = L63Parameters(9.0, 26.0, 2.8)
        
        # 3. Train using modern API
        result = modular_train!(
            deepcopy(initial_guess),
            true_solution;
            optimizer_config = adam_config(learning_rate=0.01),
            loss_function = window_rmse,
            epochs = 30,
            early_stopping_patience = 10,
            window_size = 80,
            verbose = false
        )
        
        # 4. Validate results
        @test result.best_params isa L63Parameters
        @test result.metrics.final_loss ≤ result.metrics.initial_loss
        
        # Should converge reasonably well
        param_error = parameter_error(result.best_params, true_params)
        @test param_error < 3.0
        
        # 5. Test forward prediction
        prediction = integrate(result.best_params, [1.0, 1.0, 1.1], (0.0, 1.0), 0.01)
        @test all(isfinite.(prediction.trajectory))
        
        println("   ✅ Modern API workflow completed successfully")
    end
    
    @testset "Multi-Optimizer Comparison" begin
        println("   Comparing different optimizers...")
        
        # Setup
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        true_solution = integrate(true_params, [1.0, 1.0, 1.1], (0.0, 2.0), 0.01)
        initial_guess = L63Parameters(8.5, 25.5, 2.3)
        
        optimizers = [
            ("Adam", adam_config()),
            ("SGD", sgd_config(learning_rate=0.01)),
            ("AdamW", adamw_config()),
            ("RMSprop", rmsprop_config())
        ]
        
        results = Dict()
        
        for (name, opt_config) in optimizers
            result = modular_train!(
                deepcopy(initial_guess),
                true_solution;
                optimizer_config = opt_config,
                loss_function = window_rmse,
                epochs = 20,
                verbose = false
            )
            
            results[name] = result
            
            # All should improve from initial guess
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
            @test parameter_error(result.best_params, true_params) < 5.0
        end
        
        # At least one should converge well
        final_errors = [parameter_error(results[name].best_params, true_params) for (name, _) in optimizers]
        @test minimum(final_errors) < 2.0
        
        println("   ✅ Multi-optimizer comparison completed")
    end
    
    @testset "Different Loss Function Comparison" begin
        println("   Comparing different loss functions...")
        
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        true_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        initial_guess = L63Parameters(9.2, 26.8, 2.7)
        
        loss_functions = [
            ("RMSE", window_rmse),
            ("MAE", window_mae),
            ("MSE", window_mse),
            ("Adaptive", adaptive_loss)
        ]
        
        results = Dict()
        
        for (name, loss_fn) in loss_functions
            result = modular_train!(
                deepcopy(initial_guess),
                true_solution;
                optimizer_config = adam_config(learning_rate=0.01),
                loss_function = loss_fn,
                epochs = 25,
                verbose = false
            )
            
            results[name] = result
            
            # All should improve
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
            @test parameter_error(result.best_params, true_params) < 4.0
        end
        
        # Different loss functions should give somewhat different results
        rmse_params = results["RMSE"].best_params
        mae_params = results["MAE"].best_params
        @test parameter_error(rmse_params, mae_params) > 0.01  # Should be different
        
        println("   ✅ Loss function comparison completed")
    end
end

@testset "Robustness and Edge Cases" begin
    @testset "Noisy Data Estimation" begin
        println("   Testing robustness to noise...")
        
        # Generate clean data
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        clean_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 3.0), 0.01)
        
        # Add different levels of noise
        Random.seed!(42)
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        for noise_level in noise_levels
            # Create noisy trajectory
            noise = noise_level * randn(size(clean_solution.trajectory))
            noisy_trajectory = clean_solution.trajectory + noise
            noisy_solution = L63Solution(clean_solution.times, noisy_trajectory, clean_solution.system)
            
            # Estimate parameters
            initial_guess = L63Parameters(8.0, 25.0, 2.5)
            
            result = modular_train!(
                deepcopy(initial_guess),
                noisy_solution;
                optimizer_config = adam_config(learning_rate=0.005),
                loss_function = adaptive_loss,  # Robust to outliers
                epochs = 40,
                verbose = false
            )
            
            # Should still converge (with decreasing accuracy as noise increases)
            param_error = parameter_error(result.best_params, true_params)
            
            if noise_level ≤ 0.05
                @test param_error < 2.0  # Should be quite accurate for low noise
            elseif noise_level ≤ 0.1
                @test param_error < 4.0  # Moderate accuracy for medium noise
            else
                @test param_error < 8.0  # At least somewhat reasonable for high noise
            end
            
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
        end
        
        println("   ✅ Noise robustness test completed")
    end
    
    @testset "Different System Regimes" begin
        println("   Testing different parameter regimes...")
        
        # Test different parameter regimes
        test_cases = [
            ("Classic Chaotic", L63Parameters(10.0, 28.0, 8.0/3.0)),
            ("Stable Fixed Point", L63Parameters(10.0, 0.5, 8.0/3.0)),
            ("Periodic", L63Parameters(10.0, 99.65, 8.0/3.0)),
            ("High ρ Chaotic", L63Parameters(10.0, 50.0, 8.0/3.0)),
        ]
        
        for (regime_name, true_params) in test_cases
            # Generate data
            solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 4.0), 0.01)
            
            # Create initial guess
            initial_guess = true_params + L63Parameters(
                1.0 + 0.5 * randn(),
                3.0 + 1.0 * randn(),
                0.3 + 0.2 * randn()
            )
            
            # Estimate
            result = modular_train!(
                initial_guess,
                solution;
                optimizer_config = adam_config(learning_rate=0.01),
                loss_function = window_rmse,
                epochs = 30,
                verbose = false
            )
            
            # Should improve from initial guess
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
            
            # Should get reasonably close (different regimes have different sensitivities)
            param_error = parameter_error(result.best_params, true_params)
            @test param_error < 10.0  # Generous bound for all regimes
            
            # Estimated parameters should be physically reasonable
            @test result.best_params.σ > 0  # Usually positive
            @test result.best_params.β > 0  # Usually positive
            # ρ can be negative in some contexts
        end
        
        println("   ✅ Different regime test completed")
    end
    
    @testset "Short and Long Trajectories" begin
        println("   Testing different trajectory lengths...")
        
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        initial_guess = L63Parameters(9.5, 27.0, 2.8)
        
        # Test different trajectory lengths
        trajectory_lengths = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        for T in trajectory_lengths
            solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, T), 0.01)
            
            # Adjust window size based on trajectory length
            window_size = min(50, max(10, Int(round(T / 0.01 / 5))))
            
            result = modular_train!(
                deepcopy(initial_guess),
                solution;
                optimizer_config = adam_config(learning_rate=0.01),
                loss_function = window_rmse,
                epochs = 20,
                window_size = window_size,
                verbose = false
            )
            
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
            
            # Longer trajectories should generally give better estimates
            param_error = parameter_error(result.best_params, true_params)
            if T ≥ 2.0
                @test param_error < 3.0  # Should be quite good for long trajectories
            else
                @test param_error < 8.0  # More lenient for short trajectories
            end
        end
        
        println("   ✅ Trajectory length test completed")
    end
end

@testset "Performance and Scalability" begin
    @testset "Training Time Scaling" begin
        println("   Testing training time scaling...")
        
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.1], (0.0, 5.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        # Test different epoch counts
        epoch_counts = [5, 10, 20]
        training_times = Float64[]
        
        for epochs in epoch_counts
            start_time = time()
            
            result = modular_train!(
                deepcopy(initial_guess),
                solution;
                optimizer_config = adam_config(),
                loss_function = window_rmse,
                epochs = epochs,
                verbose = false
            )
            
            elapsed = time() - start_time
            push!(training_times, elapsed)
            
            # Should still work and improve
            @test result.metrics.final_loss ≤ result.metrics.initial_loss
        end
        
        # Training time should generally increase with epochs (allowing some variation)
        @test training_times[end] ≥ training_times[1] * 0.5  # At least some scaling
        
        println("   ✅ Training time scaling test completed")
    end
    
    @testset "Memory Usage" begin
        println("   Testing memory efficiency...")
        
        # Test with larger trajectories
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        large_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.001)  # 10,000 points
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        # Should handle large datasets without issues
        @test_nowarn begin
            result = modular_train!(
                initial_guess,
                large_solution;
                optimizer_config = adam_config(),
                loss_function = window_rmse,
                epochs = 5,
                window_size = 100,
                verbose = false
            )
        end
        
        println("   ✅ Memory usage test completed")
    end
end

@testset "API Consistency and Error Handling" begin
    @testset "Invalid Inputs" begin
        println("   Testing error handling...")
        
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        # Should handle various edge cases gracefully
        
        # Very small window size
        @test_nowarn modular_train!(
            deepcopy(initial_guess), solution;
            optimizer_config = adam_config(),
            loss_function = window_rmse,
            epochs = 3,
            window_size = 1,
            verbose = false
        )
        
        # Zero epochs should return initial parameters
        result_zero = modular_train!(
            deepcopy(initial_guess), solution;
            optimizer_config = adam_config(),
            loss_function = window_rmse,
            epochs = 0,
            verbose = false
        )
        @test parameter_error(result_zero.best_params, initial_guess) < 1e-10
        
        println("   ✅ Error handling test completed")
    end
    
    @testset "API Consistency" begin
        println("   Testing API consistency...")
        
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.1], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        # Legacy API
        config = L63TrainingConfig(epochs=5, verbose=false, window_size=30)
        best_params_legacy, metrics_legacy, history_legacy = train!(deepcopy(initial_guess), solution, config)
        
        # Modern API with equivalent settings
        result_modern = modular_train!(
            deepcopy(initial_guess), solution;
            optimizer_config = adam_config(),  # Default optimizer
            loss_function = window_rmse,       # Default loss
            epochs = 5,
            window_size = 30,
            verbose = false
        )
        
        # Both should improve from initial guess
        initial_loss = window_rmse(initial_guess, solution, 1, 30)
        @test metrics_legacy.final_loss ≤ initial_loss
        @test result_modern.metrics.final_loss ≤ initial_loss
        
        # Both should return reasonable parameter estimates
        @test parameter_error(best_params_legacy, true_params) < 10.0
        @test parameter_error(result_modern.best_params, true_params) < 10.0
        
        println("   ✅ API consistency test completed")
    end
end