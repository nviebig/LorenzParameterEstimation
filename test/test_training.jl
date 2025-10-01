# ========================================
# Training API Tests
# ========================================

@testset "Legacy Training API (train!)" begin
    @testset "Basic Training" begin
        # Setup test data
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        initial_guess = L63Parameters(8.0, 25.0, 2.5)
        
        # Basic configuration
        config = L63TrainingConfig(
            epochs = 10,
            window_size = 50,
            verbose = false
        )
        
        # Train and test
        best_params, metrics, history = train!(deepcopy(initial_guess), solution, config)
        
        @test best_params isa L63Parameters
        @test metrics isa NamedTuple
        @test history isa NamedTuple
        
        # Should have improved from initial guess
        initial_loss = window_rmse(initial_guess, solution, 1, 50)
        final_loss = window_rmse(best_params, solution, 1, 50)
        @test final_loss ≤ initial_loss
        
        # History should have correct length
        @test length(history.loss_history) == config.epochs
        @test length(history.param_history) == config.epochs
    end
    
    @testset "Different Optimizers" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.8)
        
        optimizers = [
            ("Adam", adam_config().optimizer),
            ("SGD", sgd_config(learning_rate=0.01).optimizer),
            ("AdamW", adamw_config().optimizer),
            ("RMSprop", rmsprop_config().optimizer)
        ]
        
        for (name, optimizer) in optimizers
            config = L63TrainingConfig(
                optimiser = optimizer,
                epochs = 5,
                window_size = 30,
                verbose = false
            )
            
            @test_nowarn train!(deepcopy(initial_guess), solution, config)
        end
    end
    
    @testset "Different Loss Functions" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(9.5, 27.0, 2.7)
        
        loss_functions = [
            ("RMSE", window_rmse),
            ("MAE", window_mae),
            ("MSE", window_mse),
            ("Adaptive", adaptive_loss)
        ]
        
        for (name, loss_fn) in loss_functions
            config = L63TrainingConfig(
                loss = loss_fn,
                epochs = 5,
                window_size = 25,
                verbose = false
            )
            
            best_params, _, _ = train!(deepcopy(initial_guess), solution, config)
            @test best_params isa L63Parameters
        end
    end
    
    @testset "Parameter Updates" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(8.0, 25.0, 2.0)
        
        # Test selective parameter updates
        configs = [
            L63TrainingConfig(epochs=5, update_σ=true, update_ρ=false, update_β=false, verbose=false),
            L63TrainingConfig(epochs=5, update_σ=false, update_ρ=true, update_β=false, verbose=false),
            L63TrainingConfig(epochs=5, update_σ=false, update_ρ=false, update_β=true, verbose=false),
            L63TrainingConfig(epochs=5, update_σ=true, update_ρ=true, update_β=true, verbose=false)
        ]
        
        for config in configs
            best_params, _, _ = train!(deepcopy(initial_guess), solution, config)
            
            if !config.update_σ
                @test best_params.σ == initial_guess.σ
            end
            if !config.update_ρ
                @test best_params.ρ == initial_guess.ρ
            end
            if !config.update_β
                @test best_params.β == initial_guess.β
            end
        end
    end
    
    @testset "Training Configuration Validation" begin
        # Default configuration
        config_default = L63TrainingConfig()
        @test config_default.epochs > 0
        @test config_default.window_size > 0
        @test 0 < config_default.train_fraction < 1
        @test config_default.batch_size > 0
        
        # Custom configuration
        config_custom = L63TrainingConfig(
            epochs = 100,
            η = 0.01,
            window_size = 200,
            train_fraction = 0.7,
            batch_size = 32,
            shuffle = false,
            verbose = true
        )
        
        @test config_custom.epochs == 100
        @test config_custom.η == 0.01
        @test config_custom.window_size == 200
        @test config_custom.train_fraction == 0.7
        @test config_custom.batch_size == 32
        @test config_custom.shuffle == false
        @test config_custom.verbose == true
    end
end

@testset "Modern Training API (modular_train!)" begin
    @testset "Basic Modular Training" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.5), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.8)
        
        result = modular_train!(
            deepcopy(initial_guess),
            solution;
            optimizer_config = adam_config(learning_rate=0.01),
            loss_function = window_rmse,
            epochs = 10,
            early_stopping_patience = 5,
            verbose = false
        )
        
        @test result isa NamedTuple
        @test haskey(result, :best_params)
        @test haskey(result, :metrics)
        @test haskey(result, :history)
        
        @test result.best_params isa L63Parameters
        
        # Should have improved
        initial_loss = window_rmse(initial_guess, solution, 1, 50)
        final_loss = result.metrics.final_loss
        @test final_loss ≤ initial_loss
    end
    
    @testset "Early Stopping" begin
        # Create a case where parameters are already very close to optimal
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(10.01, 28.01, 8.0/3.0 + 0.01)  # Very close
        
        result = modular_train!(
            deepcopy(initial_guess),
            solution;
            optimizer_config = adam_config(learning_rate=0.001),
            loss_function = window_rmse,
            epochs = 100,
            early_stopping_patience = 3,
            verbose = false
        )
        
        # Should stop early due to convergence
        @test length(result.history.loss_history) < 100
        @test result.metrics.converged == true
    end
    
    @testset "Different Modular Configurations" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(8.5, 25.5, 2.5)
        
        # Test different optimizer configs
        optimizer_configs = [
            adam_config(),
            sgd_config(learning_rate=0.01),
            adamw_config(learning_rate=0.001),
            rmsprop_config()
        ]
        
        for opt_config in optimizer_configs
            result = modular_train!(
                deepcopy(initial_guess),
                solution;
                optimizer_config = opt_config,
                loss_function = window_rmse,
                epochs = 5,
                verbose = false
            )
            
            @test result.best_params isa L63Parameters
            @test result.metrics.final_loss ≥ 0
        end
        
        # Test different loss functions
        loss_functions = [window_rmse, window_mae, window_mse, adaptive_loss]
        
        for loss_fn in loss_functions
            result = modular_train!(
                deepcopy(initial_guess),
                solution;
                optimizer_config = adam_config(),
                loss_function = loss_fn,
                epochs = 5,
                verbose = false
            )
            
            @test result.best_params isa L63Parameters
            @test result.metrics.final_loss ≥ 0
        end
    end
    
    @testset "Parameter Selection" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(8.0, 25.0, 2.0)
        
        # Test parameter selection masks
        masks = [
            [true, false, false],   # Only σ
            [false, true, false],   # Only ρ
            [false, false, true],   # Only β
            [true, true, false],    # σ and ρ
            [true, true, true]      # All parameters
        ]
        
        for mask in masks
            result = modular_train!(
                deepcopy(initial_guess),
                solution;
                optimizer_config = adam_config(learning_rate=0.01),
                loss_function = window_rmse,
                epochs = 5,
                update_mask = mask,
                verbose = false
            )
            
            # Check that only selected parameters were updated
            if !mask[1]  # σ not updated
                @test result.best_params.σ == initial_guess.σ
            end
            if !mask[2]  # ρ not updated  
                @test result.best_params.ρ == initial_guess.ρ
            end
            if !mask[3]  # β not updated
                @test result.best_params.β == initial_guess.β
            end
        end
    end
end

@testset "Training Robustness" begin
    @testset "Noisy Data" begin
        # Generate clean trajectory
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        clean_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        
        # Add noise to trajectory
        Random.seed!(12345)
        noise_level = 0.1
        noisy_trajectory = clean_solution.trajectory + noise_level * randn(size(clean_solution.trajectory))
        noisy_solution = L63Solution(clean_solution.times, noisy_trajectory, clean_solution.system)
        
        initial_guess = L63Parameters(8.0, 25.0, 2.5)
        
        # Train on noisy data
        config = L63TrainingConfig(
            epochs = 20,
            window_size = 30,
            verbose = false
        )
        
        best_params, _, _ = train!(deepcopy(initial_guess), noisy_solution, config)
        
        # Should still converge to reasonable values
        @test abs(best_params.σ - true_params.σ) < 2.0
        @test abs(best_params.ρ - true_params.ρ) < 5.0
        @test abs(best_params.β - true_params.β) < 1.0
    end
    
    @testset "Different Initial Guesses" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.5), 0.01)
        
        # Test various initial guesses
        initial_guesses = [
            L63Parameters(5.0, 15.0, 1.0),    # Far from true
            L63Parameters(15.0, 35.0, 5.0),   # Also far
            L63Parameters(10.1, 28.1, 2.7),   # Close to true
            L63Parameters(1.0, 1.0, 1.0),     # Very different
        ]
        
        for guess in initial_guesses
            config = L63TrainingConfig(
                epochs = 15,
                window_size = 40,
                verbose = false
            )
            
            best_params, metrics, _ = train!(deepcopy(guess), solution, config)
            
            @test best_params isa L63Parameters
            @test metrics.final_loss < metrics.initial_loss
            @test all(isfinite.([best_params.σ, best_params.ρ, best_params.β]))
        end
    end
    
    @testset "Edge Case Parameters" begin
        # Test training with extreme parameter values
        solution = integrate(L63Parameters(10.0, 28.0, 8.0/3.0), [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        edge_cases = [
            L63Parameters(1e-3, 1e-3, 1e-3),      # Very small
            L63Parameters(100.0, 100.0, 100.0),   # Large
            L63Parameters(10.0, 1.0, 8.0/3.0),    # ρ < σ (non-chaotic)
        ]
        
        for params in edge_cases
            config = L63TrainingConfig(
                epochs = 5,
                window_size = 20,
                verbose = false
            )
            
            @test_nowarn train!(deepcopy(params), solution, config)
        end
    end
end

@testset "Training Metrics and History" begin
    @testset "Metrics Content" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.1], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        config = L63TrainingConfig(epochs=10, verbose=false)
        best_params, metrics, history = train!(initial_guess, solution, config)
        
        # Check metrics structure
        @test haskey(metrics, :initial_loss)
        @test haskey(metrics, :final_loss)
        @test haskey(metrics, :best_loss)
        @test haskey(metrics, :improvement)
        
        @test metrics.initial_loss ≥ 0
        @test metrics.final_loss ≥ 0
        @test metrics.best_loss ≥ 0
        @test metrics.final_loss ≤ metrics.initial_loss  # Should improve
        @test metrics.best_loss ≤ metrics.initial_loss
    end
    
    @testset "History Tracking" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        initial_guess = L63Parameters(8.5, 25.5, 2.3)
        
        config = L63TrainingConfig(epochs=8, verbose=false)
        _, _, history = train!(initial_guess, solution, config)
        
        # Check history structure
        @test haskey(history, :loss_history)
        @test haskey(history, :param_history)
        
        @test length(history.loss_history) == config.epochs
        @test length(history.param_history) == config.epochs
        
        # Loss should generally decrease (allowing for some fluctuation)
        @test history.loss_history[end] ≤ history.loss_history[1] * 1.1
        
        # Parameter history should contain L63Parameters
        @test all(p isa L63Parameters for p in history.param_history)
    end
end