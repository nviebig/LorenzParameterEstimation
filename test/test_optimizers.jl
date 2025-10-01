# ========================================
# Optimizer Configuration Tests
# ========================================

@testset "OptimizerConfig Creation" begin
    @testset "adam_config" begin
        # Default configuration
        config = adam_config()
        @test config isa OptimizerConfig
        @test config.name == "Adam"
        @test config.optimizer isa Optimisers.Adam
        
        # Custom learning rate
        config_custom = adam_config(learning_rate=0.001)
        @test config_custom.name == "Adam"
        @test config_custom.optimizer.eta == 0.001
        
        # Different parameters
        config_full = adam_config(learning_rate=0.002, beta1=0.8, beta2=0.95, epsilon=1e-6)
        @test config_full.optimizer.eta == 0.002
        @test config_full.optimizer.beta[1] == 0.8
        @test config_full.optimizer.beta[2] == 0.95
        @test config_full.optimizer.epsilon == 1e-6
    end
    
    @testset "sgd_config" begin
        config = sgd_config()
        @test config isa OptimizerConfig
        @test config.name == "SGD"
        @test config.optimizer isa Optimisers.Descent
        
        config_custom = sgd_config(learning_rate=0.1)
        @test config_custom.optimizer.eta == 0.1
    end
    
    @testset "adamw_config" begin
        config = adamw_config()
        @test config isa OptimizerConfig
        @test config.name == "AdamW"
        @test config.optimizer isa Optimisers.AdamW
        
        config_custom = adamw_config(learning_rate=0.001, weight_decay=1e-4)
        @test config_custom.optimizer.eta == 0.001
        @test config_custom.optimizer.lambda == 1e-4
    end
    
    @testset "adagrad_config" begin
        config = adagrad_config()
        @test config isa OptimizerConfig
        @test config.name == "AdaGrad"
        @test config.optimizer isa Optimisers.AdaGrad
        
        config_custom = adagrad_config(learning_rate=0.01, epsilon=1e-6)
        @test config_custom.optimizer.eta == 0.01
        @test config_custom.optimizer.epsilon == 1e-6
    end
    
    @testset "rmsprop_config" begin
        config = rmsprop_config()
        @test config isa OptimizerConfig
        @test config.name == "RMSprop"
        @test config.optimizer isa Optimisers.RMSProp
        
        config_custom = rmsprop_config(learning_rate=0.001, rho=0.9, epsilon=1e-6)
        @test config_custom.optimizer.eta == 0.001
        @test config_custom.optimizer.rho == 0.9
        @test config_custom.optimizer.epsilon == 1e-6
    end
end

@testset "Optimizer Integration" begin
    @testset "Optimisers.jl Compatibility" begin
        # Test that our configs work with Optimisers.jl setup/update
        config = adam_config(learning_rate=0.01)
        params = [1.0, 2.0, 3.0]  # Dummy parameters
        grads = [0.1, -0.2, 0.3]  # Dummy gradients
        
        # Test setup
        opt_state = Optimisers.setup(config.optimizer, params)
        @test opt_state isa NamedTuple
        
        # Test update
        new_opt_state, new_params = Optimisers.update(opt_state, params, grads)
        @test length(new_params) == length(params)
        @test new_params != params  # Should have updated
        
        # Check update direction (for Adam, should be in opposite direction of gradient)
        @test (new_params[1] - params[1]) < 0  # Gradient was positive, so param should decrease
        @test (new_params[2] - params[2]) > 0  # Gradient was negative, so param should increase
        @test (new_params[3] - params[3]) < 0  # Gradient was positive, so param should decrease
    end
    
    @testset "L63Parameters Optimization" begin
        # Test optimization with actual L63Parameters structure
        config = adam_config(learning_rate=0.1)
        initial_params = L63Parameters(9.0, 25.0, 2.5)
        target_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Convert to parameter vector for optimization
        param_vec = [initial_params.σ, initial_params.ρ, initial_params.β]
        target_vec = [target_params.σ, target_params.ρ, target_params.β]
        
        # Simple quadratic loss for testing
        grad_vec = 2.0 * (param_vec - target_vec)
        
        opt_state = Optimisers.setup(config.optimizer, param_vec)
        new_opt_state, new_param_vec = Optimisers.update(opt_state, param_vec, grad_vec)
        
        # Should move towards target
        @test norm(new_param_vec - target_vec) < norm(param_vec - target_vec)
        
        # Convert back to L63Parameters
        new_params = L63Parameters(new_param_vec[1], new_param_vec[2], new_param_vec[3])
        @test new_params isa L63Parameters
    end
end

@testset "Optimizer Behavior" begin
    @testset "Learning Rate Effects" begin
        # Test different learning rates with simple quadratic function
        param_vec = [5.0]
        target = [0.0]
        grad = 2.0 * (param_vec - target)  # Gradient of (x-0)^2
        
        # High learning rate
        config_high = adam_config(learning_rate=1.0)
        state_high = Optimisers.setup(config_high.optimizer, param_vec)
        _, params_high = Optimisers.update(state_high, param_vec, grad)
        
        # Low learning rate  
        config_low = adam_config(learning_rate=0.01)
        state_low = Optimisers.setup(config_low.optimizer, param_vec)
        _, params_low = Optimisers.update(state_low, param_vec, grad)
        
        # High learning rate should make bigger step
        step_high = abs(params_high[1] - param_vec[1])
        step_low = abs(params_low[1] - param_vec[1])
        @test step_high > step_low
    end
    
    @testset "Convergence Properties" begin
        # Test basic convergence on simple problem
        config = adam_config(learning_rate=0.1)
        param_vec = [10.0, -5.0, 3.0]
        target = [0.0, 0.0, 0.0]
        
        opt_state = Optimisers.setup(config.optimizer, param_vec)
        current_params = copy(param_vec)
        
        # Run optimization steps
        for i in 1:100
            grad = 2.0 * (current_params - target)  # Quadratic loss gradient
            opt_state, current_params = Optimisers.update(opt_state, current_params, grad)
        end
        
        # Should converge close to target
        @test norm(current_params - target) < 0.1
    end
    
    @testset "Momentum Effects" begin
        # Compare SGD with and without momentum on oscillatory function
        param_vec = [1.0]
        
        # SGD without momentum
        config_sgd = sgd_config(learning_rate=0.1)
        
        # Adam (which has momentum-like behavior)
        config_adam = adam_config(learning_rate=0.1)
        
        state_sgd = Optimisers.setup(config_sgd.optimizer, param_vec)
        state_adam = Optimisers.setup(config_adam.optimizer, param_vec)
        
        params_sgd = copy(param_vec)
        params_adam = copy(param_vec)
        
        # Simulate oscillatory gradients
        for i in 1:10
            grad = [sin(i)]  # Oscillatory gradient
            
            state_sgd, params_sgd = Optimisers.update(state_sgd, params_sgd, grad)
            state_adam, params_adam = Optimisers.update(state_adam, params_adam, grad)
        end
        
        # Both should be finite and different
        @test all(isfinite.(params_sgd))
        @test all(isfinite.(params_adam))
        @test params_sgd != params_adam
    end
end

@testset "Configuration Validation" begin
    @testset "Parameter Bounds" begin
        # Learning rates should be positive
        @test_nowarn adam_config(learning_rate=1e-6)  # Very small but positive
        @test_nowarn adam_config(learning_rate=1.0)   # Standard range
        
        # Beta parameters should be in [0, 1)
        @test_nowarn adam_config(beta1=0.0, beta2=0.0)
        @test_nowarn adam_config(beta1=0.9, beta2=0.999)
        
        # Epsilon should be small positive
        @test_nowarn adam_config(epsilon=1e-8)
        @test_nowarn adam_config(epsilon=1e-6)
        
        # Weight decay should be non-negative
        @test_nowarn adamw_config(weight_decay=0.0)
        @test_nowarn adamw_config(weight_decay=1e-4)
    end
    
    @testset "Type Consistency" begin
        # All configs should return OptimizerConfig
        configs = [
            adam_config(),
            sgd_config(), 
            adamw_config(),
            adagrad_config(),
            rmsprop_config()
        ]
        
        for config in configs
            @test config isa OptimizerConfig
            @test hasfield(typeof(config), :optimizer)
            @test hasfield(typeof(config), :name)
            @test config.name isa String
            @test length(config.name) > 0
        end
    end
    
    @testset "Unique Configurations" begin
        # Different learning rates should create different optimizers
        config1 = adam_config(learning_rate=0.01)
        config2 = adam_config(learning_rate=0.001)
        
        @test config1.optimizer.eta != config2.optimizer.eta
        @test config1.name == config2.name  # Same name, different config
        
        # Different optimizer types should be different
        adam_cfg = adam_config()
        sgd_cfg = sgd_config()
        
        @test typeof(adam_cfg.optimizer) != typeof(sgd_cfg.optimizer)
        @test adam_cfg.name != sgd_cfg.name
    end
end