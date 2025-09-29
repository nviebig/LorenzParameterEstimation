using Test
using LorenzParameterEstimation

@testset "LorenzParameterEstimation.jl" begin
    
    @testset "Parameter Types" begin
        # Test parameter construction
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        @test params.σ == 10.0
        @test params.ρ == 28.0
        @test params.β ≈ 8.0/3.0
        
        # Test keyword constructor
        params2 = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
        @test params2.σ == params.σ && params2.ρ == params.ρ && params2.β == params.β
        
        # Test arithmetic
        diff = params - params2
        @test norm(diff) ≈ 0 atol=1e-15
        
        scaled = 2.0 * params
        @test scaled.σ == 20.0 && scaled.ρ == 56.0
    end
    
    @testset "System and Solution" begin
        params = classic_params()
        u0 = [1.0, 1.0, 1.0]
        tspan = (0.0, 1.0)
        dt = 0.01
        
        system = L63System(params=params, u0=u0, tspan=tspan, dt=dt)
        @test system.params == params
        @test system.u0 == u0
        @test system.tspan == tspan
        @test system.dt == dt
        
        # Test integration
        solution = integrate(system)
        @test length(solution) > 90  # Should have ~100 points
        @test size(solution.u, 2) == 3  # 3D trajectory
        @test solution.final_state ≈ solution.u[end, :] 
    end
    
    @testset "Integration" begin
        params = classic_params()
        u0 = [1.0, 1.0, 1.0]
        
        # Short integration test
        sol = integrate(params, u0, (0.0, 0.1), 0.01)
        @test length(sol) == 11  # 0.1/0.01 + 1
        @test sol.u[1, :] ≈ u0
        
        # Test that solution evolves
        @test norm(sol.u[end, :] - u0) > 0.1
    end
    
    @testset "Loss and Gradients" begin
        # Generate test data
        true_params = classic_params()
        system = L63System(params=true_params, u0=[1.0, 1.0, 1.0], 
                          tspan=(0.0, 2.0), dt=0.01)
        target_sol = integrate(system)
        
        # Test self-consistency (loss with true parameters should be ~0)
        loss = compute_loss(true_params, target_sol, 1, 50)
        @test loss < 1e-12
        
        # Test gradients with perturbed parameters
        perturbed_params = L63Parameters(10.0, 28.5, 8.0/3.0)  # Slight ρ error
        loss_val, grads = compute_gradients(perturbed_params, target_sol, 1, 50)
        @test loss_val > 1e-6  # Should have non-trivial loss
        @test norm(grads) > 1e-6  # Should have non-trivial gradients
    end
    
    @testset "Training" begin
        # Generate synthetic training data
        true_params = L63Parameters(10.0, 25.0, 8.0/3.0)  # Slightly modified
        system = L63System(params=true_params, u0=[1.0, 1.0, 1.0],
                          tspan=(0.0, 5.0), dt=0.01)
        target_sol = integrate(system)
        
        # Initial guess with error
        initial_guess = L63Parameters(10.0, 20.0, 8.0/3.0)  # ρ error
        
        # Training config (short for testing)
        config = L63TrainingConfig(
            epochs=10, 
            η=1e-2, 
            window_size=100,
            update_σ=false,  # Only estimate ρ  
            update_ρ=true,
            update_β=false,
            verbose=false
        )
        
        # Train
        best_params, loss_hist, param_hist = estimate_parameters(target_sol, initial_guess, config=config)

        # Check convergence
        @test length(loss_hist) == 10
        @test loss_hist[end] < loss_hist[1]  # Loss should decrease
        @test abs(best_params.ρ - true_params.ρ) < abs(initial_guess.ρ - true_params.ρ)  # Better estimate
        @test length(param_hist) == length(loss_hist) + 1
        @test first(param_hist).ρ == initial_guess.ρ
        @test abs(last(param_hist).ρ - true_params.ρ) < abs(first(param_hist).ρ - true_params.ρ)
    end
    
    @testset "Utilities" begin
        params = classic_params()
        @test params.σ == 10.0 && params.ρ == 28.0
        
        stable = stable_params()  
        @test stable.ρ == 15.0
        
        # Test parameter error calculation
        true_params = classic_params()
        estimated = L63Parameters(10.0, 29.0, 8.0/3.0)  # 1 unit error in ρ
        errors = parameter_error(true_params, estimated)
        @test errors.ρ ≈ 1.0/28.0  # Relative error
    end
    
end
