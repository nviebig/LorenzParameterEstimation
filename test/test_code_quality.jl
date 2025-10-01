# ========================================
# Code Quality and Standards Tests
# ========================================

using Aqua

@testset "Code Quality Analysis" begin
    @testset "Aqua.jl Quality Checks" begin
        println("   Running Aqua.jl code quality analysis...")
        
        # Test for ambiguities
        @testset "Method Ambiguities" begin
            @test_nowarn Aqua.test_ambiguities(LorenzParameterEstimation)
        end
        
        # Test for undefined exports
        @testset "Undefined Exports" begin
            @test_nowarn Aqua.test_undefined_exports(LorenzParameterEstimation)
        end
        
        # Test for unbound args
        @testset "Unbound Args" begin
            @test_nowarn Aqua.test_unbound_args(LorenzParameterEstimation)
        end
        
        # Test for persistent tasks
        @testset "Persistent Tasks" begin
            @test_nowarn Aqua.test_persistent_tasks(LorenzParameterEstimation)
        end
        
        # Test project structure
        @testset "Project TOML" begin
            @test_nowarn Aqua.test_project_toml_formatting(LorenzParameterEstimation)
        end
        
        println("   ✅ Aqua.jl analysis completed")
    end
    
    @testset "Documentation Completeness" begin
        println("   Checking documentation completeness...")
        
        # Check that exported functions have docstrings
        exported_functions = [
            L63Parameters, L63System, L63Solution, L63TrainingConfig,
            integrate, compute_loss, compute_gradients, compute_gradients_modular,
            window_rmse, train!, estimate_parameters, modular_train!,
            window_mae, window_mse, weighted_window_loss, probabilistic_loss, adaptive_loss,
            OptimizerConfig, adam_config, sgd_config, adamw_config, adagrad_config, rmsprop_config,
            lorenz_rhs, classic_params, stable_params, parameter_error
        ]
        
        # Test that key functions have some documentation
        # (Note: This is a basic check - more sophisticated doc testing would require additional tools)
        for func in [L63Parameters, L63System, integrate, train!, window_rmse]
            docs = @doc func
            @test docs isa Base.Docs.DocStr || docs isa Markdown.MD
        end
        
        println("   ✅ Documentation check completed")
    end
    
    @testset "Type Stability" begin
        println("   Testing type stability...")
        
        # Test key functions for type stability
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        
        # lorenz_rhs should be type stable
        @test @inferred(lorenz_rhs(u0, params)) isa Vector{Float64}
        
        # rk4_step should be type stable
        @test @inferred(rk4_step(u0, params, 0.01)) isa Vector{Float64}
        
        # Parameter arithmetic should be type stable
        @test @inferred(params + params) isa L63Parameters{Float64}
        @test @inferred(params - params) isa L63Parameters{Float64}
        @test @inferred(2.0 * params) isa L63Parameters{Float64}
        
        # parameter_error should be type stable
        @test @inferred(parameter_error(params, params)) isa Float64
        
        println("   ✅ Type stability check completed")
    end
    
    @testset "Performance Regression" begin
        println("   Basic performance regression tests...")
        
        # Test that basic operations are reasonably fast
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        
        # Integration should be fast enough
        @test @elapsed(integrate(params, u0, (0.0, 1.0), 0.01)) < 1.0  # Should take less than 1 second
        
        # Loss computation should be reasonably fast
        solution = integrate(params, u0, (0.0, 0.5), 0.01)
        @test @elapsed(window_rmse(params, solution, 1, 20)) < 0.1  # Should be very fast
        
        # Gradient computation should be reasonable
        @test @elapsed(compute_gradients_modular(params, solution, 1, 15, window_rmse)) < 1.0
        
        println("   ✅ Performance regression check completed")
    end
    
    @testset "Memory Allocation" begin
        println("   Testing memory allocation patterns...")
        
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        solution = integrate(params, u0, (0.0, 0.3), 0.01)
        
        # Test that repeated loss computations don't allocate excessively
        # (Run once to compile, then measure)
        window_rmse(params, solution, 1, 15)
        
        allocs_before = @allocated window_rmse(params, solution, 1, 15)
        @test allocs_before < 10000  # Should not allocate too much
        
        # Gradient computation will allocate more, but should be reasonable
        compute_gradients_modular(params, solution, 1, 10, window_rmse)
        grad_allocs = @allocated compute_gradients_modular(params, solution, 1, 10, window_rmse)
        @test grad_allocs < 100000  # Allow more for AD but still bounded
        
        println("   ✅ Memory allocation check completed")
    end
end

@testset "Numerical Correctness" begin
    @testset "Mathematical Properties" begin
        println("   Testing mathematical properties...")
        
        # Test Lorenz system properties
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Fixed points should have zero derivative
        if params.ρ > 1
            C = sqrt(params.β * (params.ρ - 1))
            fp1 = [C, C, params.ρ - 1]
            fp2 = [-C, -C, params.ρ - 1]
            
            @test norm(lorenz_rhs(fp1, params)) < 1e-12
            @test norm(lorenz_rhs(fp2, params)) < 1e-12
        end
        
        # Origin is fixed point when ρ < 1
        stable_params = L63Parameters(10.0, 0.5, 8.0/3.0)
        @test norm(lorenz_rhs([0.0, 0.0, 0.0], stable_params)) < 1e-14
        
        println("   ✅ Mathematical properties verified")
    end
    
    @testset "Numerical Integration Accuracy" begin
        println("   Testing integration accuracy...")
        
        # Test against known analytical properties
        params = L63Parameters(10.0, 0.5, 8.0/3.0)  # Stable case
        u0 = [0.1, 0.1, 0.1]
        
        # Should converge to origin for stable case
        solution = integrate(params, u0, (0.0, 10.0), 0.01)
        @test norm(solution.final_state) < 0.01
        
        # Test energy evolution (not conserved in Lorenz, but should evolve smoothly)
        params_chaotic = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution_chaotic = integrate(params_chaotic, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        
        energies = [sum(solution_chaotic.trajectory[i, :].^2) for i in 1:size(solution_chaotic.trajectory, 1)]
        energy_diffs = diff(energies)
        
        # Should not have huge jumps (numerical stability)
        @test maximum(abs.(energy_diffs)) < 1000
        
        println("   ✅ Integration accuracy verified")
    end
    
    @testset "Gradient Accuracy" begin
        println("   Testing gradient computation accuracy...")
        
        # Compare analytical gradients with finite differences (more comprehensive than in loss tests)
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 0.4), 0.01)
        test_params = L63Parameters(9.8, 27.5, 2.7)
        
        # Test multiple loss functions
        loss_functions = [window_rmse, window_mae, window_mse]
        
        for loss_fn in loss_functions
            _, analytical_grad = compute_gradients_modular(test_params, solution, 1, 20, loss_fn)
            
            # Finite difference gradients
            h = 1e-6
            finite_diff_grad = zeros(3)
            
            # σ gradient
            params_plus = L63Parameters(test_params.σ + h, test_params.ρ, test_params.β)
            params_minus = L63Parameters(test_params.σ - h, test_params.ρ, test_params.β)
            finite_diff_grad[1] = (loss_fn(params_plus, solution, 1, 20) - loss_fn(params_minus, solution, 1, 20)) / (2h)
            
            # ρ gradient
            params_plus = L63Parameters(test_params.σ, test_params.ρ + h, test_params.β)
            params_minus = L63Parameters(test_params.σ, test_params.ρ - h, test_params.β)
            finite_diff_grad[2] = (loss_fn(params_plus, solution, 1, 20) - loss_fn(params_minus, solution, 1, 20)) / (2h)
            
            # β gradient
            params_plus = L63Parameters(test_params.σ, test_params.ρ, test_params.β + h)
            params_minus = L63Parameters(test_params.σ, test_params.ρ, test_params.β - h)
            finite_diff_grad[3] = (loss_fn(params_plus, solution, 1, 20) - loss_fn(params_minus, solution, 1, 20)) / (2h)
            
            # Compare (allowing for some numerical error)
            rel_error = abs.(analytical_grad - finite_diff_grad) ./ (abs.(finite_diff_grad) .+ 1e-8)
            @test all(rel_error .< 1e-2)  # 1% relative error tolerance
        end
        
        println("   ✅ Gradient accuracy verified")
    end
end

@testset "Error Handling and Edge Cases" begin
    @testset "Boundary Conditions" begin
        println("   Testing boundary conditions...")
        
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        
        # Window at very beginning
        @test_nowarn window_rmse(params, solution, 1, 5)
        
        # Window near end (should handle boundaries gracefully)
        n_points = length(solution.times)
        @test_nowarn window_rmse(params, solution, n_points - 5, 5)
        
        # Single point window
        @test_nowarn window_rmse(params, solution, 1, 1)
        
        println("   ✅ Boundary conditions handled correctly")
    end
    
    @testset "Extreme Parameter Values" begin
        println("   Testing extreme parameter values...")
        
        solution = integrate(L63Parameters(10.0, 28.0, 8.0/3.0), [1.0, 1.0, 1.0], (0.0, 0.5), 0.01)
        
        extreme_cases = [
            L63Parameters(1e-10, 1e-10, 1e-10),  # Very small
            L63Parameters(1e6, 1e6, 1e6),        # Very large
            L63Parameters(0.0, 0.0, 0.0),        # Zero
            L63Parameters(-10.0, -28.0, -8.0/3.0), # All negative
        ]
        
        for extreme_params in extreme_cases
            # Should not crash or produce NaN/Inf
            loss = window_rmse(extreme_params, solution, 1, 10)
            @test isfinite(loss)
            @test loss ≥ 0
            
            # Gradients should also be finite
            _, grads = compute_gradients_modular(extreme_params, solution, 1, 10, window_rmse)
            @test all(isfinite.(grads))
        end
        
        println("   ✅ Extreme parameter values handled correctly")
    end
    
    @testset "Degenerate Cases" begin
        println("   Testing degenerate cases...")
        
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Very short trajectory
        short_solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 0.01), 0.001)
        @test_nowarn window_rmse(params, short_solution, 1, 1)
        
        # Constant trajectory (degenerate case)
        times = collect(0.0:0.01:1.0)
        constant_traj = ones(length(times), 3)  # All ones
        system = L63System(params)
        constant_solution = L63Solution(times, constant_traj, system)
        
        @test_nowarn window_rmse(params, constant_solution, 1, 10)
        
        println("   ✅ Degenerate cases handled correctly")
    end
end