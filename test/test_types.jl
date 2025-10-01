# ========================================
# Core Types Tests
# ========================================

@testset "L63Parameters" begin
    @testset "Construction" begin
        # Basic construction
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        @test params.σ == 10.0
        @test params.ρ == 28.0
        @test params.β ≈ 8.0/3.0
        
        # Type consistency
        @test typeof(params) == L63Parameters{Float64}
        
        # Different numeric types
        params_f32 = L63Parameters(10.0f0, 28.0f0, 8.0f0/3.0f0)
        @test typeof(params_f32) == L63Parameters{Float32}
        
        # Keyword constructor
        params_kw = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
        @test params_kw.σ == 10.0
        @test params_kw.ρ == 28.0
        @test params_kw.β ≈ 8.0/3.0
    end
    
    @testset "Arithmetic Operations" begin
        p1 = L63Parameters(10.0, 28.0, 8.0/3.0)
        p2 = L63Parameters(1.0, 2.0, 0.5)
        
        # Addition
        p_sum = p1 + p2
        @test p_sum.σ == 11.0
        @test p_sum.ρ == 30.0
        @test p_sum.β ≈ 8.0/3.0 + 0.5
        
        # Subtraction
        p_diff = p1 - p2
        @test p_diff.σ == 9.0
        @test p_diff.ρ == 26.0
        @test p_diff.β ≈ 8.0/3.0 - 0.5
        
        # Scalar multiplication
        p_scaled = 2.0 * p1
        @test p_scaled.σ == 20.0
        @test p_scaled.ρ == 56.0
        @test p_scaled.β ≈ 16.0/3.0
        
        # Division
        p_div = p1 / 2.0
        @test p_div.σ == 5.0
        @test p_div.ρ == 14.0
        @test p_div.β ≈ 4.0/3.0
    end
    
    @testset "Validation" begin
        # Valid parameters
        @test_nowarn L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Edge cases - very small values
        @test_nowarn L63Parameters(1e-6, 1e-6, 1e-6)
        
        # Large values
        @test_nowarn L63Parameters(1000.0, 1000.0, 1000.0)
        
        # Negative values (mathematically valid in some contexts)
        @test_nowarn L63Parameters(-1.0, 5.0, 1.0)
    end
end

@testset "L63System" begin
    @testset "Construction" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        tspan = (0.0, 10.0)
        dt = 0.01
        
        # Full constructor
        system = L63System(params=params, u0=u0, tspan=tspan, dt=dt)
        @test system.params == params
        @test system.u0 == u0
        @test system.tspan == tspan
        @test system.dt == dt
        
        # Convenience constructor
        system2 = L63System(params)
        @test system2.params == params
        @test length(system2.u0) == 3
        @test system2.tspan[2] > system2.tspan[1]
        @test system2.dt > 0
    end
    
    @testset "Type Consistency" begin
        params = L63Parameters(10.0f0, 28.0f0, 8.0f0/3.0f0)
        system = L63System(params)
        @test eltype(system.u0) == Float32
        @test eltype(system.tspan) == Float32
        @test typeof(system.dt) == Float32
    end
end

@testset "L63Solution" begin
    @testset "Basic Properties" begin
        # Create a simple trajectory
        times = [0.0, 0.1, 0.2, 0.3]
        trajectory = rand(4, 3)  # 4 time points, 3 components
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        system = L63System(params)
        
        solution = L63Solution(times, trajectory, system)
        
        @test solution.times == times
        @test solution.trajectory == trajectory
        @test solution.system == system
        @test size(solution.trajectory) == (4, 3)
        @test length(solution.times) == size(solution.trajectory, 1)
    end
    
    @testset "Final State" begin
        times = [0.0, 1.0, 2.0]
        trajectory = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        system = L63System(params)
        
        solution = L63Solution(times, trajectory, system)
        @test solution.final_state == [7.0, 8.0, 9.0]
        @test solution.final_time == 2.0
    end
end

@testset "OptimizerConfig" begin
    @testset "Basic Construction" begin
        # Create mock optimizer (we can't test Optimisers.jl directly without import)
        # But we can test our wrapper structure
        config = adam_config()
        @test config isa OptimizerConfig
        @test hasfield(typeof(config), :optimizer)
        @test hasfield(typeof(config), :name)
        
        config_sgd = sgd_config(learning_rate=0.1)
        @test config_sgd isa OptimizerConfig
        @test config_sgd.name == "SGD"
    end
    
    @testset "Different Optimizers" begin
        configs = [
            adam_config(),
            sgd_config(),
            adamw_config(),
            adagrad_config(),
            rmsprop_config()
        ]
        
        for config in configs
            @test config isa OptimizerConfig
            @test config.name isa String
            @test length(config.name) > 0
        end
    end
end