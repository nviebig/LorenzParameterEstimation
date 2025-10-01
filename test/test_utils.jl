# ========================================
# Utility Functions Tests
# ========================================

@testset "Parameter Utilities" begin
    @testset "classic_params" begin
        params = classic_params()
        @test params isa L63Parameters
        @test params.σ == 10.0
        @test params.ρ == 28.0
        @test params.β ≈ 8.0/3.0
        
        # Test with different type
        params_f32 = classic_params(Float32)
        @test params_f32 isa L63Parameters{Float32}
        @test typeof(params_f32.σ) == Float32
    end
    
    @testset "stable_params" begin
        params = stable_params()
        @test params isa L63Parameters
        @test params.σ == 10.0
        @test params.ρ == 15.0  # Below critical value for chaos
        @test params.β ≈ 8.0/3.0
        
        # Should be in non-chaotic regime
        @test params.ρ < 24.74  # Approximate onset of chaos
    end
    
    @testset "parameter_error" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        estimated_params = L63Parameters(9.8, 27.5, 2.7)
        
        error = parameter_error(estimated_params, true_params)
        @test error ≥ 0
        @test isfinite(error)
        
        # Perfect match should give zero error
        zero_error = parameter_error(true_params, true_params)
        @test zero_error ≈ 0 atol=1e-15
        
        # Larger differences should give larger errors
        far_params = L63Parameters(5.0, 15.0, 1.0)
        large_error = parameter_error(far_params, true_params)
        @test large_error > error
    end
    
    @testset "parameter_error Components" begin
        true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Test individual parameter errors
        σ_diff = L63Parameters(11.0, 28.0, 8.0/3.0)
        ρ_diff = L63Parameters(10.0, 29.0, 8.0/3.0)
        β_diff = L63Parameters(10.0, 28.0, 3.0)
        
        σ_error = parameter_error(σ_diff, true_params)
        ρ_error = parameter_error(ρ_diff, true_params)
        β_error = parameter_error(β_diff, true_params)
        
        @test σ_error > 0
        @test ρ_error > 0
        @test β_error > 0
        
        # Combined error should be larger
        combined_diff = L63Parameters(11.0, 29.0, 3.0)
        combined_error = parameter_error(combined_diff, true_params)
        @test combined_error > σ_error
        @test combined_error > ρ_error
        @test combined_error > β_error
    end
end

@testset "Array Backend Utilities" begin
    @testset "similar_array" begin
        # Test with Vector input
        u0 = [1.0, 2.0, 3.0]
        
        # Create similar array with same dimensions
        similar_vec = similar_array(u0, Float64, 3)
        @test typeof(similar_vec) == typeof(u0)
        @test length(similar_vec) == 3
        @test eltype(similar_vec) == Float64
        
        # Create similar array with different dimensions
        similar_mat = similar_array(u0, Float64, 5, 3)
        @test size(similar_mat) == (5, 3)
        @test eltype(similar_mat) == Float64
        
        # Test type consistency
        u0_f32 = Float32[1.0, 2.0, 3.0]
        similar_f32 = similar_array(u0_f32, Float32, 4)
        @test eltype(similar_f32) == Float32
        @test length(similar_f32) == 4
    end
    
    @testset "Type Preservation" begin
        # The similar_array function should preserve the backend type
        # while allowing element type changes
        
        u0 = [1.0, 2.0, 3.0]  # Regular Array
        
        # Same element type
        sim1 = similar_array(u0, Float64, 3)
        @test typeof(sim1) <: Array{Float64}
        
        # Different element type
        sim2 = similar_array(u0, Float32, 3)
        @test typeof(sim2) <: Array{Float32}
        
        # Matrix creation
        sim3 = similar_array(u0, Float64, 2, 3)
        @test typeof(sim3) <: Array{Float64, 2}
        @test size(sim3) == (2, 3)
    end
end

@testset "Random Number Generation" begin
    @testset "Seed Consistency" begin
        # Test that setting seed gives reproducible results
        Random.seed!(12345)
        params1 = L63Parameters(rand(), rand(), rand())
        
        Random.seed!(12345)
        params2 = L63Parameters(rand(), rand(), rand())
        
        @test params1.σ == params2.σ
        @test params1.ρ == params2.ρ
        @test params1.β == params2.β
    end
    
    @testset "Random Parameter Generation" begin
        Random.seed!(42)
        
        # Generate random parameters in reasonable ranges
        σ_range = (5.0, 15.0)
        ρ_range = (15.0, 35.0)
        β_range = (1.0, 4.0)
        
        σ = σ_range[1] + (σ_range[2] - σ_range[1]) * rand()
        ρ = ρ_range[1] + (ρ_range[2] - ρ_range[1]) * rand()
        β = β_range[1] + (β_range[2] - β_range[1]) * rand()
        
        params = L63Parameters(σ, ρ, β)
        
        @test σ_range[1] ≤ params.σ ≤ σ_range[2]
        @test ρ_range[1] ≤ params.ρ ≤ ρ_range[2]
        @test β_range[1] ≤ params.β ≤ β_range[2]
    end
end

@testset "Mathematical Utilities" begin
    @testset "Norm Calculations" begin
        # Test different norms with parameter vectors
        params1 = L63Parameters(3.0, 4.0, 0.0)
        params2 = L63Parameters(0.0, 0.0, 0.0)
        
        diff = params1 - params2
        vec = [diff.σ, diff.ρ, diff.β]
        
        # L2 norm
        l2_norm = sqrt(sum(vec.^2))
        @test l2_norm ≈ 5.0  # 3-4-5 triangle
        
        # L1 norm
        l1_norm = sum(abs.(vec))
        @test l1_norm == 7.0
        
        # Infinity norm
        linf_norm = maximum(abs.(vec))
        @test linf_norm == 4.0
    end
    
    @testset "Statistical Functions" begin
        # Generate test data
        data = randn(100, 3)
        
        # Mean
        data_mean = mean(data, dims=1)
        @test size(data_mean) == (1, 3)
        
        # Standard deviation
        data_std = std(data, dims=1)
        @test size(data_std) == (1, 3)
        @test all(data_std .> 0)
        
        # Test with trajectory data
        params = classic_params()
        solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 2.0), 0.01)
        
        traj_mean = mean(solution.trajectory, dims=1)
        traj_std = std(solution.trajectory, dims=1)
        
        @test size(traj_mean) == (1, 3)
        @test size(traj_std) == (1, 3)
        @test all(traj_std .> 0)  # Should have variation in chaotic system
    end
end

@testset "Validation and Error Checking" begin
    @testset "Parameter Validation" begin
        # Valid parameters
        @test_nowarn L63Parameters(10.0, 28.0, 8.0/3.0)
        @test_nowarn L63Parameters(1e-6, 1e-6, 1e-6)
        @test_nowarn L63Parameters(1000.0, 1000.0, 1000.0)
        
        # Edge cases that should work
        @test_nowarn L63Parameters(0.0, 0.0, 0.0)
        @test_nowarn L63Parameters(-1.0, -1.0, -1.0)
    end
    
    @testset "Integration Parameter Validation" begin
        params = classic_params()
        u0 = [1.0, 1.0, 1.0]
        
        # Valid time spans
        @test_nowarn integrate(params, u0, (0.0, 1.0), 0.01)
        @test_nowarn integrate(params, u0, (-1.0, 1.0), 0.01)
        @test_nowarn integrate(params, u0, (1.0, 2.0), 0.01)
        
        # Very small time step
        @test_nowarn integrate(params, u0, (0.0, 0.1), 1e-6)
        
        # Larger time step (might be less accurate but should work)
        @test_nowarn integrate(params, u0, (0.0, 1.0), 0.1)
    end
    
    @testset "Numerical Stability" begin
        # Test with extreme parameters that might cause numerical issues
        extreme_cases = [
            L63Parameters(1e-10, 1e-10, 1e-10),  # Very small
            L63Parameters(1e10, 1e10, 1e10),     # Very large
            L63Parameters(1e-10, 1e10, 1.0),     # Mixed scales
        ]
        
        for params in extreme_cases
            # Basic arithmetic should work
            @test_nowarn params + params
            @test_nowarn params - params
            @test_nowarn 2.0 * params
            @test_nowarn params / 2.0
            
            # Parameter error calculation should be finite
            error = parameter_error(params, classic_params())
            @test isfinite(error)
            @test error ≥ 0
        end
    end
end

@testset "Type System Integration" begin
    @testset "Generic Programming" begin
        # Test that utilities work with different numeric types
        types_to_test = [Float32, Float64]
        
        for T in types_to_test
            params = classic_params(T)
            @test typeof(params.σ) == T
            @test typeof(params.ρ) == T
            @test typeof(params.β) == T
            
            # Arithmetic should preserve type
            doubled = 2 * params
            @test typeof(doubled.σ) == T
            
            # Error calculation should work
            error = parameter_error(params, stable_params(T))
            @test typeof(error) == T
            @test isfinite(error)
        end
    end
    
    @testset "Array Type Compatibility" begin
        # Test that similar_array works with different array types
        base_arrays = [
            [1.0, 2.0, 3.0],           # Vector{Float64}
            Float32[1.0, 2.0, 3.0],    # Vector{Float32}
        ]
        
        for base_array in base_arrays
            # Should create compatible arrays
            sim_vec = similar_array(base_array, eltype(base_array), 5)
            @test length(sim_vec) == 5
            @test eltype(sim_vec) == eltype(base_array)
            
            sim_mat = similar_array(base_array, eltype(base_array), 3, 4)
            @test size(sim_mat) == (3, 4)
            @test eltype(sim_mat) == eltype(base_array)
        end
    end
end