# ========================================
# Integration Tests
# ========================================

@testset "lorenz_rhs" begin
    @testset "Basic Functionality" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u = [1.0, 1.0, 1.0]
        
        dudt = lorenz_rhs(u, params)
        
        @test length(dudt) == 3
        @test all(isfinite.(dudt))
        
        # Test known values
        expected_dx = params.σ * (u[2] - u[1])  # σ(y - x)
        expected_dy = u[1] * (params.ρ - u[3]) - u[2]  # x(ρ - z) - y
        expected_dz = u[1] * u[2] - params.β * u[3]  # xy - βz
        
        @test dudt[1] ≈ expected_dx
        @test dudt[2] ≈ expected_dy
        @test dudt[3] ≈ expected_dz
    end
    
    @testset "Fixed Points" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        
        # Origin should be a fixed point when ρ < 1
        params_stable = L63Parameters(10.0, 0.5, 8.0/3.0)
        dudt = lorenz_rhs([0.0, 0.0, 0.0], params_stable)
        @test all(abs.(dudt) .< 1e-14)
        
        # Non-trivial fixed points for ρ > 1
        if params.ρ > 1
            C = sqrt(params.β * (params.ρ - 1))
            fp1 = [C, C, params.ρ - 1]
            fp2 = [-C, -C, params.ρ - 1]
            
            dudt1 = lorenz_rhs(fp1, params)
            dudt2 = lorenz_rhs(fp2, params)
            
            @test norm(dudt1) < 1e-12
            @test norm(dudt2) < 1e-12
        end
    end
    
    @testset "Type Consistency" begin
        # Float32
        params_f32 = L63Parameters(10.0f0, 28.0f0, 8.0f0/3.0f0)
        u_f32 = Float32[1.0, 1.0, 1.0]
        dudt_f32 = lorenz_rhs(u_f32, params_f32)
        @test eltype(dudt_f32) == Float32
        
        # Float64
        params_f64 = L63Parameters(10.0, 28.0, 8.0/3.0)
        u_f64 = [1.0, 1.0, 1.0]
        dudt_f64 = lorenz_rhs(u_f64, params_f64)
        @test eltype(dudt_f64) == Float64
    end
end

@testset "rk4_step" begin
    @testset "Basic Step" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        dt = 0.01
        
        u1 = rk4_step(u0, params, dt)
        
        @test length(u1) == 3
        @test all(isfinite.(u1))
        @test u1 != u0  # Should evolve
        
        # For small dt, should be close to Euler step
        euler_step = u0 + dt * lorenz_rhs(u0, params)
        @test norm(u1 - euler_step) < 0.1 * norm(euler_step - u0)
    end
    
    @testset "Step Size Effects" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        
        # Multiple small steps should be more accurate than one large step
        dt_small = 0.001
        dt_large = 0.01
        
        u_small = u0
        for i in 1:10
            u_small = rk4_step(u_small, params, dt_small)
        end
        
        u_large = rk4_step(u0, params, dt_large)
        
        # Reference solution (more accurate)
        u_ref = u0
        for i in 1:100
            u_ref = rk4_step(u_ref, params, dt_large/100)
        end
        
        @test norm(u_small - u_ref) < norm(u_large - u_ref)
    end
    
    @testset "Conservation Properties" begin
        # Lorenz system conserves volume (div(F) = -σ - 1 - β < 0)
        # But we can test energy-like quantities
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        dt = 0.001
        
        u = u0
        energies = Float64[]
        for i in 1:100
            push!(energies, sum(u.^2))  # L2 norm squared
            u = rk4_step(u, params, dt)
        end
        
        # Energy should evolve smoothly (no jumps)
        energy_diffs = diff(energies)
        @test all(abs.(energy_diffs) .< 1000)  # No huge jumps
    end
end

@testset "integrate" begin
    @testset "Basic Integration" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        tspan = (0.0, 1.0)
        dt = 0.01
        
        solution = integrate(params, u0, tspan, dt)
        
        @test solution isa L63Solution
        @test length(solution.times) == length(0.0:dt:1.0)
        @test size(solution.trajectory, 1) == length(solution.times)
        @test size(solution.trajectory, 2) == 3
        @test solution.times[1] == 0.0
        @test solution.times[end] ≈ 1.0
        @test solution.trajectory[1, :] ≈ u0
    end
    
    @testset "System Integration" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        system = L63System(params=params, u0=[1.0, 1.0, 1.0], tspan=(0.0, 2.0), dt=0.01)
        
        solution = integrate(system)
        
        @test solution.system == system
        @test solution.times[1] == system.tspan[1]
        @test solution.times[end] ≈ system.tspan[2]
        @test solution.trajectory[1, :] ≈ system.u0
    end
    
    @testset "Different Time Spans" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        dt = 0.01
        
        # Short integration
        sol_short = integrate(params, u0, (0.0, 0.1), dt)
        @test length(sol_short.times) ≈ 11 atol=1  # 0.1/0.01 + 1
        
        # Longer integration
        sol_long = integrate(params, u0, (0.0, 5.0), dt)
        @test length(sol_long.times) ≈ 501 atol=1  # 5.0/0.01 + 1
        
        # Both should start from same initial condition
        @test sol_short.trajectory[1, :] ≈ sol_long.trajectory[1, :]
    end
    
    @testset "Different Step Sizes" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        tspan = (0.0, 1.0)
        
        sol_coarse = integrate(params, u0, tspan, 0.1)
        sol_fine = integrate(params, u0, tspan, 0.01)
        
        @test length(sol_coarse.times) < length(sol_fine.times)
        @test sol_coarse.trajectory[1, :] ≈ sol_fine.trajectory[1, :]  # Same initial condition
        
        # Both should reach approximately same final state (for stable integration)
        @test norm(sol_coarse.final_state - sol_fine.final_state) < 10  # Allow some difference
    end
    
    @testset "Chaotic Behavior" begin
        # Classic chaotic parameters
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        tspan = (0.0, 10.0)
        dt = 0.01
        
        solution = integrate(params, u0, tspan, dt)
        
        # Should be bounded (Lorenz attractor)
        @test all(-50 .< solution.trajectory[:, 1] .< 50)  # x component
        @test all(-50 .< solution.trajectory[:, 2] .< 50)  # y component
        @test all(0 .< solution.trajectory[:, 3] .< 50)    # z component (mostly positive)
        
        # Should be non-periodic (chaotic)
        final_third = solution.trajectory[end÷3:end, :]
        @test size(unique(round.(final_third, digits=6), dims=1), 1) > size(final_third, 1) ÷ 10
    end
    
    @testset "Stable Fixed Point" begin
        # Parameters for stable fixed point
        params = L63Parameters(10.0, 0.5, 8.0/3.0)  # ρ < 1
        u0 = [0.1, 0.1, 0.1]  # Near origin
        tspan = (0.0, 10.0)
        dt = 0.01
        
        solution = integrate(params, u0, tspan, dt)
        
        # Should converge to origin
        @test norm(solution.final_state) < 0.01
    end
end