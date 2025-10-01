# ========================================
# Performance Benchmarks
# ========================================

using BenchmarkTools

@testset "Performance Benchmarks" begin
    println("   Running performance benchmarks...")
    println("   (This may take a few minutes...)")
    
    @testset "Core Function Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        
        # Benchmark lorenz_rhs
        @testset "lorenz_rhs Performance" begin
            println("     Benchmarking lorenz_rhs...")
            bench_rhs = @benchmark lorenz_rhs($u0, $params)
            
            @test median(bench_rhs).time < 1000  # Should be very fast (< 1μs)
            println("       lorenz_rhs median time: $(median(bench_rhs).time) ns")
        end
        
        # Benchmark rk4_step
        @testset "rk4_step Performance" begin
            println("     Benchmarking rk4_step...")
            dt = 0.01
            bench_step = @benchmark rk4_step($u0, $params, $dt)
            
            @test median(bench_step).time < 10000  # Should be fast (< 10μs)
            println("       rk4_step median time: $(median(bench_step).time) ns")
        end
        
        # Benchmark parameter arithmetic
        @testset "Parameter Arithmetic Performance" begin
            println("     Benchmarking parameter arithmetic...")
            params2 = L63Parameters(1.0, 2.0, 0.5)
            
            bench_add = @benchmark $params + $params2
            bench_mul = @benchmark 2.0 * $params
            
            @test median(bench_add).time < 1000   # Very fast
            @test median(bench_mul).time < 1000   # Very fast
            
            println("       Parameter addition median time: $(median(bench_add).time) ns")
            println("       Parameter multiplication median time: $(median(bench_mul).time) ns")
        end
    end
    
    @testset "Integration Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        
        # Different integration lengths
        test_cases = [
            ("Short (0.1s)", (0.0, 0.1), 0.01),
            ("Medium (1.0s)", (0.0, 1.0), 0.01), 
            ("Long (5.0s)", (0.0, 5.0), 0.01),
        ]
        
        for (name, tspan, dt) in test_cases
            @testset "$name Integration" begin
                println("     Benchmarking $name integration...")
                
                bench_integrate = @benchmark integrate($params, $u0, $tspan, $dt)
                
                # Performance expectations (adjust based on system)
                expected_times = Dict(
                    "Short (0.1s)" => 100_000,   # 100μs
                    "Medium (1.0s)" => 1_000_000, # 1ms
                    "Long (5.0s)" => 5_000_000   # 5ms
                )
                
                @test median(bench_integrate).time < expected_times[name]
                println("       $name integration median time: $(median(bench_integrate).time) ns")
            end
        end
        
        # Benchmark different step sizes
        @testset "Step Size Scaling" begin
            println("     Benchmarking step size effects...")
            tspan = (0.0, 1.0)
            
            step_sizes = [0.1, 0.01, 0.001]
            times = Float64[]
            
            for dt in step_sizes
                bench = @benchmark integrate($params, $u0, $tspan, $dt)
                push!(times, median(bench).time)
                println("       dt=$dt median time: $(median(bench).time) ns")
            end
            
            # Smaller step sizes should take longer (more steps)
            @test times[end] > times[1]  # 0.001 should take longer than 0.1
        end
    end
    
    @testset "Loss Function Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(params, [1.0, 1.0, 1.0], (0.0, 1.0), 0.01)
        test_params = L63Parameters(9.8, 27.5, 2.7)
        
        # Benchmark different loss functions
        loss_functions = [
            ("RMSE", window_rmse),
            ("MAE", window_mae),
            ("MSE", window_mse),
            ("Adaptive", adaptive_loss)
        ]
        
        for (name, loss_fn) in loss_functions
            @testset "$name Loss Performance" begin
                println("     Benchmarking $name loss function...")
                
                bench_loss = @benchmark $loss_fn($test_params, $solution, 1, 50)
                
                @test median(bench_loss).time < 10_000_000  # Should be < 10ms
                println("       $name loss median time: $(median(bench_loss).time) ns")
            end
        end
        
        # Benchmark window size effects
        @testset "Window Size Scaling" begin
            println("     Benchmarking window size effects...")
            
            window_sizes = [10, 50, 100, 200]
            times = Float64[]
            
            for window_size in window_sizes
                bench = @benchmark window_rmse($test_params, $solution, 1, $window_size)
                push!(times, median(bench).time)
                println("       Window size $window_size median time: $(median(bench).time) ns")
            end
            
            # Larger windows should generally take longer
            @test times[end] > times[1]
        end
    end
    
    @testset "Gradient Computation Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(params, [1.0, 1.0, 1.1], (0.0, 0.5), 0.01)
        test_params = L63Parameters(9.5, 27.0, 2.8)
        
        @testset "Basic Gradient Computation" begin
            println("     Benchmarking gradient computation...")
            
            bench_grad = @benchmark compute_gradients_modular($test_params, $solution, 1, 30, window_rmse)
            
            @test median(bench_grad).time < 100_000_000  # Should be < 100ms
            println("       Gradient computation median time: $(median(bench_grad).time) ns")
        end
        
        # Benchmark different loss functions for gradients
        @testset "Gradient Performance by Loss Function" begin
            println("     Benchmarking gradients for different loss functions...")
            
            loss_functions = [window_rmse, window_mae, window_mse, adaptive_loss]
            
            for loss_fn in loss_functions
                bench = @benchmark compute_gradients_modular($test_params, $solution, 1, 25, $loss_fn)
                println("       $(loss_fn) gradient median time: $(median(bench).time) ns")
                
                # All should be reasonably fast
                @test median(bench).time < 200_000_000  # < 200ms
            end
        end
        
        # Benchmark window size effects on gradients
        @testset "Gradient Window Size Scaling" begin
            println("     Benchmarking gradient computation window size scaling...")
            
            window_sizes = [10, 25, 50, 100]
            times = Float64[]
            
            for window_size in window_sizes
                bench = @benchmark compute_gradients_modular($test_params, $solution, 1, $window_size, window_rmse)
                push!(times, median(bench).time)
                println("       Gradient window size $window_size median time: $(median(bench).time) ns")
            end
            
            # Larger windows should take longer for gradients
            @test times[end] > times[1]
        end
    end
    
    @testset "Training Performance Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        solution = integrate(params, [1.0, 1.0, 1.1], (0.0, 2.0), 0.01)
        initial_guess = L63Parameters(9.0, 26.0, 2.5)
        
        @testset "Modern Training Performance" begin
            println("     Benchmarking modern training API...")
            
            bench_train = @benchmark modular_train!(
                deepcopy($initial_guess),
                $solution;
                optimizer_config = adam_config(),
                loss_function = window_rmse,
                epochs = 10,
                window_size = 50,
                verbose = false
            )
            
            # Training should complete in reasonable time
            @test median(bench_train).time < 30_000_000_000  # < 30 seconds
            println("       10-epoch training median time: $(median(bench_train).time / 1e9) seconds")
        end
        
        @testset "Legacy Training Performance" begin
            println("     Benchmarking legacy training API...")
            
            config = L63TrainingConfig(
                epochs = 10,
                window_size = 50,
                verbose = false
            )
            
            bench_train_legacy = @benchmark train!(deepcopy($initial_guess), $solution, $config)
            
            # Should also be reasonably fast
            @test median(bench_train_legacy).time < 30_000_000_000  # < 30 seconds
            println("       Legacy 10-epoch training median time: $(median(bench_train_legacy).time / 1e9) seconds")
        end
        
        # Compare different optimizers
        @testset "Optimizer Performance Comparison" begin
            println("     Benchmarking different optimizers...")
            
            optimizers = [
                ("Adam", adam_config()),
                ("SGD", sgd_config(learning_rate=0.01)),
                ("AdamW", adamw_config()),
                ("RMSprop", rmsprop_config())
            ]
            
            for (name, opt_config) in optimizers
                bench = @benchmark modular_train!(
                    deepcopy($initial_guess),
                    $solution;
                    optimizer_config = $opt_config,
                    loss_function = window_rmse,
                    epochs = 5,
                    window_size = 40,
                    verbose = false
                )
                
                println("       $name optimizer 5-epoch training median time: $(median(bench).time / 1e9) seconds")
                
                # All optimizers should complete in reasonable time
                @test median(bench).time < 20_000_000_000  # < 20 seconds
            end
        end
    end
    
    @testset "Memory Allocation Benchmarks" begin
        params = L63Parameters(10.0, 28.0, 8.0/3.0)
        u0 = [1.0, 1.0, 1.0]
        solution = integrate(params, u0, (0.0, 0.5), 0.01)
        
        @testset "Low-Level Function Allocations" begin
            println("     Benchmarking memory allocations for core functions...")
            
            # RHS function should allocate minimally
            bench_rhs = @benchmark lorenz_rhs($u0, $params)
            @test median(bench_rhs).memory < 1000  # Should allocate very little
            println("       lorenz_rhs median allocation: $(median(bench_rhs).memory) bytes")
            
            # RK4 step will allocate for intermediate vectors
            bench_step = @benchmark rk4_step($u0, $params, 0.01)
            @test median(bench_step).memory < 5000  # Should be reasonable
            println("       rk4_step median allocation: $(median(bench_step).memory) bytes")
        end
        
        @testset "Loss Function Allocations" begin
            println("     Benchmarking loss function allocations...")
            
            test_params = L63Parameters(9.8, 27.5, 2.7)
            
            # First call to compile
            window_rmse(test_params, solution, 1, 30)
            
            bench_loss = @benchmark window_rmse($test_params, $solution, 1, 30)
            @test median(bench_loss).memory < 50000  # Should be reasonable
            println("       window_rmse median allocation: $(median(bench_loss).memory) bytes")
        end
        
        @testset "Gradient Computation Allocations" begin
            println("     Benchmarking gradient computation allocations...")
            
            test_params = L63Parameters(9.5, 27.0, 2.8)
            
            # First call to compile
            compute_gradients_modular(test_params, solution, 1, 20, window_rmse)
            
            bench_grad = @benchmark compute_gradients_modular($test_params, $solution, 1, 20, window_rmse)
            
            # Gradient computation will allocate more due to AD
            @test median(bench_grad).memory < 1_000_000  # < 1MB should be reasonable
            println("       Gradient computation median allocation: $(median(bench_grad).memory) bytes")
        end
    end
    
    println("   ✅ Performance benchmarks completed")
    println("   📊 All benchmark results logged above")
end

# Performance summary function
function benchmark_summary()
    println("\n" * "="^60)
    println("PERFORMANCE BENCHMARK SUMMARY")
    println("="^60)
    println("Run the test suite to see detailed timing information.")
    println("Key performance expectations:")
    println("  • Core functions (lorenz_rhs, rk4_step): < 10μs")
    println("  • Integration (1s trajectory): < 1ms")
    println("  • Loss computation (50-point window): < 10ms")
    println("  • Gradient computation (30-point window): < 100ms")
    println("  • Training (10 epochs): < 30s")
    println("  • Memory allocations: Minimal for core functions")
    println("="^60)
end

# Call summary when this file is included
benchmark_summary()