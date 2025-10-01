# Test Configuration and Development Guide

## Running Tests

### Complete Test Suite
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run tests with coverage
julia --project=. --code-coverage=user -e "using Pkg; Pkg.test()"
```

### Individual Test Modules
```bash
# Run specific test files
julia --project=. test/test_types.jl
julia --project=. test/test_integration.jl
julia --project=. test/test_loss.jl
julia --project=. test/test_optimizers.jl
julia --project=. test/test_training.jl
julia --project=. test/test_utils.jl
julia --project=. test/test_integration_e2e.jl
julia --project=. test/test_code_quality.jl
```

### Performance Benchmarks
```bash
# Run performance benchmarks
julia --project=. test/benchmarks.jl

# Or include in test suite
julia --project=. -e "using Test; include(\"test/benchmarks.jl\")"
```

## Test Structure

### Unit Tests
- **`test_types.jl`**: Core data structures (L63Parameters, L63System, L63Solution, OptimizerConfig)
- **`test_integration.jl`**: Numerical integration (RK4, ODE solving, trajectory generation)
- **`test_loss.jl`**: Loss functions and gradient computation
- **`test_optimizers.jl`**: Optimizer configurations and Optimisers.jl integration
- **`test_training.jl`**: Both legacy and modern training APIs
- **`test_utils.jl`**: Utility functions and helper methods

### Integration Tests
- **`test_integration_e2e.jl`**: End-to-end workflows, multi-component testing, robustness

### Quality Assurance
- **`test_code_quality.jl`**: Aqua.jl analysis, type stability, performance regression, numerical correctness

### Performance
- **`benchmarks.jl`**: Comprehensive performance benchmarks for all major functions

## Continuous Integration

### GitHub Actions Workflows
- **Main CI**: Tests on Julia 1.9, 1.10, nightly across Linux/macOS/Windows
- **Documentation**: Automatic doc building (when docs/ exists)
- **Benchmarks**: Performance monitoring on main/develop branches
- **Code Quality**: Aqua.jl checks and formatting validation
- **Performance Regression**: Compare PR performance vs main branch

### Local Development
```bash
# Check code quality
julia --project=. -e "using Aqua, LorenzParameterEstimation; Aqua.test_all(LorenzParameterEstimation)"

# Format code (install JuliaFormatter.jl first)
julia --project=. -e "using JuliaFormatter; format(\".\")"

# Profile performance
julia --project=. -e "using Profile; @profile include(\"test/benchmarks.jl\")"
```

## Test Categories and Coverage

### Core Functionality (100% Coverage Goal)
- ✅ Parameter arithmetic and type safety
- ✅ Numerical integration accuracy and stability  
- ✅ Loss function correctness and gradient validation
- ✅ Optimizer configuration and Optimisers.jl compatibility
- ✅ Training API completeness (both legacy and modern)

### Robustness Testing
- ✅ Edge cases (extreme parameters, short/long trajectories)
- ✅ Numerical stability (near-singular cases, floating-point limits)
- ✅ Error handling (invalid inputs, boundary conditions)
- ✅ Noise robustness (varying SNR levels)

### Performance Validation
- ✅ Function-level benchmarks (μs-level timing)
- ✅ Algorithm scaling (linear/quadratic behavior validation)
- ✅ Memory allocation tracking
- ✅ Regression detection

### End-to-End Scenarios
- ✅ Complete parameter estimation workflows
- ✅ Multi-optimizer comparison studies
- ✅ Different loss function effectiveness
- ✅ Various dynamical regimes (chaotic, periodic, stable)

## Development Workflow

### Before Committing
1. Run full test suite: `julia --project=. -e "using Pkg; Pkg.test()"`
2. Check performance: `julia --project=. test/benchmarks.jl`
3. Validate quality: `julia --project=. -e "using Aqua, LorenzParameterEstimation; Aqua.test_all(LorenzParameterEstimation)"`
4. Format code: `julia --project=. -e "using JuliaFormatter; format(\".\")`

### Adding New Features
1. Write tests first (TDD approach)
2. Implement feature
3. Ensure all tests pass
4. Add performance benchmarks if needed
5. Update documentation

### Performance Expectations
- **Core functions**: < 10μs (lorenz_rhs, rk4_step)
- **Integration**: < 1ms per simulated second
- **Loss computation**: < 10ms for 50-point windows
- **Gradient computation**: < 100ms for 30-point windows  
- **Training**: < 30s for 10 epochs on typical problems

## Test Data and Fixtures

### Standard Test Parameters
```julia
# Classic chaotic regime
classic_params = L63Parameters(10.0, 28.0, 8.0/3.0)

# Stable fixed-point regime  
stable_params = L63Parameters(10.0, 0.5, 8.0/3.0)

# Standard initial condition
u0 = [1.0, 1.0, 1.0]

# Typical integration settings
tspan = (0.0, 2.0)
dt = 0.01
```

### Reproducibility
- All tests use `Random.seed!(12345)` for deterministic behavior
- Benchmark results may vary by system but should be consistent within runs
- Gradient validation uses finite differences with ε=1e-6

## Troubleshooting

### Common Test Failures
1. **Gradient validation fails**: Check Enzyme.jl version, ensure functions are module-level
2. **Performance regression**: Check for allocations, profile bottlenecks
3. **Integration accuracy**: Verify step size, check for numerical instability
4. **CI failures**: Check Julia version compatibility, dependency versions

### Debug Tools
```bash
# Run tests with debugging
julia --project=. -e "using Pkg; Pkg.test()" --check-bounds=yes

# Profile memory allocations
julia --project=. --track-allocation=user test/runtests.jl

# Interactive debugging
julia --project=. -i test/test_types.jl
```

This comprehensive test suite ensures code quality, performance, and correctness across all usage scenarios.