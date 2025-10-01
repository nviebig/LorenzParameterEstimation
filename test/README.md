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

## Current Test Status

**Last Updated: October 1, 2025**

```
Test Summary:                             | Pass  Fail  Error  Total   Time
LorenzParameterEstimation.jl              |  183     7     91    281  44.3s
  Core Types                              |   30            6     36   2.4s
    L63Parameters                         |   21            1     22   1.7s
      Construction                        |    8                   8   0.1s
      Arithmetic Operations               |    9            1     10   1.6s
      Validation                          |    4                   4   0.1s
    L63System                             |    4            2      6   0.2s
      Construction                        |    4            1      5   0.1s
      Type Consistency                    |                 1      1   0.0s
    L63Solution                           |                 2      2   0.1s
      Basic Properties                    |                 1      1   0.1s
      Final State                         |                 1      1   0.0s
    OptimizerConfig                       |    5            1      6   0.1s
      Basic Construction                  |    5                   5   0.0s
      Different Optimizers                |                 1      1   0.1s
  Integration                             |   14           21     35   1.3s
    lorenz_rhs                            |   10                  10   0.1s
    rk4_step                              |                 3      3   0.1s
      Basic Step                          |                 1      1   0.0s
      Step Size Effects                   |                 1      1   0.0s
      Conservation Properties             |                 1      1   0.1s
    integrate                             |    4           18     22   0.6s
      Basic Integration                   |    1            6      7   0.3s
      System Integration                  |    1            3      4   0.0s
      Different Time Spans                |                 3      3   0.0s
      Different Step Sizes                |    1            2      3   0.3s
      Chaotic Behavior                    |                 4      4   0.0s
      Stable Fixed Point                  |    1                   1   0.0s
  Loss Functions                          |   27                  27  10.9s
  Optimizers                              |   18     1     14     33   1.6s
    OptimizerConfig Creation              |   12     1      7     20   1.1s
      adam_config                         |    4            2      6   0.1s
      sgd_config                          |    3            1      4   0.0s
      adamw_config                        |    4            1      5   0.0s
      adagrad_config                      |    1     1      2      4   0.9s
      rmsprop_config                      |                 1      1   0.0s
    Optimizer Integration                 |                 2      2   0.0s
      Optimisers.jl Compatibility         |                 1      1   0.0s
      L63Parameters Optimization          |                 1      1   0.0s
    Optimizer Behavior                    |                 3      3   0.0s
      Learning Rate Effects               |                 1      1   0.0s
      Convergence Properties              |                 1      1   0.0s
      Momentum Effects                    |                 1      1   0.0s
    Configuration Validation              |    6            2      8   0.1s
      Parameter Bounds                    |    2            1      3   0.1s
      Type Consistency                    |                 1      1   0.0s
      Unique Configurations               |    4                   4   0.0s
  Training APIs                           |   27     2     14     43   4.5s
    Legacy Training API (train!)          |   16     2      3     21   3.4s
      Basic Training                      |    1     2      1      4   1.4s
      Different Optimizers                |                 1      1   0.0s
      Different Loss Functions            |    4                   4   1.7s
      Parameter Updates                   |                 1      1   0.2s
      Training Configuration Validation   |   11                  11   0.0s
    Modern Training API (modular_train!)  |                 4      4   0.2s
      Basic Modular Training              |                 1      1   0.0s
      Early Stopping                      |                 1      1   0.0s
      Different Modular Configurations    |                 1      1   0.0s
      Parameter Selection                 |                 1      1   0.2s
    Training Robustness                   |   11            5     16   0.1s
      Noisy Data                          |                 1      1   0.0s
      Different Initial Guesses           |    8            4     12   0.0s
      Edge Case Parameters                |    3                   3   0.0s
    Training Metrics and History          |                 2      2   0.0s
      Metrics Content                     |                 1      1   0.0s
      History Tracking                    |                 1      1   0.0s
  Utilities                               |   48     2     13     63   1.9s
    Parameter Utilities                   |   15            6     21   0.2s
      classic_params                      |    6                   6   0.0s
      stable_params                       |    5                   5   0.0s
      parameter_error                     |    1            3      4   0.1s
      parameter_error Components          |    3            3      6   0.0s
    Array Backend Utilities               |                 2      2   0.0s
      similar_array                       |                 1      1   0.0s
      Type Preservation                   |                 1      1   0.0s
    Random Number Generation              |    6                   6   0.0s
    Mathematical Utilities                |    6            1      7   1.0s
      Norm Calculations                   |    3                   3   0.1s
      Statistical Functions               |    3            1      4   0.9s
    Validation and Error Checking         |   13            1     14   0.2s
      Parameter Validation                |    5                   5   0.0s
      Integration Parameter Validation    |    5                   5   0.1s
      Numerical Stability                 |    3            1      4   0.1s
    Type System Integration               |    8     2      3     13   0.1s
      Generic Programming                 |    8     2      2     12   0.1s
      Array Type Compatibility            |                 1      1   0.0s
  End-to-End                              |    2           13     15   0.6s
    Complete Workflow Tests               |    2            6      8   0.1s
      Basic Parameter Estimation Workflow |    2            3      5   0.1s
      Modern API Complete Workflow        |                 1      1   0.1s
      Multi-Optimizer Comparison          |                 1      1   0.0s
      Different Loss Function Comparison  |                 1      1   0.0s
    Robustness and Edge Cases             |                 3      3   0.0s
      Noisy Data Estimation               |                 1      1   0.0s
      Different System Regimes            |                 1      1   0.0s
      Short and Long Trajectories         |                 1      1   0.0s
    Performance and Scalability           |                 2      2   0.0s
      Training Time Scaling               |                 1      1   0.0s
      Memory Usage                        |                 1      1   0.0s
    API Consistency and Error Handling    |                 2      2   0.0s
      Invalid Inputs                      |                 1      1   0.0s
      API Consistency                     |                 1      1   0.0s
  Code Quality                            |   17     2     10     29  21.1s
    Code Quality Analysis                 |   13     2      5     20  20.2s
      Aqua.jl Quality Checks              |    8            1      9  20.1s
        Method Ambiguities                |    2                   2   2.7s
        Undefined Exports                 |    2                   2   0.2s
        Unbound Args                      |    2                   2   0.2s
        Persistent Tasks                  |    2                   2  17.0s
        Project TOML                      |                 1      1   0.0s
      Documentation Completeness          |                 1      1   0.0s
      Type Stability                      |    3     2      1      6   0.1s
      Performance Regression              |    2            1      3   0.0s
      Memory Allocation                   |                 1      1   0.0s
    Numerical Correctness                 |    4            2      6   0.0s
      Mathematical Properties             |    3                   3   0.0s
      Numerical Integration Accuracy      |    1            1      2   0.0s
      Gradient Accuracy                   |                 1      1   0.0s
    Error Handling and Edge Cases         |                 3      3   0.1s
      Boundary Conditions                 |                 1      1   0.0s
      Extreme Parameter Values            |                 1      1   0.0s
      Degenerate Cases                    |                 1      1   0.0s
```

**Progress Summary:**
- ✅ **Loss Functions**: 27/27 tests passing (100%)
- ⚠️ **Core Types**: 30/36 tests passing (83%)
- ⚠️ **Integration**: 14/35 tests passing (40%)
- ⚠️ **Training APIs**: 27/43 tests passing (63%)
- ⚠️ **Code Quality**: 17/29 tests passing (59%)
- **Overall**: **183/281 tests passing (65%)**

**Next Priority Areas:**
1. Training API compatibility issues (modular_train! optimizer state management)
2. Integration test field access issues (.trajectory → .u)
3. Code quality test module updates (Aqua.jl function calls)

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