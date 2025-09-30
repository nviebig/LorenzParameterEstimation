# LorenzParameterEstimation.jl

[![Build Status](https://github.com/nviebig/LorenzParameterEstimation.jl/workflows/CI/badge.svg)](https://github.com/nviebig/LorenzParameterEstimation.jl/actions)

A Julia package for parameter estimation in the Lorenz-63 chaotic dynamical system using automatic differentiation with Enzyme.jl. This package provides efficient tools for fitting Lorenz system parameters to observational data through gradient-based optimization and windowed training approaches.

## Features

🔬 **Parameter Estimation**: Recover Lorenz-63 parameters (σ, ρ, β) from trajectory data
⚡ **Enzyme-Only Gradients**: Fast automatic differentiation using Enzyme.jl exclusively for gradient computation
🎯 **Windowed Training**: Teacher-forcing approach with short windows for stable training in chaotic systems
🚀 **Modern Optimizers**: Full Optimisers.jl integration with Adam, SGD, AdaGrad, RMSProp, and custom chains
⏰ **Early Stopping**: Automatic convergence detection with configurable patience
📊 **Comprehensive Visualization**: Built-in plotting and animation capabilities
🧪 **Robust Integration**: 4th-order Runge-Kutta integration optimized for AD compatibility
📈 **Training Diagnostics**: Loss tracking, gradient monitoring, and convergence analysis

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nviebig/LorenzParameterEstimation.jl")
```

Or in development mode:

```julia
Pkg.develop(path="path/to/LorenzParameterEstimation")
```

## Quick Start

### Basic Usage with Traditional Interface

```julia
using LorenzParameterEstimation

# Create target data from known parameters
true_params = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
target_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.01)

# Estimate parameters from trajectory data
initial_guess = L63Parameters(σ=10.0, ρ=20.0, β=8.0/3.0)  # Wrong ρ value
config = L63TrainingConfig(
    epochs = 100,
    η = 5e-3,
    window_size = 300,
    update_ρ = true,   # Only optimize ρ
    update_σ = false,  # Keep σ fixed
    update_β = false   # Keep β fixed
)

best_params, loss_history, param_history = train!(initial_guess, target_solution, config)
```

### Modern Interface with Advanced Optimizers
```

```

### Modern Interface with Advanced Optimizers

For maximum flexibility with state-of-the-art optimizers and early stopping:

```julia

```julia
using LorenzParameterEstimation

# Generate target data
true_params = L63Parameters(10.0, 28.0, 8.0/3.0)
target_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.01)
initial_guess = L63Parameters(8.0, 25.0, 2.0)

# Train with Adam optimizer (automatic early stopping)
result = modular_train!(
    initial_guess, 
    target_solution,
    optimizer_config = adam_config(learning_rate=0.01),
    epochs = 200,
    window_size = 100,
    update_σ = false,  # Only train ρ
    update_ρ = true,
    update_β = false,
    early_stopping_patience = 20,
    verbose = true
)

print("Best parameters: ", result.best_params)
print("Best parameters: ", result.best_params)
```

### Available Optimizers

The package provides convenient configurations for popular optimizers:

```julia
# Built-in optimizer configurations
adam_result = modular_train!(params, target, optimizer_config = adam_config(learning_rate=0.01))
sgd_result = modular_train!(params, target, optimizer_config = sgd_config(learning_rate=0.005))
adagrad_result = modular_train!(params, target, optimizer_config = adagrad_config(learning_rate=0.1))
rmsprop_result = modular_train!(params, target, optimizer_config = rmsprop_config(learning_rate=0.001))

# Custom optimizer chains with gradient clipping
using Optimisers
custom_optimizer = Optimisers.OptimiserChain(
    Optimisers.ClipNorm(1.0),
    Optimisers.Adam(0.01, (0.9, 0.99))
)
custom_config = OptimizerConfig(custom_optimizer, 0.01, name="Adam+ClipNorm")
custom_result = modular_train!(params, target, optimizer_config = custom_config)
```

### Early Stopping

Automatic convergence detection prevents overfitting:

```julia
result = modular_train!(
    initial_guess, target_solution,
    optimizer_config = adam_config(),
    epochs = 500,  # Maximum epochs
    early_stopping_patience = 20,     # Stop after 20 epochs without improvement
    early_stopping_min_delta = 1e-6,  # Minimum improvement threshold
    verbose = true
)
# Training may stop before 500 epochs if convergence is detected
```

## Core Types

### Parameters

- `L63Parameters{T}`: Immutable parameter container for Lorenz-63 coefficients (σ, ρ, β)

### System Specification

- `L63System{T}`: Complete system specification with parameters, initial conditions, and integration settings
- `L63Solution{T}`: Integration results with trajectory data and metadata

### Training Configuration

- `L63TrainingConfig{T}`: Traditional training configuration for `train!` function
- `OptimizerConfig`: Modern optimizer configuration for `modular_train!` with Optimisers.jl support

## Key Functions

### Integration

```julia
# Integrate a complete system
solution = integrate(system::L63System)

# Quick integration from parameters
solution = integrate(params, u0, tspan, dt)
```

### Parameter Estimation

```julia
# Traditional training interface
best_params, loss_history, param_history = train!(params, target_solution, config)

# Modern training with optimizers and early stopping
result = modular_train!(
    params, target_solution;
    optimizer_config = adam_config(),
    epochs = 100,
    update_σ = false, update_ρ = true, update_β = false,
    early_stopping_patience = 20
)

# Loss computation with Enzyme gradients
loss_value, gradients = compute_gradients(params, target_solution, window_start, window_size)
```

### Optimizer Configurations

```julia
# Pre-configured optimizers
adam_config(learning_rate=0.001)
sgd_config(learning_rate=0.01)
adagrad_config(learning_rate=0.1) 
rmsprop_config(learning_rate=0.001)

# Custom optimizer setup
OptimizerConfig(optimizer, learning_rate, name="Custom")
```

### Visualization

The package includes comprehensive visualization tools:

```julia
using Plots, StatsPlots

# Compare parameter estimation results
plot_parameter_comparison(initial_params, fitted_params, true_params)

# Visualize trajectory differences  
plot_trajectory_comparison(initial_sol, fitted_sol, true_sol)

# Time series analysis
plot_time_series([true_sol, fitted_sol], labels=["True", "Fitted"])

# 3D phase portraits
plot_phase_portrait(solution, title="Lorenz Attractor")
```

The training notebook `examples/l63_training.ipynb` demonstrates comprehensive visualizations including:

- Parameter comparison bar charts across multiple optimizers
- 3D trajectory overlays showing initial vs. final vs. true systems  
- Time series comparisons highlighting parameter estimation quality
- Performance summary tables with error metrics

```julia
```julia
# Create training animation
animate_comparison(true_traj, fitted_traj, true_params, fitted_params)
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

- `l63_training.ipynb`: Interactive Jupyter notebook with multiple optimizer comparisons and visualizations
- `l63_lux.ipynb`: Jupyter notebook demonstrating the core Enzyme-based training workflow
- Comprehensive parameter estimation workflows with both traditional and modern interfaces

### Quick Example: Optimizer Comparison

The notebook `examples/l63_training.ipynb` demonstrates training with multiple optimizers:

```julia
# Compare different optimizers on the same problem
results = []
for (name, config) in [("Adam", adam_config(0.01)), 
                       ("SGD", sgd_config(0.005)),
                       ("AdaGrad", adagrad_config(0.1))]
    result = modular_train!(initial_guess, target_solution, 
                           optimizer_config=config, epochs=100,
                           update_σ=false, update_ρ=true, update_β=false)
    push!(results, (name, result))
end
```

## Mathematical Background

The Lorenz-63 system is defined by:

```math
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz
```

Where:

- σ: Prandtl number (buoyancy frequency)
- ρ: Rayleigh number (density parameter)  
- β: Geometric parameter (thermal expansion coefficient)

### Training Methodology

1. **Teacher Forcing**: Start each training window from observed data
2. **Windowed Loss**: Use short integration windows to maintain gradient stability
3. **RMSE Objective**: Root mean square error between predicted and observed trajectories
4. **Gradient Clipping**: L2 norm clipping to prevent gradient explosion
5. **Selective Updates**: Choose which parameters to optimize

## Performance

The package is optimized for performance with Enzyme.jl providing state-of-the-art automatic differentiation:

- **Enzyme-only gradients**: Fastest available AD for Julia, no overhead from multiple AD backends
- **Optimisers.jl integration**: Modern optimization algorithms with efficient state management
- **Early stopping**: Prevents unnecessary computation when convergence is reached
- **Type-stable implementations**: Optimized for performance throughout
- **Memory efficient**: Array wrapping strategy for parameter updates minimizes allocations

Typical performance on a modern CPU:

- Parameter estimation: ~1-5 seconds for 100 epochs with early stopping
- Gradient computation: Native Enzyme performance, significantly faster than finite differences
- Integration: ~1000 steps/second for dense output
- Optimizer overhead: Minimal due to efficient Optimisers.jl integration

## Contributing

Contributions are welcome! Please see the contributing guidelines and feel free to submit issues and pull requests.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{lorenz_parameter_estimation,
  author = {Niklas Viebig},
  title = {LorenzParameterEstimation.jl: Parameter Estimation for Lorenz-63 Systems},
  year = {2025},
  url = {https://github.com/nviebig/LorenzParameterEstimation.jl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Related Work

- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl): High-performance automatic differentiation
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl): Comprehensive ODE solving
- [Optimization.jl](https://github.com/SciML/Optimization.jl): General optimization interface

---

For more detailed documentation, see the `docs/` directory or visit the [online documentation](https://nviebig.github.io/LorenzParameterEstimation.jl/).

