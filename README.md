# LorenzParameterEstimation.jl




A Julia package for parameter estimation in the Lorenz-63 chaotic dynamical system using automatic differentiation with Enzyme.jl. This package provides efficient tools for fitting Lorenz system parameters to observational data through gradient-based optimization and windowed training approaches.

## Features

🔬 **Parameter Estimation**: Recover Lorenz-63 parameters (σ, ρ, β) from trajectory data
⚡ **Enzyme Integration**: Fast automatic differentiation using Enzyme.jl for gradient computation
🎯 **Windowed Training**: Teacher-forcing approach with short windows for stable training in chaotic systems
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

### Basic Usage

```julia
using LorenzParameterEstimation

# Create a Lorenz system with known parameters
true_params = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
system = L63System(
    params = true_params,
    u0 = [1.0, 1.0, 1.0],
    tspan = (0.0, 10.0),
    dt = 0.01
)

# Generate reference trajectory
solution = integrate(system)

# Estimate parameters from noisy data
initial_guess = L63Parameters(σ=10.0, ρ=20.0, β=8.0/3.0)  # Wrong ρ value
config = L63TrainingConfig(
    epochs = 100,
    η = 5e-3,
    window_size = 300,
    update_ρ = true,   # Only optimize ρ
    update_σ = false,  # Keep σ fixed
    update_β = false   # Keep β fixed
)

best_params, loss_history = train!(initial_guess, solution, config)
```

### Enzyme-based Training (Advanced)

For maximum performance with automatic differentiation:

```julia
using LorenzParameterEstimation
using Enzyme

# Generate target data
true_params = ModelParameters(10.0, 28.0, 8.0/3.0)
x0 = [1.0, 1.0, 1.0]
M = 5000
dt = 0.01

_, target_trajectory = integrate_model(x0, true_params, M, dt)

# Set up training
initial_guess = ModelParameters(10.0, 15.0, 8.0/3.0)  # Wrong ρ
config = TrainConfig(
    epochs = 150,
    η = 5e-3,
    Mwin = 400,
    dt = dt,
    clip = 5.0,
    update_σ = false,
    update_ρ = true,
    update_β = false
)

# Train with Enzyme gradients
best_params, loss_hist = train_enzyme!(initial_guess, target_trajectory, config)
```

## Core Types

### Parameters

- `L63Parameters{T}`: Immutable parameter container
- `ModelParameters`: Mutable parameter container for Enzyme training

### System Specification

- `L63System{T}`: Complete system specification with parameters, initial conditions, and integration settings
- `L63Solution{T}`: Integration results with trajectory data and metadata

### Training Configuration

- `L63TrainingConfig{T}`: Standard training configuration
- `TrainConfig`: Enzyme-optimized training configuration

## Key Functions

### Integration

```julia
# Integrate a complete system
solution = integrate(system::L63System)

# Quick integration
solution = integrate(params, u0, tspan, dt)

# Enzyme-compatible integration
final_state, trajectory = integrate_model(u0, params, steps, dt)
```

### Parameter Estimation

```julia
# Standard training
best_params, history = train!(params, target_data, config)

# Enzyme-based training (fastest)
best_params, history = train_enzyme!(params, target_data, config)

# Compute loss between trajectories
loss = compute_loss(params, target_solution, window_start, window_length)
```

### Visualization

```julia
using Plots

# Plot 3D trajectory
plot_trajectory(solution)

# Phase portrait comparison
plot_phase_portrait([sol1, sol2], labels=["True", "Fitted"])

# Create training animation
animate_comparison(true_traj, fitted_traj, true_params, fitted_params)
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

- `l63_enzyme_training.jl`: Complete Enzyme-based parameter estimation workflow
- `l63_training.ipynb`: Interactive Jupyter notebook with visualizations
- `lorenz_training_evolution.gif`: Example training animation

## Mathematical Background
The Lorenz-63 system is governed by the following set of ordinary differential equations:

```math
\begin{aligned}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
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

The package is optimized for performance:

- Enzyme.jl provides fastest automatic differentiation
- `@inline` functions reduce call overhead
- Pre-allocated arrays minimize memory allocation
- Type-stable implementations throughout

Typical performance on a modern CPU:

- Parameter estimation: ~1-5 seconds for 100 epochs
- Gradient computation: ~10x faster than finite differences
- Integration: ~1000 steps/second for dense output

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

