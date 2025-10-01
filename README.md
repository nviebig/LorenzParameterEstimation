# LorenzParameterEstimation.jl

[![CI](https://github.com/nviebig/LorenzParameterEstimation.jl/workflows/CI/badge.svg)](https://github.com/nviebig/LorenzParameterEstimation.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nviebig/LorenzParameterEstimation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nviebig/LorenzParameterEstimation.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/julia-v1.9+-blue.svg)](https://julialang.org)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for **parameter estimation in the Lorenz-63 chaotic dynamical system** using modern automatic differentiation and optimization techniques. The main goal is to recover unknown parameters (σ, ρ, β) from observed trajectory data, which is challenging due to the chaotic nature of the system.

## 📖 Package Overview

### **What This Package Does**
This package solves the inverse problem of estimating parameters in the Lorenz-63 system from observed trajectory data. Unlike forward simulation, parameter estimation requires:

1. **Differentiating through chaotic dynamics** - Computing gradients through thousands of integration steps
2. **Handling trajectory divergence** - Even small parameter errors cause exponential divergence
3. **Robust optimization** - Advanced optimizers that handle noisy gradients and multiple local minima
4. **Validation and diagnostics** - Comprehensive tools to assess estimation quality

### **Core Architecture**

The package is built around several key design principles:

1. **Modular Design**: Separate concerns (integration, loss computation, optimization, visualization)
2. **Type Safety**: Structured types prevent common errors and support different precision/backends
3. **Performance First**: Enzyme.jl for state-of-the-art automatic differentiation
4. **Flexible APIs**: Both legacy and modern interfaces for different use cases

## ✨ Features

- 🔬 **Parameter Estimation**: Recover Lorenz-63 parameters (σ, ρ, β) from trajectory data
- ⚡ **Enzyme-Only Gradients**: Fast automatic differentiation using Enzyme.jl exclusively  
- 🎯 **Windowed Training**: Teacher-forcing approach for stable training in chaotic systems
- 🚀 **Modern Optimizers**: Full Optimisers.jl integration (Adam, SGD, AdaGrad, RMSProp, custom chains)
- ⏰ **Early Stopping**: Automatic convergence detection with configurable patience
- 📊 **Comprehensive Loss Functions**: RMSE, MAE, MSE, adaptive/Huber, weighted losses
- 🧪 **Robust Integration**: 4th-order Runge-Kutta integration optimized for AD compatibility
- 📈 **Training Diagnostics**: Loss tracking, gradient monitoring, and convergence analysis
- 🎨 **Rich Visualization**: Built-in plotting and animation capabilities via extensions
- 🔧 **Noise Robustness**: Tools for handling noisy observations and robust estimation

## 🏗️ Architecture Deep Dive

### **File Structure and Purpose**

#### **Core Computational Engine**
- **`src/types.jl`**: Fundamental data structures with type safety
  - `L63Parameters{T}`: Parameters (σ, ρ, β) with arithmetic operations
  - `L63System{T}`: Complete system specification  
  - `L63Solution{T}`: Integration results with metadata
  - `L63TrainingConfig{T}`: Legacy training configuration

- **`src/integration.jl`**: High-performance ODE integration
  - `lorenz_rhs()`: Right-hand side of Lorenz-63 equations
  - `rk4_step()`: Enzyme-optimized Runge-Kutta step
  - `integrate()`: Full trajectory integration with dense output

- **`src/loss.jl`**: Loss functions and gradient computation
  - **Windowed loss approach**: Solves chaotic trajectory divergence problem
  - **Multiple loss functions**: RMSE, MAE, MSE, adaptive (Huber), weighted
  - **Enzyme.jl gradients**: Module-level functions for AD compatibility
  - `compute_gradients_modular()`: Main gradient computation engine

#### **Modern Optimization Stack**
- **`src/optimizers.jl`**: Clean API for advanced optimizers
  - Pre-configured optimizers: `adam_config()`, `sgd_config()`, `rmsprop_config()`
  - Custom optimizer chains with gradient clipping
  - Learning rate scheduling support
  - Quick presets: `robust_optimizer()`, `fast_optimizer()`

- **`src/training.jl`**: Dual training interfaces
  - **`modular_train!()`**: Modern TensorFlow-like API with early stopping
  - **`train!()`**: Legacy interface for backward compatibility
  - Mini-batching, validation splits, comprehensive metrics tracking

#### **Supporting Infrastructure**  
- **`src/utils.jl`**: Parameter utilities and helper functions
  - Standard parameter sets: `classic_params()`, `stable_params()`
  - Noise generation and validation splits
  - Array backend utilities for GPU compatibility

- **`src/visualization.jl`**: Plotting interface (extension-based)
  - Functions defined but implemented only when Plots.jl is loaded
  - Keeps core package lightweight while providing rich visualization

### **Key Design Decisions & Why They Matter**

#### **1. Windowed Loss - Solving the Chaos Problem**
```julia
# The core insight: Why windowed training works
function compute_loss(params, target_solution, window_start, window_length)
    # Teacher forcing: start from observed state
    u = target_solution[window_start]
    
    # Short horizon integration (1-2 Lyapunov times)
    for i in 1:window_length
        u = rk4_step(u, params, dt)
        # Compare to target immediately
    end
end
```

**Problem**: Lorenz-63 is chaotic - small parameter errors cause exponential trajectory divergence. Long-horizon loss gives meaningless gradients dominated by chaotic drift.

**Solution**: 
- **Teacher forcing**: Start each window from observed state
- **Short horizons**: Only integrate for ~1-2 Lyapunov times  
- **Multiple windows**: Train on many short segments instead of one long trajectory

#### **2. Enzyme.jl for Gradients - Performance First**
**Why not ForwardDiff/Zygote?** Parameter estimation requires differentiating through thousands of integration steps. Enzyme.jl provides:
- **Native performance**: Compiles to optimized machine code
- **Memory efficiency**: No tape overhead
- **Numerical stability**: Exact derivatives, not finite differences

**Critical requirement**: Functions must be defined at module level for Enzyme scoping.

#### **3. Modern vs Legacy APIs**

**Legacy API (`train!` + `L63TrainingConfig`)**:
```julia
config = L63TrainingConfig(epochs=200, η=1e-1, window_size=400)
best_params, loss_hist, param_hist = train!(guess_params, solution, config)
# NO EARLY STOPPING - runs all epochs!
```

**Modern API (`modular_train!`)**:
```julia
result = modular_train!(
    params, target_solution,
    optimizer_config = adam_config(learning_rate=0.01),
    loss_function = window_rmse,
    early_stopping_patience = 25  # Stops when converged
)
```

**Key difference**: Early stopping prevents overfitting and saves computation.

#### **4. Type System Benefits**
- **`L63Parameters{T}`**: Prevents dimension errors, supports different precisions
- **Generic implementations**: Work with Float32/Float64, CPU/GPU arrays
- **Arithmetic operations**: Enable gradient updates with `params - α * gradients`

#### **5. Extension-Based Visualization**
- **Core package**: Lightweight, no plotting dependencies
- **When Plots.jl loaded**: Rich visualization automatically available
- **Benefits**: Fast loading, optional visualization, clean dependencies

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nviebig/LorenzParameterEstimation.jl")
```

Or in development mode:

```julia
Pkg.develop(path="path/to/LorenzParameterEstimation")
```

## 🚀 Quick Start

### Basic Usage with Modern API (Recommended)

```julia
using LorenzParameterEstimation

# Generate target data from known parameters
true_params = L63Parameters(σ=10.0, ρ=28.0, β=8.0/3.0)
target_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.01)

# Estimate parameters from trajectory data
initial_guess = L63Parameters(σ=10.0, ρ=20.0, β=8.0/3.0)  # Wrong ρ value

# Train with automatic early stopping
result = modular_train!(
    initial_guess, 
    target_solution,
    optimizer_config = adam_config(learning_rate=0.01),
    epochs = 200,
    window_size = 100,
    update_σ = false,  # Only optimize ρ  
    update_ρ = true,
    update_β = false,
    early_stopping_patience = 25,
    verbose = true
)

println("Best parameters: ", result.best_params)
```

### Traditional Interface (Legacy)

```julia
# Traditional interface (still works, but no early stopping)
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

### Optimizer Comparison

Compare different optimizers on the same problem:

```julia
# Compare multiple optimizers
optimizers = [
    ("Adam", adam_config(learning_rate=0.01)),
    ("SGD", sgd_config(learning_rate=0.005)),
    ("AdaGrad", adagrad_config(learning_rate=0.1)),
    ("RMSprop", rmsprop_config(learning_rate=0.001))
]

results = []
for (name, config) in optimizers
    result = modular_train!(
        initial_guess, target_solution,
        optimizer_config = config,
        epochs = 100,
        update_σ = false,
        update_ρ = true, 
        update_β = false,
        early_stopping_patience = 20
    )
    push!(results, (name, result))
    println("$name: $(result.best_params)")
end
```

### Custom Loss Functions

Use different loss functions for robustness:

```julia
# Standard RMSE (default)
result_rmse = modular_train!(params, target, loss_function = window_rmse)

# Robust MAE for noisy data
result_mae = modular_train!(params, target, loss_function = window_mae)

# Adaptive/Huber loss for outlier robustness
result_adaptive = modular_train!(params, target, loss_function = adaptive_loss)

# Weighted loss emphasizing later time steps
weighted_rmse = weighted_window_loss(window_rmse, 1.5)
result_weighted = modular_train!(params, target, loss_function = weighted_rmse)

# Custom loss function
custom_loss(pred, target) = mean(abs.(pred .- target).^1.5)  # L1.5 loss
result_custom = modular_train!(params, target, loss_function = custom_loss)
```

## 📝 Core Types and Functions

### Essential Types

- **`L63Parameters{T}`**: Parameter container with arithmetic operations
- **`L63System{T}`**: Complete system specification  
- **`L63Solution{T}`**: Integration results with metadata
- **`OptimizerConfig`**: Modern optimizer configuration

### Key Functions

#### Integration
```julia
# Quick integration
solution = integrate(params, u0, tspan, dt)

# Full system integration  
system = L63System(params=params, u0=u0, tspan=tspan, dt=dt)
solution = integrate(system)
```

#### Training (Two APIs)

**Modern API (Recommended)**:
```julia
result = modular_train!(
    params, target_solution;
    optimizer_config = adam_config(),
    loss_function = window_rmse,
    epochs = 100,
    early_stopping_patience = 20
)
```

**Legacy API**:
```julia
config = L63TrainingConfig(epochs=100, η=1e-3, window_size=300)
best_params, loss_hist, param_hist = train!(params, target_solution, config)
```

#### Loss and Gradients
```julia
# Compute loss and gradients
loss_value, gradients = compute_gradients_modular(
    params, target_solution, window_start, window_size, loss_function
)
```

### Optimizer Configurations

```julia
# Pre-configured optimizers
adam_config(learning_rate=0.001, gradient_clip_norm=1.0)
sgd_config(learning_rate=0.01, momentum=0.9) 
adagrad_config(learning_rate=0.1)
rmsprop_config(learning_rate=0.001)
adamw_config(learning_rate=0.001, weight_decay=1e-2)

# Quick presets
robust_optimizer()     # AdamW with conservative settings
fast_optimizer()       # Adam with higher learning rate
conservative_optimizer() # AdamW with low learning rate

# Custom optimizer chains
using Optimisers
custom_optimizer = Optimisers.OptimiserChain(
    Optimisers.ClipNorm(1.0),
    Optimisers.Adam(0.01, (0.9, 0.99))
)
custom_config = OptimizerConfig(custom_optimizer, 0.01, name="Custom")
```

## 🎨 Visualization and Analysis

The package provides rich visualization capabilities when Plots.jl is loaded:

```julia
using Plots, StatsPlots

# Parameter comparison across optimizers
result_adam = modular_train!(params, target, optimizer_config=adam_config())
result_sgd = modular_train!(params, target, optimizer_config=sgd_config())

# Compare final parameters
param_names = ["σ", "ρ", "β"]
adam_vals = [result_adam.best_params.σ, result_adam.best_params.ρ, result_adam.best_params.β]
sgd_vals = [result_sgd.best_params.σ, result_sgd.best_params.ρ, result_sgd.best_params.β]
true_vals = [true_params.σ, true_params.ρ, true_params.β]

groupedbar([adam_vals sgd_vals true_vals], 
           labels=["Adam" "SGD" "True"],
           xticks=(1:3, param_names),
           title="Parameter Comparison")

# 3D trajectory comparison
true_sol = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.01)
fitted_sol = integrate(result_adam.best_params, [1.0, 1.0, 1.0], (0.0, 10.0), 0.01)

plot(true_sol.u[:, 1], true_sol.u[:, 2], true_sol.u[:, 3],
     label="True", linecolor=:blue, seriestype=:path3d)
plot!(fitted_sol.u[:, 1], fitted_sol.u[:, 2], fitted_sol.u[:, 3],
      label="Fitted", linecolor=:red, seriestype=:path3d)

# Loss convergence analysis
train_losses = [metric.train_loss for metric in result_adam.metrics_history]
plot(1:length(train_losses), train_losses, 
     xlabel="Epoch", ylabel="Loss", title="Training Convergence",
     yscale=:log10)
```

## 🧮 Mathematical Background

The Lorenz-63 system is defined by:

```math
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y  
dz/dt = xy - βz
```

Where:
- **σ**: Prandtl number (buoyancy frequency)
- **ρ**: Rayleigh number (density parameter)  
- **β**: Geometric parameter (thermal expansion coefficient)

### Why Parameter Estimation is Hard

1. **Chaotic dynamics**: Small parameter errors cause exponential trajectory divergence
2. **Sensitive dependence**: Parameters affect global behavior in complex ways
3. **Multiple time scales**: Different parameters operate on different time scales
4. **Local minima**: Many parameter combinations produce similar short-term behavior

### Windowed Training Solution

The key insight is **teacher forcing with short windows**:

```julia
# Instead of: integrate full trajectory and compare (unstable gradients)
predicted_full = integrate(estimated_params, u0, (0, T), dt)
loss = distance(predicted_full, observed_full)  # BAD: chaotic divergence

# Do: integrate short windows starting from observations (stable gradients)  
for window_start in 1:stride:N-window_size
    u0_window = observed[window_start]  # Teacher forcing
    predicted_window = integrate(estimated_params, u0_window, window_size*dt, dt)
    observed_window = observed[window_start:window_start+window_size]
    loss += distance(predicted_window, observed_window)  # GOOD: stable
end
```

## 🔧 Advanced Usage

### Noise Robustness

Handle noisy observations with robust loss functions:

```julia
# Generate noisy data
true_solution = integrate(true_params, [1.0, 1.0, 1.0], (0.0, 20.0), 0.01)
noisy_solution = generate_noisy_observations(true_solution, 0.1)  # 10% noise

# Use robust loss functions
result_rmse = modular_train!(params, noisy_solution, loss_function=window_rmse)
result_mae = modular_train!(params, noisy_solution, loss_function=window_mae)  
result_adaptive = modular_train!(params, noisy_solution, loss_function=adaptive_loss)

# Adaptive loss typically performs best under noise
println("RMSE noise performance: ", result_rmse.best_params)
println("MAE noise performance: ", result_mae.best_params)  
println("Adaptive noise performance: ", result_adaptive.best_params)
```

### Custom Optimizer Chains

Create sophisticated optimization strategies:

```julia
using Optimisers

# Gradient clipping + learning rate decay + Adam
custom_chain = Optimisers.OptimiserChain(
    Optimisers.ClipNorm(1.0),                    # Clip gradients
    Optimisers.OptimiserChain(                   # Scheduled learning rate
        Optimisers.Descent(0.0),                 # Placeholder for LR
        Optimisers.Adam(1.0, (0.9, 0.999))      # Adam with unit LR
    )
)

# Apply learning rate schedule manually
for epoch in 1:epochs
    lr = 0.01 * 0.95^epoch  # Exponential decay
    # Update optimizer learning rate and train
end
```

### Parameter Subset Training

Train only specific parameters:

```julia
# Only estimate ρ (common for bifurcation studies)
result_rho_only = modular_train!(
    initial_guess, target_solution,
    update_σ = false,
    update_ρ = true,  # Only ρ
    update_β = false
)

# Estimate σ and ρ, keep β fixed (physical constraint)
result_sigma_rho = modular_train!(
    initial_guess, target_solution,
    update_σ = true,
    update_ρ = true,
    update_β = false  # Keep geometric parameter fixed
)
```

### Validation and Cross-Validation

Assess estimation quality:

```julia
# Split data for validation
train_solution, val_solution = validation_split(full_solution, 0.8)

# Train on training set
result = modular_train!(params, train_solution, verbose=true)

# Evaluate on validation set
val_loss = compute_loss(result.best_params, val_solution, 1, 100)
println("Validation loss: ", val_loss)

# Parameter error analysis
errors = parameter_error(true_params, result.best_params)
println("Relative errors: σ=$(errors.σ), ρ=$(errors.ρ), β=$(errors.β)")
```

## ⚡ Performance and Optimization

### Performance Characteristics

- **Gradient computation**: ~10-100x faster than finite differences via Enzyme.jl
- **Training speed**: Typically 1-5 seconds for 100 epochs with early stopping
- **Memory efficiency**: Optimized array operations with minimal allocations
- **Scalability**: Linear scaling with trajectory length and window count

### Optimization Tips

1. **Start with robust_optimizer()** for initial experiments
2. **Use early stopping** to prevent overfitting and save computation
3. **Tune window_size** based on Lyapunov time (~100-300 for classic parameters)
4. **Consider MAE or adaptive loss** for noisy data
5. **Use mini-batching** for very long trajectories

### Typical Performance

On a modern CPU (M1/Intel):
- **100 epochs**: 1-3 seconds
- **1000 window evaluations**: <1 second  
- **Gradient computation**: ~10ms per window
- **Early stopping**: Often converges in 20-50 epochs

## 📚 Examples and Tutorials

The `examples/` directory contains comprehensive demonstrations:

### Interactive Notebooks
- **`examples/basic_training/l63_training.ipynb`**: Complete workflow with multiple optimizers
- **`examples/l63_enzyme/l63.ipynb`**: Core Enzyme-based training demonstration

### Standalone Scripts  
- **`examples/api_examples/clean_training_api.jl`**: Modern API usage patterns
- **`examples/tests/`**: Testing and validation scripts

### Key Example: Multi-Optimizer Comparison

From `examples/basic_training/l63_training.ipynb`:

```julia
# Compare optimizers systematically
optimizers = [
    ("Adam", adam_config(learning_rate=0.01)),
    ("SGD", sgd_config(learning_rate=0.005)), 
    ("AdaGrad", adagrad_config(learning_rate=0.1)),
    ("Custom", OptimizerConfig(
        Optimisers.OptimiserChain(
            Optimisers.ClipNorm(1.0),
            Optimisers.Adam(0.01, (0.9, 0.99))
        ), 0.01, name="Adam+Clip"))
]

results = []
for (name, config) in optimizers
    @time result = modular_train!(
        initial_guess, target_solution,
        optimizer_config = config,
        epochs = 200,
        early_stopping_patience = 25,
        verbose = false
    )
    push!(results, (name, result))
end

# Analyze results
for (name, result) in results
    errors = parameter_error(true_params, result.best_params)
    println("$name: Total error = $(errors.total), Epochs = $(length(result.metrics_history))")
end
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **New loss functions**: Implement additional robust loss functions
2. **Optimization strategies**: Add learning rate schedules, adaptive methods
3. **Visualization**: Extend plotting capabilities
4. **Performance**: Further optimization of core routines
5. **Documentation**: Additional examples and tutorials

Please see the contributing guidelines and feel free to submit issues and pull requests.

## 📋 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{lorenz_parameter_estimation,
  author = {Niklas Viebig},
  title = {LorenzParameterEstimation.jl: Parameter Estimation for Lorenz-63 Systems},
  year = {2025},
  url = {https://github.com/nviebig/LorenzParameterEstimation.jl}
}
```

## 🔗 Related Work

- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl): High-performance automatic differentiation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl): Modern optimization algorithms
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl): Comprehensive ODE solving
- [Optimization.jl](https://github.com/SciML/Optimization.jl): General optimization interface

---

*For detailed documentation and examples, see the `examples/` directory and the interactive notebooks.*

