# LorenzParameterEstimation Examples

This directory contains examples demonstrating different aspects of the Lorenz parameter estimation package.

## Directory Structure

### 📁 `basic_training/`
Basic parameter estimation examples:
- `l63_training.jl` - Simple training script
- `l63_training.ipynb` - Jupyter notebook tutorial
- `lorenz_training_evolution.gif` - Animation of training progress

### 📁 `api_examples/`
API usage demonstrations:
- `clean_training_api.jl` - TensorFlow-like API examples showing different optimizers and loss functions

### 📁 `tests/`
Test scripts for development:
- `test_fixed_api.jl` - Tests for the fixed training APIs
- `test_optimizers.jl` - Verification that different optimizers work correctly

### 📁 `noise_robustness/`
Examples for handling noisy data:
- Coming soon: noise injection, robust loss functions, uncertainty quantification

## Quick Start

```julia
# Basic parameter estimation
include("basic_training/l63_training.jl")

# Try different optimizers and loss functions
include("api_examples/clean_training_api.jl")

# Test noise robustness
include("noise_robustness/noisy_data_example.jl")
```

## Key Features Demonstrated

- ✅ **Multiple Optimizers**: Adam, AdamW, SGD, RMSprop
- ✅ **Multiple Loss Functions**: RMSE, MAE, MSE, Adaptive loss
- ✅ **Clean API**: TensorFlow-like configuration
- 🔄 **Noise Handling**: Robust estimation with noisy observations
- 📊 **Visualization**: Training progress and parameter evolution