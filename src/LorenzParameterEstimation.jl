module LorenzParameterEstimation

# Core types
export L63Parameters, L63System, L63Solution, L63TrainingConfig

# Core functionality  
export integrate, compute_loss, compute_gradients, window_rmse, train!, estimate_parameters

# Modern modular training
export modular_train!

# Loss functions (using Flux.jl ecosystem)
export window_mae, window_mse, weighted_window_loss, probabilistic_loss, adaptive_loss
export mse, mae, huber_loss  # Re-exported from Flux

# Optimizer configurations
export OptimizerConfig, build_optimizer
export sgd_config, adam_config, adamw_config, rmsprop_config, adagrad_config, lion_config
export robust_optimizer, fast_optimizer, conservative_optimizer
export SchedulerConfig, exponential_decay_schedule, cosine_annealing_schedule, step_decay_schedule

# Bayesian inference - requires Turing.jl
# export bayesian_parameter_estimation, variational_inference, posterior_predictive_check
# export lorenz_bayesian_model

# Utility functions
export lorenz_rhs, classic_params, stable_params, parameter_error

# Visualization functions (loaded via extensions when Plots.jl is available)
export plot_trajectory, plot_phase_portrait, animate_comparison, create_training_gif

using LinearAlgebra
using Statistics
using Printf
using Random

# Include all submodules
include("types.jl")
include("integration.jl") 
include("loss.jl")
include("optimizers.jl")
include("training.jl")
# include("bayesian.jl")  # Commented out - requires Turing.jl
include("utils.jl")
include("visualization.jl")

end # module
