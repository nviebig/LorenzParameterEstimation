module LorenzParameterEstimation

# Core types
export L63Parameters, L63System, L63Solution, L63TrainingConfig

# Core functionality  
export integrate, compute_loss, compute_gradients, train!, estimate_parameters

# Utility functions
export lorenz_rhs, classic_params, stable_params, parameter_error

# Visualization functions (loaded via extensions when Plots.jl is available)
export plot_trajectory, plot_phase_portrait, animate_comparison, create_training_gif

using LinearAlgebra
using Statistics
using Printf

# Include all submodules
include("types.jl")
include("integration.jl") 
include("loss.jl")
include("training.jl")
include("utils.jl")
include("visualization.jl")

end # module