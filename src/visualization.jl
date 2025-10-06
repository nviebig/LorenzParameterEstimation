# Visualization functions (conditional on Plots.jl)

"""
    plot_trajectory(solution::L63Solution; components=:xyz, kwargs...)

Plot trajectory components over time.
Implemented when the `LorenzVisualizationExt` extension is loaded (i.e. when
`Plots` is available).
"""
function plot_trajectory end

"""
    plot_phase_portrait(solution::L63Solution; dims=(1,2,3), kwargs...)

Plot the Lorenz attractor in 2D/3D. Implemented in the
`LorenzVisualizationExt` extension.
"""
function plot_phase_portrait end

"""
    animate_comparison(true_sol::L63Solution, fitted_sol::L63Solution; kwargs...)

Create an animated comparison of two trajectories. Implemented in the
`LorenzVisualizationExt` extension.
"""
function animate_comparison end

"""
    create_training_gif(true_params::L63Parameters, param_history::Vector, loss_history::Vector, target_solution::L63Solution; kwargs...)

Create an animated training evolution. Implemented in the
`LorenzVisualizationExt` extension.
"""
function create_training_gif end

"""
    plot_loss_evolution(metrics::TrainingMetrics; kwargs...)

Plot loss evolution showing individual window losses, batch averages, and epoch averages.
Implemented in the `LorenzVisualizationExt` extension.
"""
function plot_loss_evolution end

"""
    plot_gradient_evolution(metrics::TrainingMetrics; kwargs...)

Plot gradient evolution showing individual window gradients vs averaged gradients.
Demonstrates chaotic individual behavior vs meaningful averaged information.
Implemented in the `LorenzVisualizationExt` extension.
"""
function plot_gradient_evolution end

"""
    plot_gradient_scatter(metrics::TrainingMetrics, batch_range=nothing; kwargs...)

Create scatter plots of individual gradients within batches to show chaotic behavior.
Implemented in the `LorenzVisualizationExt` extension.
"""
function plot_gradient_scatter end

"""
    create_gradient_chaos_gif(metrics::TrainingMetrics; kwargs...)

Create animated visualization showing how individual chaotic gradients average to meaningful information.
Implemented in the `LorenzVisualizationExt` extension.
"""
function create_gradient_chaos_gif end
