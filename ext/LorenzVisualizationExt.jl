module LorenzVisualizationExt

using LorenzParameterEstimation
using Plots, Images, FileIO

# Import the parent module functions to extend them
import LorenzParameterEstimation: plot_trajectory, plot_phase_portrait, animate_comparison, create_training_gif, 
    plot_loss_evolution, plot_gradient_evolution, plot_gradient_scatter, create_gradient_chaos_gif

"""
    plot_trajectory(solution::L63Solution; components=:xyz, kwargs...)

Plot trajectory components over time.
"""
function plot_trajectory(solution::L63Solution; components=:xyz, kwargs...)
    if components == :xyz
        p1 = plot(solution.t, solution.u[:, 1], label="X", title="X Component", kwargs...)
        p2 = plot(solution.t, solution.u[:, 2], label="Y", title="Y Component", kwargs...)  
        p3 = plot(solution.t, solution.u[:, 3], label="Z", title="Z Component", kwargs...)
        return plot(p1, p2, p3, layout=(3,1))
    elseif components == :xy
        return plot(solution.t, solution.u[:, 1:2], label=["X" "Y"], kwargs...)
    else
        idx = components isa Symbol ? (components == :x ? 1 : components == :y ? 2 : 3) : components
        return plot(solution.t, solution.u[:, idx], kwargs...)
    end
end

"""
    plot_phase_portrait(solution::L63Solution; dims=(1,2,3), kwargs...)

Plot 3D phase portrait (attractor).
"""

function plot_phase_portrait(solution::L63Solution; dims=(1,2,3), kwargs...)
    if length(dims) == 3
        return plot(solution.u[:, dims[1]], solution.u[:, dims[2]], solution.u[:, dims[3]], 
                   xlabel="X", ylabel="Y", zlabel="Z", kwargs...)
    elseif length(dims) == 2  
        return plot(solution.u[:, dims[1]], solution.u[:, dims[2]], kwargs...)
    else
        error("dims must specify 2 or 3 dimensions")
    end
end

"""
    animate_comparison(true_sol::L63Solution, fitted_sol::L63Solution; 
                      fps=15, filename="comparison.gif")

Create animated comparison of two trajectories.
"""
function animate_comparison(true_sol::L63Solution{T}, fitted_sol::L63Solution{T}; 
                          fps::Int=15, step::Int=50, 
                          filename::String="lorenz_comparison.gif") where {T}
    
    gr()  # Use GR backend for animations
    N = min(length(true_sol), length(fitted_sol))
    indices = 1:clamp(step, 1, max(1, N÷20)):N
    
    anim = Animation()
    
    for n in indices
        p = plot(true_sol.u[1:n, 1], true_sol.u[1:n, 2], true_sol.u[1:n, 3],
                label="True (ρ=$(true_sol.system.params.ρ))",
                linecolor=:blue, linewidth=1.0, alpha=0.8,
                camera=(45, 30), size=(900, 700))
        
        plot!(p, fitted_sol.u[1:n, 1], fitted_sol.u[1:n, 2], fitted_sol.u[1:n, 3],
              label="Fitted (ρ=$(round(fitted_sol.system.params.ρ, digits=3)))", 
              linecolor=:red, linewidth=1.0, alpha=0.8, linestyle=:dash)
        
        title!(p, "Lorenz Comparison (t = $(round(true_sol.t[n], digits=2)))")
        xlabel!(p, "X"); ylabel!(p, "Y"); zlabel!(p, "Z")
        
        frame(anim, p)
    end
    
    gif(anim, filename, fps=fps)
    return filename
end

"""
    create_training_gif(true_params::L63Parameters, param_history::Vector, 
                       loss_history::Vector, target_solution::L63Solution;
                       fps=2, filename="training_evolution.gif", stride=1)

Create animated training evolution showing parameter convergence.
"""
function create_training_gif(true_params::L63Parameters, param_history::Vector,
                           loss_history::Vector, target_solution::L63Solution;
                           fps::Int=2, filename::String="training_evolution.gif",
                           stride::Int=1)
    
    gr()  # GR backend
    frames = []
    
    n_epochs = length(loss_history)
    params_len = length(param_history)
    params_len == n_epochs + 1 || error("param_history must have length epochs+1 (includes epoch 0 state)")
    stride = max(stride, 1)
    epoch_indices = collect(0:stride:n_epochs)
    if epoch_indices[end] != n_epochs
        push!(epoch_indices, n_epochs)
    end
    
    for epoch in epoch_indices
        params_idx = epoch + 1
        current_params = param_history[params_idx]
        loss_idx = max(epoch, 1)
        current_loss = loss_history[loss_idx]
        
        # Generate trajectory with current parameters
        current_system = L63System(
            params=current_params,
            u0=target_solution.system.u0,
            tspan=target_solution.system.tspan, 
            dt=target_solution.system.dt
        )
        current_sol = integrate(current_system)
        
        # Create 4-panel plot
        p1 = plot(target_solution.u[:, 1], target_solution.u[:, 2], target_solution.u[:, 3],
                 label="True (ρ=$(true_params.ρ))", linecolor=:blue, linewidth=1.0, alpha=0.7)
        plot!(p1, current_sol.u[:, 1], current_sol.u[:, 2], current_sol.u[:, 3],
              label="Fitted (ρ=$(round(current_params.ρ, digits=3)))",
              linecolor=:red, linewidth=1.0, alpha=0.8, linestyle=:dash)
        title!(p1, "Epoch $epoch: ρ = $(round(current_params.ρ, digits=4))")
        xlabel!(p1, "X"); ylabel!(p1, "Y"); zlabel!(p1, "Z")
        
        # Loss history
        loss_epochs = 1:max(loss_idx, 1)
        loss_vals = loss_history[loss_epochs]
        p2 = plot(loss_epochs .- 1, loss_vals, linewidth=2, color=:red,
                 xlabel="Epoch", ylabel="RMSE Loss",
                 title="Loss: $(round(current_loss, digits=6))",
                 legend=false, yscale=:log10)
        
        # Parameter evolution
        ρ_vals = [p.ρ for p in param_history[1:params_idx]]
        p3 = plot(0:(params_idx-1), ρ_vals, linewidth=2, color=:green,
                 xlabel="Epoch", ylabel="ρ parameter", title="ρ Evolution",
                 legend=false)
        hline!(p3, [true_params.ρ], linestyle=:dot, color=:black, linewidth=2, alpha=0.7)
        
        # X-Y projection comparison
        p4 = plot(target_solution.u[:, 1], target_solution.u[:, 2], 
                 label="True", linecolor=:blue, linewidth=1.0, alpha=0.7)
        plot!(p4, current_sol.u[:, 1], current_sol.u[:, 2],
              label="Fitted", linecolor=:red, linewidth=1.0, alpha=0.8, linestyle=:dash)
        title!(p4, "X-Y Projection")
        xlabel!(p4, "X"); ylabel!(p4, "Y")
        
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
        push!(frames, combined_plot)
    end
    
    # Create animation
    anim = Animation()
    for plt in frames
        frame(anim, plt)
    end
    
    gif(anim, filename, fps=fps)
    return filename
end

"""
    plot_loss_evolution(metrics::TrainingMetrics; show_individual=true, show_batches=true, 
                       show_epochs=true, max_individual=1000)

Plot loss evolution showing individual window losses, batch averages, and epoch averages.
"""
function plot_loss_evolution(metrics::LorenzParameterEstimation.TrainingMetrics{T}; 
                            show_individual::Bool=true, show_batches::Bool=true, 
                            show_epochs::Bool=true, max_individual::Int=1000) where {T}
    
    p = plot(xlabel="Training Progress", ylabel="Loss", title="Loss Evolution", yscale=:log10)
    
    if show_individual && !isempty(metrics.individual_losses)
        # Sample individual losses if too many
        n_individual = length(metrics.individual_losses)
        if n_individual > max_individual
            step = max(1, n_individual ÷ max_individual)
            sampled_indices = 1:step:n_individual
            sampled_losses = metrics.individual_losses[sampled_indices]
            sampled_x = sampled_indices
        else
            sampled_losses = metrics.individual_losses
            sampled_x = 1:length(sampled_losses)
        end
        
        scatter!(p, sampled_x, sampled_losses, alpha=0.3, markersize=1, 
                label="Individual Windows", color=:lightblue)
    end
    
    if show_batches && !isempty(metrics.batch_losses)
        n_batches = length(metrics.batch_losses)
        plot!(p, 1:n_batches, metrics.batch_losses, linewidth=2, 
              label="Batch Averages", color=:orange)
    end
    
    if show_epochs && !isempty(metrics.epoch_losses)
        n_epochs = length(metrics.epoch_losses)
        plot!(p, 1:n_epochs, metrics.epoch_losses, linewidth=3, 
              label="Epoch Averages", color=:red)
    end
    
    return p
end

"""
    plot_gradient_evolution(metrics::TrainingMetrics; parameter=:σ, show_individual=true, 
                           show_batches=true, show_epochs=true, max_individual=1000)

Plot gradient evolution for a specific parameter, showing chaotic individual behavior vs meaningful averages.
"""
function plot_gradient_evolution(metrics::LorenzParameterEstimation.TrainingMetrics{T}; 
                                parameter::Symbol=:σ, show_individual::Bool=true,
                                show_batches::Bool=true, show_epochs::Bool=true, 
                                max_individual::Int=1000) where {T}
    
    # Select the appropriate gradient arrays
    individual_grads, batch_grads, epoch_grads = if parameter == :σ
        (metrics.individual_gradients_σ, metrics.batch_gradients_σ, metrics.epoch_gradients_σ)
    elseif parameter == :ρ
        (metrics.individual_gradients_ρ, metrics.batch_gradients_ρ, metrics.epoch_gradients_ρ)
    elseif parameter == :β
        (metrics.individual_gradients_β, metrics.batch_gradients_β, metrics.epoch_gradients_β)
    else
        error("Parameter must be :σ, :ρ, or :β")
    end
    
    p = plot(xlabel="Training Progress", ylabel="Gradient Value", 
             title="Gradient Evolution for Parameter $parameter")
    
    if show_individual && !isempty(individual_grads)
        # Sample individual gradients if too many
        n_individual = length(individual_grads)
        if n_individual > max_individual
            step = max(1, n_individual ÷ max_individual)
            sampled_indices = 1:step:n_individual
            sampled_grads = individual_grads[sampled_indices]
            sampled_x = sampled_indices
        else
            sampled_grads = individual_grads
            sampled_x = 1:length(sampled_grads)
        end
        
        scatter!(p, sampled_x, sampled_grads, alpha=0.3, markersize=1, 
                label="Individual Windows (Chaotic)", color=:lightblue)
    end
    
    if show_batches && !isempty(batch_grads)
        n_batches = length(batch_grads)
        plot!(p, 1:n_batches, batch_grads, linewidth=2, 
              label="Batch Averages (Meaningful)", color=:orange)
    end
    
    if show_epochs && !isempty(epoch_grads)
        n_epochs = length(epoch_grads)
        plot!(p, 1:n_epochs, epoch_grads, linewidth=3, 
              label="Epoch Averages (Stable)", color=:red)
    end
    
    # Add horizontal line at zero for reference
    hline!(p, [0], linestyle=:dot, color=:black, alpha=0.5, label="Zero")
    
    return p
end

"""
    plot_gradient_scatter(metrics::TrainingMetrics, batch_range=nothing; 
                         parameters=[:σ, :ρ, :β])

Create scatter plots of individual gradients within batches to show chaotic behavior.
"""
function plot_gradient_scatter(metrics::LorenzParameterEstimation.TrainingMetrics{T}, 
                              batch_range=nothing; parameters::Vector{Symbol}=[:σ, :ρ, :β]) where {T}
    
    if batch_range === nothing
        batch_range = 1:min(10, maximum(metrics.batch_indices))  # First 10 batches by default
    end
    
    plots_list = []
    
    for param in parameters
        individual_grads = if param == :σ
            metrics.individual_gradients_σ
        elseif param == :ρ
            metrics.individual_gradients_ρ
        elseif param == :β
            metrics.individual_gradients_β
        else
            continue
        end
        
        p = plot(xlabel="Batch Index", ylabel="Gradient Value", 
                title="Individual Gradients: $param")
        
        for batch_idx in batch_range
            batch_mask = metrics.batch_indices .== batch_idx
            if any(batch_mask)
                batch_grads = individual_grads[batch_mask]
                x_vals = fill(batch_idx, length(batch_grads))
                scatter!(p, x_vals, batch_grads, alpha=0.6, markersize=3, 
                        label=batch_idx == first(batch_range) ? "Individual Windows" : "")
            end
        end
        
        # Add batch averages as red diamonds
        for batch_idx in batch_range
            batch_mask = metrics.batch_indices .== batch_idx
            if any(batch_mask)
                batch_grads = individual_grads[batch_mask]
                avg_grad = mean(batch_grads)
                scatter!(p, [batch_idx], [avg_grad], marker=:diamond, markersize=6, 
                        color=:red, label=batch_idx == first(batch_range) ? "Batch Average" : "")
            end
        end
        
        push!(plots_list, p)
    end
    
    return plot(plots_list..., layout=(length(plots_list), 1), size=(800, 300*length(plots_list)))
end

"""
    create_gradient_chaos_gif(metrics::TrainingMetrics; parameter=:σ, fps=2, 
                             filename="gradient_chaos.gif", max_batches=20)

Create animated visualization showing how individual chaotic gradients average to meaningful information.
"""
function create_gradient_chaos_gif(metrics::LorenzParameterEstimation.TrainingMetrics{T}; 
                                  parameter::Symbol=:σ, fps::Int=2,
                                  filename::String="gradient_chaos.gif", 
                                  max_batches::Int=20) where {T}
    
    # Select gradients for the chosen parameter
    individual_grads, batch_grads = if parameter == :σ
        (metrics.individual_gradients_σ, metrics.batch_gradients_σ)
    elseif parameter == :ρ
        (metrics.individual_gradients_ρ, metrics.batch_gradients_ρ)
    elseif parameter == :β
        (metrics.individual_gradients_β, metrics.batch_gradients_β)
    else
        error("Parameter must be :σ, :ρ, or :β")
    end
    
    if isempty(individual_grads) || isempty(batch_grads)
        error("No gradient data available for parameter $parameter")
    end
    
    n_batches = min(max_batches, length(batch_grads))
    anim = Animation()
    
    for batch_idx in 1:n_batches
        # Individual gradients for this batch
        batch_mask = metrics.batch_indices .== batch_idx
        current_batch_grads = individual_grads[batch_mask]
        
        if isempty(current_batch_grads)
            continue
        end
        
        # Create 3-panel plot
        # Panel 1: Scatter of individual gradients
        p1 = scatter(fill(1, length(current_batch_grads)), current_batch_grads,
                    alpha=0.6, markersize=4, color=:lightblue, 
                    title="Individual Gradients (Batch $batch_idx)")
        scatter!(p1, [1], [mean(current_batch_grads)], marker=:diamond, 
                markersize=8, color=:red, label="Average")
        xlabel!(p1, ""); ylabel!(p1, "Gradient Value")
        xlims!(p1, 0.5, 1.5)
        
        # Panel 2: Running average of batch averages
        p2 = plot(1:batch_idx, batch_grads[1:batch_idx], linewidth=3, color=:red,
                 title="Batch Averages (Meaningful Signal)")
        scatter!(p2, 1:batch_idx, batch_grads[1:batch_idx], markersize=4, color=:red)
        xlabel!(p2, "Batch Index"); ylabel!(p2, "Average Gradient")
        hline!(p2, [0], linestyle=:dot, color=:black, alpha=0.5)
        
        # Panel 3: Histogram of current batch gradients
        p3 = histogram(current_batch_grads, bins=min(10, length(current_batch_grads)),
                      alpha=0.7, color=:lightblue, 
                      title="Distribution (Batch $batch_idx)")
        vline!(p3, [mean(current_batch_grads)], linewidth=3, color=:red, 
               label="Mean = $(round(mean(current_batch_grads), digits=4))")
        xlabel!(p3, "Gradient Value"); ylabel!(p3, "Count")
        
        combined_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
        frame(anim, combined_plot)
    end
    
    gif(anim, filename, fps=fps)
    return filename
end

end # module
