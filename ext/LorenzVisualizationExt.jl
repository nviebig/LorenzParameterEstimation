module LorenzVisualizationExt

using LorenzParameterEstimation
using Plots, Images, FileIO, Dates

# Import the parent module functions to extend them
import LorenzParameterEstimation: plot_trajectory, plot_phase_portrait, animate_comparison, create_training_gif

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
                       fps=2, filename=nothing, stride=1, label="", output_dir=nothing)

Create animated training evolution showing parameter convergence. When `filename` is omitted,
a timestamped name is generated within `data/generated_gifs` (relative to the repository root).
"""
function create_training_gif(true_params::L63Parameters, param_history::Vector,
                           loss_history::Vector, target_solution::L63Solution;
                           fps::Int=2, filename::Union{Nothing,String}=nothing,
                           stride::Int=1, label::AbstractString="",
                           output_dir::Union{Nothing,String}=nothing)
    
    gr()  # GR backend
    frames = Any[]

    n_epochs = length(loss_history)
    params_len = length(param_history)
    params_len == n_epochs + 1 || error("param_history must have length epochs+1 (includes epoch 0 state)")
    stride = max(stride, 1)
    epoch_indices = collect(0:stride:n_epochs)
    if epoch_indices[end] != n_epochs
        push!(epoch_indices, n_epochs)
    end

    default_root = joinpath(@__DIR__, "..", "..", "data", "generated_gifs")
    chosen_root = isnothing(output_dir) ? default_root : output_dir
    root_abs = isabspath(chosen_root) ? chosen_root : abspath(chosen_root)

    sanitized_label = lowercase(label)
    sanitized_label = replace(sanitized_label, r"[^a-z0-9]+" => "_")
    sanitized_label = strip(sanitized_label, '_')
    base_name = isempty(sanitized_label) ? "lorenz_training" : string("lorenz_training_", sanitized_label)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

    if filename === nothing
        fname = string(base_name, "_", timestamp, ".gif")
        target_path = joinpath(root_abs, fname)
    else
        fname = String(filename)
        fname = endswith(lowercase(fname), ".gif") ? fname : string(fname, ".gif")
        target_path = isabspath(fname) ? fname : joinpath(root_abs, fname)
    end

    mkpath(dirname(target_path))

    param_syms = (:σ, :ρ, :β)
    param_colors = Dict(:σ => :blue, :ρ => :red, :β => :green)
    param_labels = Dict(:σ => "σ", :ρ => "ρ", :β => "β")

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
        param_info = string(
            "Epoch ", epoch, ": σ=", round(current_params.σ, digits=3),
            ", ρ=", round(current_params.ρ, digits=3),
            ", β=", round(current_params.β, digits=3)
        )
        p1 = plot(target_solution.u[:, 1], target_solution.u[:, 2], target_solution.u[:, 3],
                 label="True", linecolor=:blue, linewidth=1.0, alpha=0.7)
        plot!(p1, current_sol.u[:, 1], current_sol.u[:, 2], current_sol.u[:, 3],
              label="Fitted", linecolor=:red, linewidth=1.0, alpha=0.8, linestyle=:dash)
        title!(p1, param_info)
        xlabel!(p1, "X"); ylabel!(p1, "Y"); zlabel!(p1, "Z")

        # Loss history
        loss_epochs = 1:max(loss_idx, 1)
        loss_vals = loss_history[loss_epochs]
        p2 = plot(loss_epochs .- 1, loss_vals, linewidth=2, color=:purple,
                 xlabel="Epoch", ylabel="RMSE Loss",
                 title="Loss: $(round(current_loss, digits=6))",
                 legend=false, yscale=:log10)

        # Parameter evolution
        epochs_plot = 0:(params_idx - 1)
        p3 = plot(legend=:bottomright)
        for sym in param_syms
            history_vals = [getfield(p, sym) for p in param_history[1:params_idx]]
            plot!(p3, epochs_plot, history_vals;
                  linewidth=2, color=param_colors[sym], label=param_labels[sym])
            hline!(p3, [getfield(true_params, sym)];
                   linestyle=:dash, color=param_colors[sym], alpha=0.6,
                   label="$(param_labels[sym]) (true)")
        end
        xlabel!(p3, "Epoch")
        ylabel!(p3, "Parameter Value")
        title!(p3, "Parameter Convergence")

        # X-Y projection comparison
        p4 = plot(target_solution.u[:, 1], target_solution.u[:, 2],
                 label="True", linecolor=:blue, linewidth=1.0, alpha=0.7)
        plot!(p4, current_sol.u[:, 1], current_sol.u[:, 2],
              label="Fitted", linecolor=:red, linewidth=1.0, alpha=0.8, linestyle=:dash)
        title!(p4, "X-Y Projection")
        xlabel!(p4, "X"); ylabel!(p4, "Y")

        combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 820))
        push!(frames, combined_plot)
    end

    # Create animation
    anim = Animation()
    for plt in frames
        frame(anim, plt)
    end

    gif(anim, target_path, fps=fps)
    return abspath(target_path)

end


end # module
