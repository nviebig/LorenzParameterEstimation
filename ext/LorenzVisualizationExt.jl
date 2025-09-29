module LorenzVisualizationExt

using LorenzParameterEstimation
using GLMakie
using Dates

using GLMakie: Observable, tightlimits!, axislegend

# Import the parent module functions to extend them
import LorenzParameterEstimation: plot_trajectory, plot_phase_portrait, animate_comparison, create_training_gif

"""
    plot_trajectory(solution::L63Solution; components=:xyz, kwargs...)

Plot trajectory components over time.
"""
function plot_trajectory(solution::L63Solution; components=:xyz, kwargs...)
    GLMakie.activate!()
    t = solution.t
    data = solution.u

    kw = Dict{Symbol, Any}(kwargs)
    color = get!(kw, :color) do
        haskey(kw, :linecolor) ? kw[:linecolor] : :dodgerblue
    end
    linewidth = get(kw, :linewidth, 2)

    fig = Figure(resolution=(900, 900))

    if components == :xyz
        labels = ["X", "Y", "Z"]
        for (i, comp) in enumerate(1:3)
            ax = Axis(fig[i, 1]; xlabel="t", ylabel=labels[i], title="$(labels[i]) Component")
            lines!(ax, t, data[:, comp]; color=color, linewidth=linewidth)
            tightlimits!(ax)
        end
        return fig
    elseif components == :xy
        ax = Axis(fig[1, 1]; xlabel="t", ylabel="value", title="X/Y Components")
        lines!(ax, t, data[:, 1]; color=:dodgerblue, linewidth=linewidth, label="X")
        lines!(ax, t, data[:, 2]; color=:darkorange, linewidth=linewidth, label="Y")
        axislegend(ax)
        tightlimits!(ax)
        return fig
    else
        idx = components isa Symbol ? (components == :x ? 1 : components == :y ? 2 : 3) : components
        ax = Axis(fig[1, 1]; xlabel="t", ylabel="value", title="Component $(idx)")
        lines!(ax, t, data[:, idx]; color=color, linewidth=linewidth)
        tightlimits!(ax)
        return fig
    end
end

"""
    plot_phase_portrait(solution::L63Solution; dims=(1,2,3), kwargs...)

Plot 3D phase portrait (attractor).
"""

function plot_phase_portrait(solution::L63Solution; dims=(1,2,3), kwargs...)
    GLMakie.activate!()
    data = solution.u

    kw = Dict{Symbol, Any}(kwargs)
    color = get!(kw, :color) do
        haskey(kw, :linecolor) ? kw[:linecolor] : :steelblue
    end
    linewidth = get(kw, :linewidth, 1.5)

    fig = Figure(resolution=(800, 700))

    if length(dims) == 3
        ax = Axis3(fig[1, 1]; xlabel="X", ylabel="Y", zlabel="Z", title="Lorenz Attractor")
        lines!(ax, data[:, dims[1]], data[:, dims[2]], data[:, dims[3]]; color=color, linewidth=linewidth)
        tightlimits!(ax)
        return fig
    elseif length(dims) == 2
        ax = Axis(fig[1, 1]; xlabel="Dim $(dims[1])", ylabel="Dim $(dims[2])", title="Phase Portrait")
        lines!(ax, data[:, dims[1]], data[:, dims[2]]; color=color, linewidth=linewidth)
        tightlimits!(ax)
        return fig
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

    GLMakie.activate!()
    N = min(length(true_sol), length(fitted_sol))
    stride = clamp(step, 1, max(1, N ÷ 20))
    indices = collect(1:stride:N)
    if indices[end] != N
        push!(indices, N)
    end

    fig = Figure(resolution=(900, 700))
    ax = Axis3(fig[1, 1]; xlabel="X", ylabel="Y", zlabel="Z")

    true_x = Observable(true_sol.u[1:1, 1])
    true_y = Observable(true_sol.u[1:1, 2])
    true_z = Observable(true_sol.u[1:1, 3])
    fitted_x = Observable(fitted_sol.u[1:1, 1])
    fitted_y = Observable(fitted_sol.u[1:1, 2])
    fitted_z = Observable(fitted_sol.u[1:1, 3])

    lines!(ax, true_x, true_y, true_z;
        color=RGBAf0(44 / 255, 123 / 255, 182 / 255, 0.9), linewidth=2,
        label="True")
    lines!(ax, fitted_x, fitted_y, fitted_z;
        color=RGBAf0(214 / 255, 39 / 255, 40 / 255, 0.9), linewidth=2,
        linestyle=:dash, label="Fitted")
    axislegend(ax, position=:lt)

    target_path = isabspath(filename) ? filename : abspath(filename)
    mkpath(dirname(target_path))

    record(fig, target_path, eachindex(indices); framerate=fps) do i
        n = indices[i]
        set!(true_x, true_sol.u[1:n, 1])
        set!(true_y, true_sol.u[1:n, 2])
        set!(true_z, true_sol.u[1:n, 3])
        set!(fitted_x, fitted_sol.u[1:n, 1])
        set!(fitted_y, fitted_sol.u[1:n, 2])
        set!(fitted_z, fitted_sol.u[1:n, 3])
        current_ρ = round(fitted_sol.system.params.ρ, digits=3)
        ax.title = "Lorenz Comparison (t = $(round(true_sol.t[n], digits=2)))\nρ_true=$(true_sol.system.params.ρ), ρ_fit=$(current_ρ)"
    end

    return target_path
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

    GLMakie.activate!()

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

    fig = Figure(resolution=(1100, 820))
    ax_phase = Axis3(fig[1, 1]; title="", xlabel="X", ylabel="Y", zlabel="Z")
    ax_loss = Axis(fig[1, 2]; xlabel="Epoch", ylabel="log10 RMSE Loss", title="Loss")
    ax_params = Axis(fig[2, 1]; xlabel="Epoch", ylabel="Parameter Value", title="Parameter Convergence")
    ax_xy = Axis(fig[2, 2]; xlabel="X", ylabel="Y", title="X-Y Projection")

    true_color = RGBAf0(44 / 255, 123 / 255, 182 / 255, 0.75)
    fitted_color = RGBAf0(214 / 255, 39 / 255, 40 / 255, 0.9)
    param_colors = Dict(:σ => RGBAf0(31 / 255, 119 / 255, 180 / 255, 1.0),
                        :ρ => RGBAf0(214 / 255, 39 / 255, 40 / 255, 1.0),
                        :β => RGBAf0(44 / 255, 160 / 255, 44 / 255, 1.0))
    param_labels = Dict(:σ => "σ", :ρ => "ρ", :β => "β")

    lines!(ax_phase, target_solution.u[:, 1], target_solution.u[:, 2], target_solution.u[:, 3];
        color=true_color, linewidth=1.5, label="True")
    lines!(ax_xy, target_solution.u[:, 1], target_solution.u[:, 2];
        color=true_color, linewidth=1.5, label="True")

    fitted_x = Observable(target_solution.u[1:1, 1])
    fitted_y = Observable(target_solution.u[1:1, 2])
    fitted_z = Observable(target_solution.u[1:1, 3])
    fitted_xy_x = Observable(target_solution.u[1:1, 1])
    fitted_xy_y = Observable(target_solution.u[1:1, 2])

    lines!(ax_phase, fitted_x, fitted_y, fitted_z; color=fitted_color, linewidth=2, linestyle=:dash, label="Fitted")
    lines!(ax_xy, fitted_xy_x, fitted_xy_y; color=fitted_color, linewidth=2, linestyle=:dash, label="Fitted")
    axislegend(ax_phase, position=:rt)
    axislegend(ax_xy, position=:rt)

    loss_eps = eps(eltype(loss_history))
    loss_epochs = Observable(collect(0:0))
    loss_values = Observable([log10(loss_history[1] + loss_eps)])
    lines!(ax_loss, loss_epochs, loss_values; color=RGBAf0(148 / 255, 103 / 255, 189 / 255, 1.0), linewidth=2)

    param_epoch_obs = Observable(collect(0:0))
    param_values = Dict{Symbol, Observable}()
    for sym in (:σ, :ρ, :β)
        vals = Observable([getfield(param_history[1], sym)])
        plt = lines!(ax_params, param_epoch_obs, vals;
            color=param_colors[sym], linewidth=2, label=param_labels[sym])
        param_values[sym] = vals
        hlines!(ax_params, [getfield(true_params, sym)]; color=param_colors[sym], linestyle=:dash, linewidth=1.5)
    end
    axislegend(ax_params, position=:rb)

    tightlimits!(ax_phase)
    tightlimits!(ax_xy)

    record(fig, target_path, eachindex(epoch_indices); framerate=fps) do frame_idx
        epoch = epoch_indices[frame_idx]
        params_idx = epoch + 1
        current_params = param_history[params_idx]
        loss_idx = max(epoch, 1)
        current_loss = loss_history[loss_idx]

        current_system = L63System(
            params=current_params,
            u0=target_solution.system.u0,
            tspan=target_solution.system.tspan,
            dt=target_solution.system.dt
        )
        current_sol = integrate(current_system)

        set!(fitted_x, current_sol.u[:, 1])
        set!(fitted_y, current_sol.u[:, 2])
        set!(fitted_z, current_sol.u[:, 3])
        set!(fitted_xy_x, current_sol.u[:, 1])
        set!(fitted_xy_y, current_sol.u[:, 2])

        loss_epochs_vec = collect(0:(loss_idx - 1))
        set!(loss_epochs, loss_epochs_vec)
        loss_data = log10.(loss_history[1:loss_idx] .+ loss_eps)
        set!(loss_values, loss_data)
        ax_loss.title = "Loss: $(round(current_loss, digits=6))"

        epochs_plot = collect(0:(params_idx - 1))
        set!(param_epoch_obs, epochs_plot)
        for sym in (:σ, :ρ, :β)
            history_vals = [getfield(p, sym) for p in param_history[1:params_idx]]
            set!(param_values[sym], history_vals)
        end

        param_info = string(
            "Epoch ", epoch,
            ": σ=", round(current_params.σ, digits=3),
            ", ρ=", round(current_params.ρ, digits=3),
            ", β=", round(current_params.β, digits=3)
        )
        ax_phase.title = param_info
        ax_xy.title = "X-Y Projection (Epoch $(epoch))"
    end

    return abspath(target_path)

end


end # module
