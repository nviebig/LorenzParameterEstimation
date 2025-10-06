#!/usr/bin/env julia

"""
Gradient Chaos Demonstration
===========================

This script demonstrates how individual gradients can be chaotic and "all over the place" 
due to the sensitive dependence on initial conditions in the Lorenz system, but how 
averaging over batches provides meaningful, non-zero information for parameter estimation.

The key insight: While individual windows may give wildly different gradients due to chaos,
the average gradient over multiple windows contains the signal needed for optimization.
"""

using LorenzParameterEstimation
using Plots  # Must be loaded first to enable visualization extension
using Random
using Statistics
using Printf

# Set up plotting backend
gr()

# Set random seed for reproducibility  
Random.seed!(42)

println("🌀 Gradient Chaos Demonstration for Lorenz Parameter Estimation")
println("=" ^ 70)

# ================================ Setup ================================

# Create a reference Lorenz system with known parameters
println("📐 Setting up reference system...")
true_params = L63Parameters(σ=10.0, ρ=28.0, β=8/3)

# Use longer integration with multiple random initial conditions
# This will create the chaotic behavior we want to demonstrate
tspan = (0.0, 50.0)  # Longer integration time
dt = 0.01
n_points = Int((tspan[2] - tspan[1]) / dt) + 1

# Generate multiple trajectories from different initial conditions
println("🎯 Generating reference data with random initial conditions...")
reference_trajectories = []
initial_conditions = []

# Create 5 different initial conditions around the attractor
base_ics = [
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, 25.0], 
    [10.0, 10.0, 10.0],
    [0.0, 1.0, 20.0],
    [5.0, -5.0, 15.0]
]

for (i, ic) in enumerate(base_ics)
    # Add small random perturbations to each initial condition
    perturbed_ic = ic .+ 0.1 * randn(3)
    
    system = L63System(
        params=true_params,
        u0=perturbed_ic,
        tspan=tspan,
        dt=dt
    )
    
    solution = integrate(system)
    push!(reference_trajectories, solution)
    push!(initial_conditions, perturbed_ic)
    
    println("  IC $i: [$(join(round.(perturbed_ic, digits=3), ", "))]")
end

# ================================ Training Setup ================================

# Start with incorrect initial parameters (this is what we're trying to estimate)
initial_guess = L63Parameters(σ=8.0, ρ=25.0, β=3.0)
println("\n🎲 Initial parameter guess:")
println("  σ = $(initial_guess.σ) (true: $(true_params.σ))")
println("  ρ = $(initial_guess.ρ) (true: $(true_params.ρ))")  
println("  β = $(initial_guess.β) (true: $(true_params.β))")

# Training configuration designed to show chaotic behavior
println("\n⚙️  Training configuration:")
training_config = (
    epochs = 50,
    window_size = 200,      # Longer windows to see more chaos
    stride = 50,            # More overlap
    batch_size = 16,        # Moderate batch size
    eval_every = 5,
    verbose = true
)

for (key, val) in pairs(training_config)
    println("  $key = $val")
end

# ================================ Custom Training with Metrics Tracking ================================

println("\n🚀 Starting training with gradient tracking...")

# Create metrics tracking structure
metrics = TrainingMetrics()

# We'll implement a simplified tracking version of training to capture detailed metrics
function train_with_tracking!(params, target_solutions; 
                             epochs=50, window_size=200, stride=50, batch_size=16, eval_every=5, verbose=true)
    
    reset_metrics!(metrics)
    current_params = deepcopy(params)
    param_history = [deepcopy(current_params)]
    loss_history = Float64[]
    
    # Create windows from all trajectories
    all_windows = []
    for sol in target_solutions
        n_windows = (length(sol) - window_size) ÷ stride + 1
        for i in 1:n_windows
            window_start = (i-1) * stride + 1
            push!(all_windows, (sol, window_start))
        end
    end
    
    println("  Total training windows: $(length(all_windows))")
    
    # Training loop
    for epoch in 1:epochs
        
        # Shuffle windows for this epoch
        shuffled_windows = shuffle(all_windows)
        
        # Process in batches
        n_batches = ceil(Int, length(shuffled_windows) / batch_size)
        epoch_loss = 0.0
        epoch_grad_σ = 0.0
        epoch_grad_ρ = 0.0
        epoch_grad_β = 0.0
        
        for batch_idx in 1:n_batches
            batch_start = (batch_idx - 1) * batch_size + 1
            batch_end = min(batch_idx * batch_size, length(shuffled_windows))
            batch_windows = shuffled_windows[batch_start:batch_end]
            
            # Compute gradients for each window in batch
            batch_loss = 0.0
            batch_grad_σ = 0.0
            batch_grad_ρ = 0.0
            batch_grad_β = 0.0
            
            for (sol, window_start) in batch_windows
                loss_val, gradients = compute_gradients_with_tracking(
                    current_params, sol, window_start, window_size, window_rmse;
                    metrics=metrics, batch_idx=batch_idx
                )
                
                batch_loss += loss_val
                batch_grad_σ += gradients.σ
                batch_grad_ρ += gradients.ρ
                batch_grad_β += gradients.β
            end
            
            # Average over batch
            batch_size_actual = length(batch_windows)
            batch_loss /= batch_size_actual
            batch_grad_σ /= batch_size_actual
            batch_grad_ρ /= batch_size_actual
            batch_grad_β /= batch_size_actual
            
            # Record batch metrics
            avg_batch_grads = L63Parameters(batch_grad_σ, batch_grad_ρ, batch_grad_β)
            record_batch_metrics!(metrics, batch_loss, avg_batch_grads)
            
            # Update parameters (simple gradient descent)
            learning_rate = 0.001
            current_params = L63Parameters(
                current_params.σ - learning_rate * batch_grad_σ,
                current_params.ρ - learning_rate * batch_grad_ρ,
                current_params.β - learning_rate * batch_grad_β
            )
            
            epoch_loss += batch_loss
            epoch_grad_σ += batch_grad_σ
            epoch_grad_ρ += batch_grad_ρ
            epoch_grad_β += batch_grad_β
        end
        
        # Average over epoch
        epoch_loss /= n_batches
        epoch_grad_σ /= n_batches
        epoch_grad_ρ /= n_batches
        epoch_grad_β /= n_batches
        
        # Record epoch metrics
        avg_epoch_grads = L63Parameters(epoch_grad_σ, epoch_grad_ρ, epoch_grad_β)
        record_epoch_metrics!(metrics, epoch_loss, avg_epoch_grads)
        
        push!(loss_history, epoch_loss)
        push!(param_history, deepcopy(current_params))
        
        if epoch % eval_every == 0
            println("  Epoch $epoch: Loss = $(round(epoch_loss, digits=6)), " *
                   "σ = $(round(current_params.σ, digits=3)), " *
                   "ρ = $(round(current_params.ρ, digits=3)), " *
                   "β = $(round(current_params.β, digits=3))")
        end
    end
    
    return current_params, param_history, loss_history
end

# Run training
final_params, param_history, loss_history = train_with_tracking!(
    initial_guess, reference_trajectories;
    epochs=training_config.epochs,
    window_size=training_config.window_size,
    stride=training_config.stride,
    batch_size=training_config.batch_size,
    eval_every=training_config.eval_every,
    verbose=training_config.verbose
)

println("\n✅ Training completed!")
println("🎯 Final parameters:")
println("  σ = $(round(final_params.σ, digits=3)) (true: $(true_params.σ), error: $(round(abs(final_params.σ - true_params.σ), digits=3)))")
println("  ρ = $(round(final_params.ρ, digits=3)) (true: $(true_params.ρ), error: $(round(abs(final_params.ρ - true_params.ρ), digits=3)))")  
println("  β = $(round(final_params.β, digits=3)) (true: $(true_params.β), error: $(round(abs(final_params.β - true_params.β), digits=3)))")

# ================================ Analysis and Visualization ================================

println("\n📊 Analyzing gradient behavior...")

# Check if we have gradient data
if !isempty(metrics.individual_gradients_σ)
    n_individual = length(metrics.individual_gradients_σ)
    n_batches = length(metrics.batch_gradients_σ)
    n_epochs = length(metrics.epoch_gradients_σ)
    
    println("  📈 Collected $n_individual individual gradient measurements")
    println("  📈 Collected $n_batches batch averages")  
    println("  📈 Collected $n_epochs epoch averages")
    
    # Analyze gradient statistics for a few representative batches
    println("\n🔍 Gradient variability analysis:")
    for batch_idx in [1, 5, 10, min(15, n_batches)]
        if batch_idx <= n_batches
            stats = compute_gradient_statistics(metrics, batch_idx)
            if stats !== nothing
                println("  Batch $batch_idx ($(stats.count) windows):")
                println("    σ: mean=$(round(stats.σ.mean, digits=4)), std=$(round(stats.σ.std, digits=4)), range=[$(round(stats.σ.min, digits=4)), $(round(stats.σ.max, digits=4))]")
                println("    ρ: mean=$(round(stats.ρ.mean, digits=4)), std=$(round(stats.ρ.std, digits=4)), range=[$(round(stats.ρ.min, digits=4)), $(round(stats.ρ.max, digits=4))]")
                println("    β: mean=$(round(stats.β.mean, digits=4)), std=$(round(stats.β.std, digits=4)), range=[$(round(stats.β.min, digits=4)), $(round(stats.β.max, digits=4))]")
            end
        end
    end
    
    # Create visualizations
    println("\n🎨 Creating visualizations...")
    
    # First, let's try to manually create some basic plots using Plots directly
    
    # 1. Loss evolution plot
    println("  📉 Loss evolution plot...")
    
    # Basic loss plot
    p_loss = plot(xlabel="Measurement Index", ylabel="Loss (log scale)", 
                 title="Loss Evolution", yscale=:log10, size=(800, 600))
    
        # Plot individual losses (sampled)
        if !isempty(metrics.individual_losses)
            n_individual = length(metrics.individual_losses)
            max_individual = 2000
            if n_individual > max_individual
                sample_step = max(1, n_individual ÷ max_individual)
                sampled_indices = 1:sample_step:n_individual
                sampled_losses = metrics.individual_losses[sampled_indices]
                sampled_x = sampled_indices
            else
                sampled_losses = metrics.individual_losses
                sampled_x = 1:length(sampled_losses)
            end
            
            scatter!(p_loss, sampled_x, sampled_losses, alpha=0.3, markersize=1, 
                    label="Individual Windows", color=:lightblue)
        end
    
    # Plot batch averages
    if !isempty(metrics.batch_losses)
        n_batches = length(metrics.batch_losses)
        plot!(p_loss, 1:n_batches, metrics.batch_losses, linewidth=2, 
              label="Batch Averages", color=:orange)
    end
    
    # Plot epoch averages
    if !isempty(metrics.epoch_losses)
        n_epochs = length(metrics.epoch_losses)
        plot!(p_loss, 1:n_epochs, metrics.epoch_losses, linewidth=3, 
              label="Epoch Averages", color=:red)
    end
    
    savefig(p_loss, "loss_evolution.png")
    
    # 2. Gradient evolution plots for each parameter
    println("  📈 Gradient evolution plots...")
    
    for (param_name, individual_grads, batch_grads, epoch_grads) in [
        ("sigma", metrics.individual_gradients_σ, metrics.batch_gradients_σ, metrics.epoch_gradients_σ),
        ("rho", metrics.individual_gradients_ρ, metrics.batch_gradients_ρ, metrics.epoch_gradients_ρ),
        ("beta", metrics.individual_gradients_β, metrics.batch_gradients_β, metrics.epoch_gradients_β)
    ]
        
        p_grad = plot(xlabel="Measurement Index", ylabel="Gradient Value", 
                     title="Gradient Evolution for Parameter $param_name", size=(800, 600))
        
        # Individual gradients (sampled)
        if !isempty(individual_grads)
            n_individual = length(individual_grads)
            max_individual = 2000
            if n_individual > max_individual
                sample_step = max(1, n_individual ÷ max_individual)
                sampled_indices = 1:sample_step:n_individual
                sampled_grads = individual_grads[sampled_indices]
                sampled_x = sampled_indices
            else
                sampled_grads = individual_grads
                sampled_x = 1:length(sampled_grads)
            end
            
            scatter!(p_grad, sampled_x, sampled_grads, alpha=0.3, markersize=1, 
                    label="Individual Windows (Chaotic)", color=:lightblue)
        end
        
        # Batch averages
        if !isempty(batch_grads)
            n_batches = length(batch_grads)
            plot!(p_grad, 1:n_batches, batch_grads, linewidth=2, 
                  label="Batch Averages (Meaningful)", color=:orange)
        end
        
        # Epoch averages
        if !isempty(epoch_grads)
            n_epochs = length(epoch_grads)
            plot!(p_grad, 1:n_epochs, epoch_grads, linewidth=3, 
                  label="Epoch Averages (Stable)", color=:red)
        end
        
        # Add horizontal line at zero
        hline!(p_grad, [0], linestyle=:dot, color=:black, alpha=0.5, label="Zero")
        
        savefig(p_grad, "gradient_evolution_$(param_name).png")
    end
    
    # 3. Scatter plots showing chaotic behavior within batches
    println("  🎲 Gradient scatter plots...")
    
    batch_range = 1:min(10, maximum(metrics.batch_indices))
    plots_list = []
    
    for (param_name, individual_grads) in [
        ("sigma", metrics.individual_gradients_σ),
        ("rho", metrics.individual_gradients_ρ),
        ("beta", metrics.individual_gradients_β)
    ]
        
        p = plot(xlabel="Batch Index", ylabel="Gradient Value", 
                title="Individual Gradients: $param_name", size=(800, 400))
        
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
    
    combined_scatter = plot(plots_list..., layout=(3, 1), size=(800, 900))
    savefig(combined_scatter, "gradient_scatter.png")
    
    # 4. Animated demonstration of chaos vs meaningful averages
    println("  🎬 Creating animated demonstrations...")
    
    # Create simple GIF for sigma parameter
    parameter = :σ
    individual_grads = metrics.individual_gradients_σ
    batch_grads = metrics.batch_gradients_σ
    
    if !isempty(individual_grads) && !isempty(batch_grads)
        n_batches = min(20, length(batch_grads))
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
                        title="Individual Gradients (Batch $batch_idx)",
                        ylabel="Gradient Value", xlabel="")
            scatter!(p1, [1], [mean(current_batch_grads)], marker=:diamond, 
                    markersize=8, color=:red, label="Average")
            xlims!(p1, 0.5, 1.5)
            
            # Panel 2: Running average of batch averages
            p2 = plot(1:batch_idx, batch_grads[1:batch_idx], linewidth=3, color=:red,
                     title="Batch Averages (Meaningful Signal)",
                     xlabel="Batch Index", ylabel="Average Gradient")
            scatter!(p2, 1:batch_idx, batch_grads[1:batch_idx], markersize=4, color=:red)
            hline!(p2, [0], linestyle=:dot, color=:black, alpha=0.5)
            
            # Panel 3: Histogram of current batch gradients
            p3 = histogram(current_batch_grads, bins=min(10, length(current_batch_grads)),
                          alpha=0.7, color=:lightblue, 
                          title="Distribution (Batch $batch_idx)",
                          xlabel="Gradient Value", ylabel="Count")
            vline!(p3, [mean(current_batch_grads)], linewidth=3, color=:red, 
                   label="Mean = $(round(mean(current_batch_grads), digits=4))")
            
            combined_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
            frame(anim, combined_plot)
        end
        
        gif_filename = "gradient_chaos_sigma.gif"
        gif(anim, gif_filename, fps=1)
        println("    ✅ Saved: $gif_filename")
    end
    
    # 5. Combined summary plot
    println("  📋 Creating summary visualization...")
    
    # Create 4-panel summary
    p1 = deepcopy(p_loss)
    title!(p1, "Loss Evolution")
    
    p2 = plot(xlabel="Index", ylabel="Gradient Value", title="σ Gradients")
    if !isempty(metrics.individual_gradients_σ)
        n_individual = length(metrics.individual_gradients_σ)
        max_individual = 1000
        if n_individual > max_individual
            sample_step = max(1, n_individual ÷ max_individual)
            sampled_indices = 1:sample_step:n_individual
            sampled_grads = metrics.individual_gradients_σ[sampled_indices]
            sampled_x = sampled_indices
        else
            sampled_grads = metrics.individual_gradients_σ
            sampled_x = 1:length(sampled_grads)
        end
        scatter!(p2, sampled_x, sampled_grads, alpha=0.3, markersize=1, 
                label="Individual", color=:lightblue)
    end
    if !isempty(metrics.batch_gradients_σ)
        plot!(p2, 1:length(metrics.batch_gradients_σ), metrics.batch_gradients_σ, 
              linewidth=2, label="Batch Avg", color=:orange)
    end
    
    p3 = plot(xlabel="Index", ylabel="Gradient Value", title="ρ Gradients")
    if !isempty(metrics.individual_gradients_ρ)
        n_individual = length(metrics.individual_gradients_ρ)
        max_individual = 1000
        if n_individual > max_individual
            sample_step = max(1, n_individual ÷ max_individual)
            sampled_indices = 1:sample_step:n_individual
            sampled_grads = metrics.individual_gradients_ρ[sampled_indices]
            sampled_x = sampled_indices
        else
            sampled_grads = metrics.individual_gradients_ρ
            sampled_x = 1:length(sampled_grads)
        end
        scatter!(p3, sampled_x, sampled_grads, alpha=0.3, markersize=1, 
                label="Individual", color=:lightblue)
    end
    if !isempty(metrics.batch_gradients_ρ)
        plot!(p3, 1:length(metrics.batch_gradients_ρ), metrics.batch_gradients_ρ, 
              linewidth=2, label="Batch Avg", color=:orange)
    end
    
    p4 = plot(xlabel="Index", ylabel="Gradient Value", title="β Gradients")
    if !isempty(metrics.individual_gradients_β)
        n_individual = length(metrics.individual_gradients_β)
        max_individual = 1000
        if n_individual > max_individual
            sample_step = max(1, n_individual ÷ max_individual)
            sampled_indices = 1:sample_step:n_individual
            sampled_grads = metrics.individual_gradients_β[sampled_indices]
            sampled_x = sampled_indices
        else
            sampled_grads = metrics.individual_gradients_β
            sampled_x = 1:length(sampled_grads)
        end
        scatter!(p4, sampled_x, sampled_grads, alpha=0.3, markersize=1, 
                label="Individual", color=:lightblue)
    end
    if !isempty(metrics.batch_gradients_β)
        plot!(p4, 1:length(metrics.batch_gradients_β), metrics.batch_gradients_β, 
              linewidth=2, label="Batch Avg", color=:orange)
    end
    
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
                   plot_title="Gradient Chaos Analysis: Individual vs Averaged Information")
    savefig(combined, "gradient_chaos_summary.png")
    
    println("\n🎉 All visualizations created!")
    println("📁 Files generated:")
    println("  • loss_evolution.png - Loss evolution showing individual windows vs averages")
    println("  • gradient_evolution_*.png - Gradient evolution for each parameter") 
    println("  • gradient_scatter.png - Scatter plots showing chaotic individual gradients")
    println("  • gradient_chaos_*.gif - Animated demonstrations of chaos vs averaging")
    println("  • gradient_chaos_summary.png - Combined summary visualization")
    
else
    println("⚠️  No gradient data collected. Check training implementation.")
end

println("\n" * "=" ^ 70)
println("🌟 Demonstration complete!")
println("\nKey insights demonstrated:")
println("  1. 🌪️  Individual gradients are highly variable due to chaos")
println("  2. 📊 Batch averages contain meaningful optimization signal")  
println("  3. 📈 Epoch averages provide stable parameter updates")
println("  4. 🎯 Despite chaos, averaging enables successful parameter estimation")