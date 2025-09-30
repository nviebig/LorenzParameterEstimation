# ================================ Bayesian Inference ================================

using Turing
using Distributions
using Random
using LinearAlgebra
using Statistics: mean, std

"""
    @model lorenz_bayesian_model(data, window_size, dt, prior_σ, prior_ρ, prior_β)

Bayesian model for Lorenz-63 parameter estimation using Turing.jl.

# Arguments
- `data`: Observed trajectory data (Matrix)
- `window_size`: Length of prediction windows
- `dt`: Integration time step
- `prior_σ`, `prior_ρ`, `prior_β`: Prior distributions for parameters

# Example
```julia
# Define priors
prior_σ = Normal(10.0, 5.0)
prior_ρ = Normal(28.0, 10.0) 
prior_β = Normal(8/3, 2.0)

# Create model
model = lorenz_bayesian_model(data, 100, 0.01, prior_σ, prior_ρ, prior_β)

# Sample
chain = sample(model, NUTS(), 1000)
```
"""
@model function lorenz_bayesian_model(data, window_size, dt, prior_σ, prior_ρ, prior_β, noise_std_prior=InverseGamma(2, 3))
    # Prior distributions for parameters
    σ ~ prior_σ
    ρ ~ prior_ρ  
    β ~ prior_β
    
    # Prior for observation noise
    noise_std ~ noise_std_prior
    
    # Create parameter object
    params = L63Parameters(σ, ρ, β)
    
    # Likelihood: compute predicted windows and compare to observed
    n_obs, n_dims = size(data)
    n_windows = n_obs - window_size
    
    if n_windows > 0
        for i in 1:min(n_windows, 50)  # Limit to avoid excessive computation
            u0 = data[i, :]
            
            # Simulate forward window_size steps
            predicted = _simulate_window_turing(params, u0, window_size, dt)
            observed = data[i+1:i+window_size, :]
            
            # Likelihood: observed data given predicted trajectory
            for j in 1:window_size, k in 1:n_dims
                observed[j, k] ~ Normal(predicted[j, k], noise_std)
            end
        end
    end
end

"""
    _simulate_window_turing(params, u0, window_size, dt)

Helper function for simulating Lorenz trajectory in Turing model.
Must be compatible with automatic differentiation.
"""
function _simulate_window_turing(params::L63Parameters, u0::Vector, window_size::Int, dt::Real)
    trajectory = Matrix{eltype(u0)}(undef, window_size, 3)
    state = copy(u0)
    
    for i in 1:window_size
        state = rk4_step(state, params, dt)
        trajectory[i, :] = state
    end
    
    return trajectory
end

"""
    bayesian_parameter_estimation(target_solution; 
                                 n_samples=1000, 
                                 n_chains=4,
                                 window_size=100,
                                 subsample_factor=10,
                                 kwargs...)

Perform Bayesian parameter estimation using MCMC.

# Arguments
- `target_solution::L63Solution`: Target trajectory
- `n_samples::Int=1000`: Number of MCMC samples per chain
- `n_chains::Int=4`: Number of parallel chains
- `window_size::Int=100`: Size of trajectory windows for likelihood
- `subsample_factor::Int=10`: Use every Nth data point to reduce computation

# Keyword Arguments
- `prior_σ`: Prior distribution for σ (default: Normal(10, 5))
- `prior_ρ`: Prior distribution for ρ (default: Normal(28, 10))
- `prior_β`: Prior distribution for β (default: Normal(8/3, 2))
- `noise_std_prior`: Prior for observation noise (default: InverseGamma(2, 3))
- `sampler`: MCMC sampler (default: NUTS())
- `progress::Bool=true`: Show sampling progress

# Returns
- `chains`: MCMCChains object with posterior samples
- `summary`: Posterior summary statistics
"""
function bayesian_parameter_estimation(
    target_solution::L63Solution{T};
    n_samples::Int = 1000,
    n_chains::Int = 4,
    window_size::Int = 100,
    subsample_factor::Int = 10,
    # Priors
    prior_σ = Normal(10.0, 5.0),
    prior_ρ = Normal(28.0, 10.0),
    prior_β = Normal(8.0/3.0, 2.0),
    noise_std_prior = InverseGamma(2, 3),
    # Sampling
    sampler = NUTS(),
    progress::Bool = true,
    kwargs...
) where {T}
    
    # Subsample data to reduce computational cost
    data_indices = 1:subsample_factor:length(target_solution)
    data = target_solution.u[data_indices, :]
    dt_effective = target_solution.system.dt * subsample_factor
    
    # Create Bayesian model
    model = lorenz_bayesian_model(data, window_size, dt_effective, 
                                 prior_σ, prior_ρ, prior_β, noise_std_prior)
    
    println("🔬 Starting Bayesian parameter estimation")
    println("   Data points: $(size(data, 1))")
    println("   Window size: $window_size")
    println("   Chains: $n_chains × $n_samples samples")
    println()
    
    # Sample from posterior
    chains = sample(model, sampler, MCMCThreads(), n_samples, n_chains; 
                   progress = progress, kwargs...)
    
    # Compute summary statistics
    summary_stats = summarystats(chains)
    
    # Extract parameter estimates
    σ_est = mean(chains[:σ])
    ρ_est = mean(chains[:ρ])
    β_est = mean(chains[:β])
    noise_est = mean(chains[:noise_std])
    
    estimated_params = L63Parameters(σ_est, ρ_est, β_est)
    
    println("✅ Bayesian estimation completed")
    println("   Parameter estimates (posterior means):")
    println("   σ = $(round(σ_est, digits=4)) ± $(round(std(chains[:σ]), digits=4))")
    println("   ρ = $(round(ρ_est, digits=4)) ± $(round(std(chains[:ρ]), digits=4))")
    println("   β = $(round(β_est, digits=4)) ± $(round(std(chains[:β]), digits=4))")
    println("   noise_std = $(round(noise_est, digits=4)) ± $(round(std(chains[:noise_std]), digits=4))")
    println()
    
    return (
        chains = chains,
        summary = summary_stats,
        estimated_params = estimated_params,
        model = model
    )
end

"""
    variational_inference(target_solution; kwargs...)

Perform variational inference for faster approximate Bayesian estimation.

# Arguments  
- `target_solution::L63Solution`: Target trajectory
- `n_samples::Int=1000`: Number of samples from variational posterior

# Returns
- `q`: Variational posterior distribution
- `estimated_params`: Point estimates
- `elbo_trace`: Evidence lower bound trace
"""
function variational_inference(
    target_solution::L63Solution{T};
    n_samples::Int = 1000,
    window_size::Int = 100,
    subsample_factor::Int = 10,
    # Priors (same as above)
    prior_σ = Normal(10.0, 5.0),
    prior_ρ = Normal(28.0, 10.0), 
    prior_β = Normal(8.0/3.0, 2.0),
    noise_std_prior = InverseGamma(2, 3),
    # VI specific
    optimizer = Optimisers.Adam(0.01),
    n_iterations::Int = 2000,
    kwargs...
) where {T}
    
    # Subsample data
    data_indices = 1:subsample_factor:length(target_solution)
    data = target_solution.u[data_indices, :]
    dt_effective = target_solution.system.dt * subsample_factor
    
    # Create model
    model = lorenz_bayesian_model(data, window_size, dt_effective,
                                 prior_σ, prior_ρ, prior_β, noise_std_prior)
    
    println("⚡ Starting variational inference")
    println("   Iterations: $n_iterations")
    
    # Perform VI using ADVI
    q, elbo_trace = vi(model, ADVI(10, n_iterations), optimizer)
    
    # Sample from variational posterior
    posterior_samples = rand(q, n_samples)
    
    # Extract parameter estimates
    σ_samples = posterior_samples[1, :]
    ρ_samples = posterior_samples[2, :]
    β_samples = posterior_samples[3, :]
    
    estimated_params = L63Parameters(
        mean(σ_samples),
        mean(ρ_samples), 
        mean(β_samples)
    )
    
    println("✅ Variational inference completed")
    println("   Parameter estimates:")
    println("   σ = $(round(mean(σ_samples), digits=4)) ± $(round(std(σ_samples), digits=4))")
    println("   ρ = $(round(mean(ρ_samples), digits=4)) ± $(round(std(ρ_samples), digits=4))")
    println("   β = $(round(mean(β_samples), digits=4)) ± $(round(std(β_samples), digits=4))")
    println()
    
    return (
        q = q,
        estimated_params = estimated_params,
        elbo_trace = elbo_trace,
        posterior_samples = posterior_samples
    )
end

"""
    posterior_predictive_check(chains, target_solution; n_samples=100)

Perform posterior predictive checks to validate the Bayesian model.

# Arguments
- `chains`: MCMC chains from bayesian_parameter_estimation
- `target_solution`: Original target trajectory
- `n_samples::Int=100`: Number of posterior samples to use

# Returns
- `predicted_trajectories`: Array of predicted trajectories
- `prediction_summary`: Summary statistics of predictions
"""
function posterior_predictive_check(
    chains,
    target_solution::L63Solution{T};
    n_samples::Int = 100,
    prediction_length::Int = 500
) where {T}
    
    # Extract parameter samples from chains
    n_chain_samples = length(chains[:σ])
    sample_indices = rand(1:n_chain_samples, n_samples)
    
    predicted_trajectories = Vector{Matrix{T}}(undef, n_samples)
    
    println("🔮 Performing posterior predictive checks")
    println("   Using $n_samples posterior samples")
    
    for (i, idx) in enumerate(sample_indices)
        # Extract parameters
        σ_sample = chains[:σ][idx]
        ρ_sample = chains[:ρ][idx]
        β_sample = chains[:β ][idx]
        
        params_sample = L63Parameters(σ_sample, ρ_sample, β_sample)
        
        # Generate prediction
        u0 = target_solution.system.u0
        tspan = (0.0, T(prediction_length * target_solution.system.dt))
        
        try
            sol = integrate(params_sample, u0, tspan, target_solution.system.dt)
            predicted_trajectories[i] = sol.u
        catch
            # If integration fails, use NaN trajectory
            predicted_trajectories[i] = fill(T(NaN), prediction_length, 3)
        end
    end
    
    # Compute prediction summary
    valid_predictions = filter(traj -> !any(isnan, traj), predicted_trajectories)
    
    if !isempty(valid_predictions)
        # Stack valid predictions
        stacked = cat(valid_predictions..., dims=3)
        
        prediction_mean = mean(stacked, dims=3)[:, :, 1]
        prediction_std = std(stacked, dims=3)[:, :, 1]
        
        prediction_summary = (
            mean = prediction_mean,
            std = prediction_std,
            n_valid = length(valid_predictions),
            n_total = n_samples
        )
    else
        prediction_summary = nothing
    end
    
    println("✅ Posterior predictive check completed")
    println("   Valid predictions: $(length(valid_predictions))/$n_samples")
    
    return (
        predicted_trajectories = predicted_trajectories,
        prediction_summary = prediction_summary
    )
end

# Export functions
export lorenz_bayesian_model, bayesian_parameter_estimation, variational_inference
export posterior_predictive_check