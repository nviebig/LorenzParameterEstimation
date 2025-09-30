# ================================ Optimizer Configurations ================================

using Optimisers
using Random

"""
    OptimizerConfig

Container for optimizer settings with easy presets and customization.
"""
struct OptimizerConfig{T<:Real}
    optimizer::Any
    learning_rate::T
    gradient_clip_norm::Union{Nothing, T}
    name::String
    
    function OptimizerConfig(optimizer, learning_rate::Real, gradient_clip_norm::Union{Nothing, Real} = nothing; name::String = "Custom")
        T = typeof(learning_rate)
        clip_norm = isnothing(gradient_clip_norm) ? nothing : T(gradient_clip_norm)
        new{T}(optimizer, T(learning_rate), clip_norm, name)
    end
end

"""
    build_optimizer(config::OptimizerConfig)

Build the actual optimizer chain with gradient clipping if specified.
"""
function build_optimizer(config::OptimizerConfig)
    if isnothing(config.gradient_clip_norm)
        return config.optimizer
    else
        return Optimisers.OptimiserChain(
            Optimisers.ClipNorm(config.gradient_clip_norm),
            config.optimizer
        )
    end
end

# ================================ Preset Optimizers ================================

"""
    sgd_config(; learning_rate=1e-2, momentum=0.9, gradient_clip_norm=1.0)

Stochastic Gradient Descent with momentum.
"""
function sgd_config(; learning_rate::Real = 1e-2, momentum::Real = 0.9, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    optimizer = Optimisers.Momentum(learning_rate, momentum)
    return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "SGD")
end

"""
    adam_config(; learning_rate=1e-3, β1=0.9, β2=0.999, gradient_clip_norm=1.0)

Adam optimizer - good default choice.
"""
function adam_config(; learning_rate::Real = 1e-3, β1::Real = 0.9, β2::Real = 0.999, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    optimizer = Optimisers.Adam(learning_rate, (β1, β2))
    return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "Adam")
end

"""
    adamw_config(; learning_rate=1e-3, β1=0.9, β2=0.999, weight_decay=1e-2, gradient_clip_norm=1.0)

AdamW optimizer with weight decay - good for regularization.
"""
function adamw_config(; learning_rate::Real = 1e-3, β1::Real = 0.9, β2::Real = 0.999, 
                      weight_decay::Real = 1e-2, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    optimizer = Optimisers.AdamW(learning_rate, (β1, β2), weight_decay)
    return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "AdamW")
end

"""
    rmsprop_config(; learning_rate=1e-3, ρ=0.9, gradient_clip_norm=1.0)

RMSprop optimizer - good for recurrent-like problems.
"""
function rmsprop_config(; learning_rate::Real = 1e-3, ρ::Real = 0.9, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    optimizer = Optimisers.RMSprop(learning_rate, ρ)
    return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "RMSprop")
end

"""
    adagrad_config(; learning_rate=1e-2, gradient_clip_norm=1.0)

Adagrad optimizer - adaptive learning rates.
"""
function adagrad_config(; learning_rate::Real = 1e-2, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    optimizer = Optimisers.AdaGrad(learning_rate)
    return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "Adagrad")
end

"""
    lion_config(; learning_rate=1e-4, β1=0.9, β2=0.99, weight_decay=1e-2, gradient_clip_norm=1.0)

Lion optimizer - newer, memory efficient alternative to Adam.
"""
function lion_config(; learning_rate::Real = 1e-4, β1::Real = 0.9, β2::Real = 0.99,
                     weight_decay::Real = 1e-2, gradient_clip_norm::Union{Nothing, Real} = 1.0)
    # Note: Lion might not be available in all Optimisers.jl versions
    # Fallback to AdamW if Lion is not available
    try
        optimizer = Optimisers.Lion(learning_rate, (β1, β2), weight_decay)
        return OptimizerConfig(optimizer, learning_rate, gradient_clip_norm, name = "Lion")
    catch
        @warn "Lion optimizer not available, falling back to AdamW"
        return adamw_config(learning_rate=learning_rate, β1=β1, β2=β2, 
                           weight_decay=weight_decay, gradient_clip_norm=gradient_clip_norm)
    end
end

# ================================ Scheduler Support ================================

"""
    SchedulerConfig

Configuration for learning rate scheduling.
"""
struct SchedulerConfig{T<:Real}
    schedule_type::Symbol
    initial_lr::T
    schedule_params::NamedTuple
    
    function SchedulerConfig(schedule_type::Symbol, initial_lr::Real; kwargs...)
        T = typeof(initial_lr)
        new{T}(schedule_type, T(initial_lr), values(kwargs))
    end
end

"""
    exponential_decay_schedule(; decay_rate=0.96, decay_steps=100)

Exponential decay learning rate schedule.
"""
function exponential_decay_schedule(initial_lr::Real; decay_rate::Real = 0.96, decay_steps::Int = 100)
    return SchedulerConfig(:exponential, initial_lr, decay_rate=decay_rate, decay_steps=decay_steps)
end

"""
    cosine_annealing_schedule(; T_max=100, eta_min=1e-6)

Cosine annealing learning rate schedule.
"""
function cosine_annealing_schedule(initial_lr::Real; T_max::Int = 100, eta_min::Real = 1e-6)
    return SchedulerConfig(:cosine, initial_lr, T_max=T_max, eta_min=eta_min)
end

"""
    step_decay_schedule(; step_size=30, gamma=0.1)

Step decay learning rate schedule.
"""
function step_decay_schedule(initial_lr::Real; step_size::Int = 30, gamma::Real = 0.1)
    return SchedulerConfig(:step, initial_lr, step_size=step_size, gamma=gamma)
end

"""
    apply_schedule(schedule::SchedulerConfig, epoch::Int)

Apply the learning rate schedule for a given epoch.
"""
function apply_schedule(schedule::SchedulerConfig, epoch::Int)
    if schedule.schedule_type == :exponential
        decay_rate = schedule.schedule_params.decay_rate
        decay_steps = schedule.schedule_params.decay_steps
        return schedule.initial_lr * decay_rate^(epoch / decay_steps)
    elseif schedule.schedule_type == :cosine
        T_max = schedule.schedule_params.T_max
        eta_min = schedule.schedule_params.eta_min
        return eta_min + (schedule.initial_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    elseif schedule.schedule_type == :step
        step_size = schedule.schedule_params.step_size
        gamma = schedule.schedule_params.gamma
        return schedule.initial_lr * gamma^(epoch ÷ step_size)
    else
        return schedule.initial_lr
    end
end

# ================================ Quick Presets ================================

"""
    robust_optimizer()

Robust optimizer for stable training.
"""
robust_optimizer() = adamw_config(learning_rate=3e-4, weight_decay=1e-3, gradient_clip_norm=0.5)

"""
    fast_optimizer()

Fast optimizer for quick experimentation.
"""
fast_optimizer() = adam_config(learning_rate=1e-3, gradient_clip_norm=1.0)

"""
    conservative_optimizer()

Conservative optimizer for research/fine-tuning.
"""
conservative_optimizer() = adamw_config(learning_rate=1e-4, weight_decay=1e-2, gradient_clip_norm=0.1)

# Export everything
export OptimizerConfig, build_optimizer
export sgd_config, adam_config, adamw_config, rmsprop_config, adagrad_config, lion_config
export SchedulerConfig, exponential_decay_schedule, cosine_annealing_schedule, step_decay_schedule, apply_schedule
export robust_optimizer, fast_optimizer, conservative_optimizer