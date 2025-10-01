# ================================ Optimizer Configurations ================================

using Optimisers

"""
    OptimizerConfig

Simple container for optimizer settings. Only stores the actual optimizer and name for display.
"""
struct OptimizerConfig{T<:Real}
    optimizer::Any
    learning_rate::T  # For reference/display only
    name::String
    
    function OptimizerConfig(optimizer, learning_rate::Real; name::String = "Custom")
        T = typeof(learning_rate)
        new{T}(optimizer, T(learning_rate), name)
    end
end

# ================================ Optimizer Factory Functions ================================

"""
    adam_config(; learning_rate=1e-3, β1=0.9, β2=0.999)

Adam optimizer - good default choice for most problems.
"""
function adam_config(; learning_rate::Real = 1e-3, β1::Real = 0.9, β2::Real = 0.999)
    optimizer = Optimisers.Adam(learning_rate, (β1, β2))
    return OptimizerConfig(optimizer, learning_rate, name = "Adam")
end

"""
    sgd_config(; learning_rate=1e-2, momentum=0.9)

Stochastic Gradient Descent with momentum.
"""
function sgd_config(; learning_rate::Real = 1e-2, momentum::Real = 0.9)
    optimizer = Optimisers.Momentum(learning_rate, momentum)
    return OptimizerConfig(optimizer, learning_rate, name = "SGD")
end

"""
    adamw_config(; learning_rate=1e-3, β1=0.9, β2=0.999, weight_decay=1e-2)

AdamW optimizer with weight decay - good for regularization.
"""
function adamw_config(; learning_rate::Real = 1e-3, β1::Real = 0.9, β2::Real = 0.999, 
                      weight_decay::Real = 1e-2)
    optimizer = Optimisers.AdamW(learning_rate, (β1, β2), weight_decay)
    return OptimizerConfig(optimizer, learning_rate, name = "AdamW")
end

"""
    adagrad_config(; learning_rate=1e-2)

Adagrad optimizer - adaptive learning rates.
"""
function adagrad_config(; learning_rate::Real = 1e-2)
    optimizer = Optimisers.AdaGrad(learning_rate)
    return OptimizerConfig(optimizer, learning_rate, name = "Adagrad")
end

"""
    rmsprop_config(; learning_rate=1e-3, ρ=0.9)

RMSprop optimizer - good for recurrent-like problems.
"""
function rmsprop_config(; learning_rate::Real = 1e-3, ρ::Real = 0.9)
    optimizer = Optimisers.RMSprop(learning_rate, ρ)
    return OptimizerConfig(optimizer, learning_rate, name = "RMSprop")
end

# Export only what's actually used
export OptimizerConfig
export adam_config, sgd_config, adamw_config, adagrad_config, rmsprop_config