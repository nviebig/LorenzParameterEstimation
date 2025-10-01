# ================================ Integration ================================

"""
    lorenz_rhs(u, params::L63Parameters)

Compute the right-hand side of the extended Lorenz-63 system with coordinate shifts and theta parameter.

# Arguments
- `u`: State vector [x, y, z]
- `params`: System parameters (including σ, ρ, β, x_s, y_s, z_s, θ)

# Returns
- `du`: Time derivatives [dx/dt, dy/dt, dz/dt]

# Extended System
- dx/dt = σ((y - y_s) - (x - x_s))
- dy/dt = θ(x - x_s)(ρ - (z - z_s)) - (y - y_s)  
- dz/dt = (x - x_s)(y - y_s) - β(z - z_s)
"""
function lorenz_rhs(u, params::L63Parameters{T}) where {T}
    x, y, z = u[1], u[2], u[3]
    
    # Apply coordinate shifts
    x_shifted = x - params.x_s
    y_shifted = y - params.y_s
    z_shifted = z - params.z_s
    
    # Extended Lorenz equations with theta parameter
    dx = params.σ * (y_shifted - x_shifted)
    dy = params.θ * x_shifted * (params.ρ - z_shifted) - y_shifted
    dz = x_shifted * y_shifted - params.β * z_shifted
    
    return (dx, dy, dz)
end

"""
    rk4_step(u, params::L63Parameters, dt)

Single Runge-Kutta 4th order step for the Lorenz-63 system.

# Arguments
- `u`: Current state vector
- `params`: System parameters
- `dt`: Time step size

# Returns
- New state vector after one RK4 step
"""
function rk4_step(u::AbstractVector{T}, params::L63Parameters{T}, dt::T) where {T}
    # RK4 for Lorenz system - optimized for Enzyme compatibility
    x1, x2, x3 = u[1], u[2], u[3]
    
    # Stage 1
    k1x, k1y, k1z = lorenz_rhs((x1, x2, x3), params)
    
    # Stage 2
    k2x, k2y, k2z = lorenz_rhs((x1 + 0.5*dt*k1x, x2 + 0.5*dt*k1y, x3 + 0.5*dt*k1z), params)
    
    # Stage 3
    k3x, k3y, k3z = lorenz_rhs((x1 + 0.5*dt*k2x, x2 + 0.5*dt*k2y, x3 + 0.5*dt*k2z), params)
    
    # Stage 4
    k4x, k4y, k4z = lorenz_rhs((x1 + dt*k3x, x2 + dt*k3y, x3 + dt*k3z), params)
    
    # Final update
    c = dt / 6
    result = similar(u)
    result[1] = x1 + c * (k1x + 2*k2x + 2*k3x + k4x)
    result[2] = x2 + c * (k1y + 2*k2y + 2*k3y + k4y)
    result[3] = x3 + c * (k1z + 2*k2z + 2*k3z + k4z)
    return result
end

"""
    integrate(system::L63System; dense_output=false)

Integrate a Lorenz-63 system using RK4.

# Arguments
- `system::L63System`: System specification
- `dense_output::Bool`: If true, store all intermediate steps

# Returns
- `L63Solution`: Complete solution object
"""
function integrate(system::L63System{T,V}; dense_output::Bool=true) where {T,V}
    
    # Time setup
    t0, tf = system.tspan
    dt = system.dt
    n_steps = Int(ceil((tf - t0) / dt))
    actual_tf = t0 + n_steps * dt
    
    # Storage setup
    if dense_output
        times = similar_array(system.u0, T, n_steps + 1)
        trajectory = similar_array(system.u0, T, n_steps + 1, 3)
        times[1] = t0
        trajectory[1, :] .= system.u0
    end
    
    # Integration
    u = similar(system.u0)
    u .= system.u0
    
    for i in 1:n_steps
        u = rk4_step(u, system.params, dt)
        
        if dense_output
            times[i + 1] = t0 + i * dt
            trajectory[i + 1, :] .= u
        end
    end
    
    if dense_output
        return L63Solution(times, trajectory, system)
    else
        # Return only final state
        final_time = similar_array(system.u0, T, 1)
        final_time[1] = actual_tf
        final_traj = similar_array(system.u0, T, 1, 3)
        final_traj[1, :] .= u
        return L63Solution(final_time, final_traj, system)
    end
end

"""
    integrate(params::L63Parameters, u0::Vector, tspan::Tuple, dt::Real)

Convenience method for quick integration.
"""
function integrate(params::L63Parameters, u0::AbstractVector, tspan::Tuple, dt::Real)
    system = L63System(params=params, u0=u0, tspan=tspan, dt=dt)
    return integrate(system)
end