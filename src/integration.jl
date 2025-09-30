# ================================ Integration ================================

"""
    lorenz_rhs(u, params::L63Parameters)

Compute the right-hand side of the Lorenz-63 system.

# Arguments
- `u`: State vector [x, y, z]
- `params`: System parameters

# Returns
- `du`: Time derivatives [dx/dt, dy/dt, dz/dt]
"""
function lorenz_rhs(u, params::L63Parameters{T}) where {T}
    x, y, z = u[1], u[2], u[3]
    dx = params.σ * (y - x)
    dy = x * (params.ρ - z) - y 
    dz = x * y - params.β * z
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
function rk4_step(u::Vector{T}, params::L63Parameters{T}, dt::T) where {T}
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
    return [
        x1 + c * (k1x + 2*k2x + 2*k3x + k4x),
        x2 + c * (k1y + 2*k2y + 2*k3y + k4y), 
        x3 + c * (k1z + 2*k2z + 2*k3z + k4z)
    ]
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
function integrate(system::L63System{T}; dense_output::Bool=true) where {T}
    
    # Time setup
    t0, tf = system.tspan
    dt = system.dt
    n_steps = Int(ceil((tf - t0) / dt))
    actual_tf = t0 + n_steps * dt
    
    # Storage setup
    if dense_output
        times = Vector{T}(undef, n_steps + 1)
        trajectory = Matrix{T}(undef, n_steps + 1, 3)
        times[1] = t0
        trajectory[1, :] .= system.u0
    end
    
    # Integration
    u = copy(system.u0)
    
    for i in 1:n_steps
        u = rk4_step(u, system.params, dt)
        
        if dense_output
            times[i + 1] = t0 + i * dt
            trajectory[i + 1, :] .= u
        end
    end
    
    if dense_output
        return L63Solution{T}(times, trajectory, system)
    else
        # Return only final state
        final_time = [actual_tf]
        final_traj = reshape(u, 1, 3)
        return L63Solution{T}(final_time, final_traj, system)
    end
end

"""
    integrate(params::L63Parameters, u0::Vector, tspan::Tuple, dt::Real)

Convenience method for quick integration.
"""
function integrate(params::L63Parameters, u0::Vector, tspan::Tuple, dt::Real)
    system = L63System(params=params, u0=u0, tspan=tspan, dt=dt)
    return integrate(system)
end