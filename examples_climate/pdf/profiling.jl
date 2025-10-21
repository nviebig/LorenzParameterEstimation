using LorenzParameterEstimation
using Plots, Random, Statistics, LinearAlgebra
using StatsBase, PlotlyJS, Revise, Profile, ProfileView, ProfileCanvas, Optimisers

params_classic = classic_params()
init_params = L63Parameters(σ=10.0, ρ=25.0, β=8/3)

T_coarse   = 10.0
M_coarse   = 400
u0_coarse  = Float64[1, 1, 25]
dt_coarse  = T_coarse / M_coarse
tspan_coarse = (0.0, T_coarse)

# Target trajectory (classic Lorenz)
base_params_test = classic_params()
trajectory_target = integrate(base_params_test, u0_coarse, tspan_coarse, dt_coarse)
trajectory_guess  = integrate(init_params,       u0_coarse, tspan_coarse, dt_coarse)

# Build target PDF on a moderate grid and KEEP the same h for training
xs = trajectory_target.u[:, 1]; ys = trajectory_target.u[:, 2]; zs = trajectory_target.u[:, 3]

nbins = 64  # 64 is heavy; 16–32 is fine to start
xs = trajectory_target.u[:,1]; ys = trajectory_target.u[:,2]; zs = trajectory_target.u[:,3]
hw = maximum(abs, vcat(xs, ys, zs)) * 1.50

pdf_cfg = make_pdf_config(Float64; nbins=nbins, range_halfwidth=hw, loss_mode=:kl)
centers = pdf_cfg.centers
Δ = (centers[end]-centers[1])/(length(centers)-1)
h = 0.5*Δ 
pdf_cfg = make_pdf_config(Float64; nbins=nbins, range_halfwidth=hw, loss_mode=:kl, bandwidth=h)


p3d = zeros(Float64, nbins^3)
R = max(1, ceil(Int, 3h/Δ))
soft_hist_3d_local(xs, ys, zs, centers, h, p3d; R=R)
p3d ./= sum(p3d)
target_pdf = (centers = centers, p3d = p3d)

# Multiple ICs near [1,1,25]
rng_test = MersenneTwister(42)
B_test   = 10
ics_test = [[1.0, 1.0, 25.0] .+ 0.20 .* randn(rng_test, 3) for _ in 1:B_test]
U0_test  = hcat(ics_test...)

# Constant-bandwidth schedule that EXACTLY matches pdf_cfg.h
sched = PdfSchedule{Float64}(h/Δ, h/Δ, 1.0)  # fixed bandwidth = h


cfg_test = ClimateConfig(
    dt = dt_coarse,
    steps = M_coarse * 3,
    samples_per_epoch = round(Int, 0.5*M_coarse),
    initial_conditions = U0_test,
    pdf = pdf_cfg,          # has bandwidth=h
    schedule = sched,       # forces h_curr == h
    rng = MersenneTwister(42)
)

# Train (update only ρ). Key tweaks: smaller LR, less frequent resampling, tighter clip.
Profile.@profile begin
    res_test = train_statistics(
        init_params;
        target = target_pdf,
        stats = (:pdf3d,),
        optimizer = Optimisers.Adam(1e-3),
        update_mask = (σ=false, ρ=true, β=false, x_s=false, y_s=false, z_s=false, θ=false),
        epochs = 10_000,
        refresh = 5,
        early_stopping_patience = 1500,
        early_stopping_min_delta = 1e-6,
        rel_delta = 1e-5,
        cfg = cfg_test,
        verbose = true,
        gradient_verbose = false,
        gradient_clip_norm = 5.0,
    )
end


println("""
Parameter recovery summary:
  Initial guess:      $(init_params)
  Recovered params:   $(res_test.params)
  Target (goal):      $(base_params_test)
""")

