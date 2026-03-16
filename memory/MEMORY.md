# LorenzParameterEstimation — Project Memory

## Overview
Master thesis project on parameter estimation for the Lorenz-63 system using two paradigms:
- **Weather approach**: trajectory matching (windowed mini-batch SGD, Enzyme AD)
- **Climate approach**: statistical invariant matching (mean, 3D PDF, TLPP, Wasserstein)

## Key Paths
- Root: `/Users/niklasviebig/master_thesis/LorenzParameterEstimation/`
- Weather experiments: `examples_weather/`
- Climate experiments: `examples_climate/`
- Full experiment overview: `EXPERIMENTS_OVERVIEW.md` (created 2026-03-03)
- Julia package: `src/` (LorenzParameterEstimation.jl)

## Package API (stable)
- `L63Parameters(σ, ρ, β, x_s, y_s, z_s, θ)` — full parameter struct
- `integrate(params, u0, tspan, dt)` → L63Solution (.t, .u, .final_state)
- `modular_train!(params, sol; optimizer_config, loss_function, window_size, stride, batch_size, update_σ, update_ρ, update_β, update_x_s, update_y_s, update_z_s, update_θ, ...)` — mini-batch SGD weather training
- `train_statistics(params; target, stats, cfg, optimizer, update_mask, ...)` — climate training
- `ClimateConfig(dt, steps, samples_per_epoch, initial_conditions, pdf, schedule, rng, batch_size)`
- `TrainingMetrics{T}` — tracks per-window/batch/epoch gradients
- `soft_hist_3d_local` — differentiable 3D KDE (separable Gaussian kernel)
- `adam_config`, `adamw_config`, `sgd_config`, `adagrad_config` — optimizer helpers
- `classic_params()`, `stable_params()`, `with_coordinate_shifts()`, `with_theta()`

## Key Experimental Results
- Short windows (50–100 steps = 0.28–0.56 Lyapunov times) are optimal for weather approach
- Gradient variance scales strongly with window size (50→200 gives ~19× higher std)
- Batch size 8–16 is optimal; larger batches don't help and can hurt for small windows
- Custom Adam + ClipNorm(1.0) outperforms plain Adam/SGD/AdaGrad in all tests
- Full-trajectory NLL without windowing fails (gets stuck)
- Climate mean matching works for coordinate shifts; PDF matching works for ρ
- TLPP approach needs >100 epochs; gradient always clipped → need lower clip threshold

## Key System Constants
- Lyapunov time τ_λ ≈ 0.9 time units = 180 timesteps (at dt=0.005)
- Standard integration: T=100, M=20000, dt=0.005, u0=[1,1,1]
- Classic chaos: σ=10, ρ=28, β=8/3

## Known Issues / Gotchas
- `PdfConfig` constructor must use keyword args: `PdfConfig{Float64}(centers=..., bandwidth=..., loss_mode=..., ...)`
- `soft_hist_3d_local` is a package function — cannot redefine in global scope (import explicitly)
- `Enzyme.set_runtime_activity(Enzyme.Reverse)` required for chaotic systems
- Gradient norms at true parameters = 0 (correct — flat at minimum)
