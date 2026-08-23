# Experiment 02 — Amplitude preservation vs scheme and CFL.
#
# Goal: isolate how much the sinusoid amplitude is destroyed by the
# discretisation alone. Three truth-only runs (no PF, no noise) all start
# from the same initial state β = β_prior exactly, then evolve under:
#
#   A) Upwind, default CFL  (n_integration_step = 10, Δt = 360 s, CFL ≈ 0.09)
#   B) Upwind, CFL ≈ 1      (n_integration_step = 1,  Δt = 3600 s, CFL ≈ 0.9)
#   C) Lax–Wendroff         (n_integration_step = 10, Δt = 360 s, CFL ≈ 0.09)
#
# Amplitude is measured as |F(3, 3)| · 4 / (nx·ny) — the magnitude of the
# 2D-Fourier coefficient at the (kₓ=3, k_y=3) mode of (β − 2000). This is
# translation-invariant, so an advected sinusoid registers a constant
# amplitude as long as no diffusion is present.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_02_cfl_and_laxwendroff.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots

const FIG_NAME = "amplitude_vs_time_exp02.png"
const NOTES_BASE = "truth-only (σ_init=σ_proc=0); amplitude at Fourier (3,3) mode."

# ── Helper: track sinusoid-amplitude of the truth over T steps ───────────
# Manual single-mode DFT — no FFTW dependency needed for one Fourier
# coefficient. Cost is O(nx·ny) per call, negligible.
function amplitude_33(state::AbstractVector, nx::Int, ny::Int)
    F = 0.0 + 0.0im
    @inbounds for j in 1:ny, i in 1:nx
        phase = -2π * (3*(i-1)/nx + 3*(j-1)/ny)
        F += (state[(i-1)*ny + j] - 2000.0) * complex(cos(phase), sin(phase))
    end
    return 4 * abs(F) / (nx * ny)
end

function truth_amplitude_series(params::Dict; T::Int=100, SEED::Int=123)
    Glacier._CFL_WARNED[] = false
    model = Glacier.init(Dict("glacier" => params))
    nx, ny = model.parameters.nx, model.parameters.ny
    rng = MersenneTwister(SEED)
    s = zeros(nx * ny)
    ParticleDA.sample_initial_state!(s, model, rng)   # σ_init=0 → s = β_prior

    amps = zeros(T + 1)
    amps[1] = amplitude_33(s, nx, ny)
    for t in 1:T
        ParticleDA.update_state_deterministic!(s, model, t)
        ParticleDA.update_state_stochastic!(s, model, rng)
        amps[t + 1] = amplitude_33(s, nx, ny)
    end
    times_h = collect(0:T) .* params["time_step"] ./ 3600.0
    return (times_h = times_h, amps = amps)
end

# ── Shared baseline — zero noise so amplitude decay isolates the scheme ──
base = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 0.0,           # no initial perturbation
    "process_std_beta" => 0.0,        # no process noise
    "obs_noise_std" => 0.10,          # unused (no PF)
    "advection_epsilon" => 0.0,       # strictly linear (v = 1, no β-feedback)
    "min_beta" => 0.0,
    "noise_length_scale" => 15_000.0, # unused with σ=0 but kept for log
)
const T_STEPS = 100
dx = base["x_length"] / base["nx"]

# A: upwind, default CFL
a_params = merge(base, Dict{String, Any}(
    "n_integration_step" => 10, "time_step" => 3600.0,
    "advection_type" => "linear"))
# B: upwind, CFL ≈ 1
b_params = merge(base, Dict{String, Any}(
    "n_integration_step" => 1,  "time_step" => 3600.0,
    "advection_type" => "linear"))
# C: Lax–Wendroff at default CFL
c_params = merge(base, Dict{String, Any}(
    "n_integration_step" => 10, "time_step" => 3600.0,
    "advection_type" => "lax_wendroff"))

cfl_of(p) = 1.0 * (p["time_step"] / p["n_integration_step"]) / dx

println("=== Experiment 02: Amplitude vs scheme/CFL ===")
println("Running A (upwind, CFL=$(round(cfl_of(a_params), digits=2))) …")
a = truth_amplitude_series(a_params; T=T_STEPS)
println("Running B (upwind, CFL=$(round(cfl_of(b_params), digits=2))) …")
b = truth_amplitude_series(b_params; T=T_STEPS)
println("Running C (Lax–Wendroff, CFL=$(round(cfl_of(c_params), digits=2))) …")
c = truth_amplitude_series(c_params; T=T_STEPS)

# ── Summary table ────────────────────────────────────────────────────────
loss_pct(s) = 100 * (s.amps[1] - s.amps[end]) / s.amps[1]
println("\n=== Summary ===")
@printf "%-25s  %-8s  %-12s  %-12s  %-12s\n" "Scheme" "CFL" "A_initial" "A_final" "% loss"
@printf "%-25s  %-8.3f  %-12.1f  %-12.1f  %-12.2f\n" "Upwind (default)" cfl_of(a_params) a.amps[1] a.amps[end] loss_pct(a)
@printf "%-25s  %-8.3f  %-12.1f  %-12.1f  %-12.2f\n" "Upwind (CFL≈1)"   cfl_of(b_params) b.amps[1] b.amps[end] loss_pct(b)
@printf "%-25s  %-8.3f  %-12.1f  %-12.1f  %-12.2f\n" "Lax–Wendroff"     cfl_of(c_params) c.amps[1] c.amps[end] loss_pct(c)

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(a.times_h, a.amps;
         xlabel = "time (h)", ylabel = "amplitude at Fourier (k=3,3) mode",
         title  = "Exp 02 — Amplitude decay vs scheme & CFL",
         label  = @sprintf("Upwind, CFL=%.2f", cfl_of(a_params)),
         lw=2.5, color=:steelblue, legend=:topright, ylim=(0, 2200))
plot!(p, b.times_h, b.amps;
      label = @sprintf("Upwind, CFL=%.2f", cfl_of(b_params)),
      lw=2.5, color=:firebrick)
plot!(p, c.times_h, c.amps;
      label = @sprintf("Lax–Wendroff, CFL=%.2f", cfl_of(c_params)),
      lw=2.5, color=:darkgreen)
hline!(p, [a.amps[1]]; linestyle=:dot, color=:gray, label="initial amplitude")
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("\nSaved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Log: one row per series ──────────────────────────────────────────────
common = (;
    figure          = FIG_NAME,
    sigma_init      = 0.0,
    sigma_proc      = 0.0,
    length_scale_km = base["noise_length_scale"] / 1000,
    particles       = 0,
    n_obs           = 0,
    sigma_obs       = 0.0,
    T_steps         = T_STEPS,
)
log_experiment!(; common...,
    model_type = "Upwind (default)",
    obs_interval_s = a_params["time_step"],
    notes = @sprintf("CFL=%.2f, n_integ=%d, A loss %.1f%%. %s",
                     cfl_of(a_params), a_params["n_integration_step"],
                     loss_pct(a), NOTES_BASE))
log_experiment!(; common...,
    model_type = "Upwind (CFL≈1)",
    obs_interval_s = b_params["time_step"],
    notes = @sprintf("CFL=%.2f, n_integ=%d, A loss %.1f%%. %s",
                     cfl_of(b_params), b_params["n_integration_step"],
                     loss_pct(b), NOTES_BASE))
log_experiment!(; common...,
    model_type = "Lax–Wendroff",
    obs_interval_s = c_params["time_step"],
    notes = @sprintf("CFL=%.2f, n_integ=%d, A loss %.1f%%. %s",
                     cfl_of(c_params), c_params["n_integration_step"],
                     loss_pct(c), NOTES_BASE))
println("Logged → $LOG_PATH")
