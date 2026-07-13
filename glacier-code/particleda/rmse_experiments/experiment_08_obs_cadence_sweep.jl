# Experiment 08 — RMSE: observation cadence sweep on linear advection.
#
# Holds the spatial setup (sensors, prior, σ_obs, σ_init) constant and
# varies how often the filter is given an observation. Cadence values
# match plausible real-world data sources for glacier velocity:
#   - 10 min   in-situ GPS (high-rate)
#   - 30 min   in-situ GPS (moderate)
#   - 1 h      baseline (run10 default)
#   - 3 h      mid-tempo InSAR window
#   - 6 h      mid-tempo InSAR window
#   - 12 h    daily InSAR-style
#
# All runs cover the same physical time (≈ 100 h) by setting T so
# T·time_step ≈ 360_000 s. The RMSE-vs-time plot puts every curve on a
# shared physical-time x-axis.
#
# ── Stability adjustments (also documented in rmse_experiment_log.md) ───
# 1. `n_integration_step` is set per cadence so the internal Δt satisfies
#    dt_inner ≤ 720 s (CFL safety ceiling = 800 s for dx = 4 km, v = 1 m/s).
# 2. `process_std_beta` is rescaled as σ_ref · √(time_step / time_step_ref).
#    Reason: process noise is a discretised random walk; keeping σ constant
#    per step makes the per-unit-physical-time variance scale with 1/dt.
#    The √dt scaling keeps the SDE-equivalent diffusion constant, so curves
#    differ only because of obs cadence — not because finer cadence
#    accidentally drowned itself in noise. Reference: σ_ref = 28 at dt_ref = 3600 s.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_08_obs_cadence_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots, Printf

const FIG_NAME = "rmse_obs_cadence_sweep_exp08.png"
const NOTES    = "Linear adv; obs cadence sweep over fixed ≈100h. σ_proc rescaled √(dt/3600)·28; n_integration_step set so dt_inner ≤ 720s."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "linear",
)
const NPRT = 1000
const TOTAL_TIME_S = 360_000.0          # ~100 hours of physical time
const SIGMA_PROC_REF = 28.0
const DT_REF         = 3600.0

# Cadence table: (label, time_step_s, n_integration_step, T_steps)
# T is chosen so T·time_step ≈ TOTAL_TIME_S.
# n_int is chosen so dt_inner = time_step / n_int ≤ 720 s.
const CADENCES = [
    ("10 min",   600.0,    1,  600),  # dt_inner = 600 s
    ("30 min",  1800.0,    3,  200),  # dt_inner = 600 s
    ("1 h",     3600.0,   10,  100),  # baseline; dt_inner = 360 s
    ("3 h",    10800.0,   15,   33),  # dt_inner = 720 s
    ("6 h",    21600.0,   30,   17),  # dt_inner = 720 s
    ("12 h",   43200.0,   60,    9),  # dt_inner = 720 s
]
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick, :purple]

println("=== Experiment 08: obs cadence sweep on linear advection ===")
results = []
runtimes = Float64[]
for (label, dt_obs, n_int, T) in CADENCES
    σ_proc_scaled = SIGMA_PROC_REF * sqrt(dt_obs / DT_REF)
    println(@sprintf("Running cadence = %s  (T=%d, n_int=%d, dt_inner=%.0fs, σ_proc=%.2f) …",
                     label, T, n_int, dt_obs/n_int, σ_proc_scaled))
    params = merge(base_params, Dict{String, Any}(
        "time_step"          => dt_obs,
        "n_integration_step" => n_int,
        "process_std_beta"   => σ_proc_scaled,
    ))
    t0 = time()
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    push!(runtimes, time() - t0)
    push!(results, (label=label, dt_obs=dt_obs, n_int=n_int,
                    T=T, σ_proc=σ_proc_scaled, res=res))
    @printf("  %s  RMSE final=%-6.1f  mean RMSE=%-6.1f  mean ESS=%-6.1f  time=%.1fs\n",
            label, res.rmse_global[end], mean(res.rmse_global),
            mean(res.ess), runtimes[end])
end

# ── Plot (shared physical-time x-axis) ──────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 08 — observation cadence sweep (linear advection)",
         legend=:topright, lw=2.5, xlim=(0, 105))
for (i, r) in enumerate(results)
    plot!(p, r.res.model_time, r.res.rmse_global;
          label="every $(r.label)", lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Runtime summary ─────────────────────────────────────────────────────
total_time = sum(runtimes)
println("\n--- Runtime summary ---")
for (i, r) in enumerate(results)
    @printf("  %-7s  %.2f s\n", r.label, runtimes[i])
end
@printf("  TOTAL    %.2f s (%.1f min)\n", total_time, total_time/60)

# ── Log one row per cadence ─────────────────────────────────────────────
n_sensors = length(results[1].res.sensor_idx)
for (i, r) in enumerate(results)
    runtime_note = @sprintf("%s; cadence=%s; n_int=%d; dt_inner=%.0fs; σ_proc=%.2f; wall %.1fs",
                            NOTES, r.label, r.n_int, r.dt_obs/r.n_int, r.σ_proc, runtimes[i])
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Linear",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = r.σ_proc,
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = n_sensors,
        obs_interval_s  = r.dt_obs,
        sigma_obs       = base_params["obs_noise_std"],
        T_steps         = r.T,
        notes           = runtime_note,
    )
end
println("Logged $(length(results)) rows → $(LOG_PATH)")
