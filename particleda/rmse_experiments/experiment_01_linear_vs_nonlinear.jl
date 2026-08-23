# Experiment 01 — RMSE: Linear vs Nonlinear advection.
#
# Runs the bootstrap PF twice with identical parameters except for
# `advection_type`. Plots global RMSE vs time for both on one figure.
# Logs both rows into glacier-notes/rmse_experiment_log.md.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_01_linear_vs_nonlinear.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots

const FIG_NAME = "rmse_linear_vs_nonlinear_exp01.png"
const NOTES    = "Baseline; on-cross-section sensor file; same seeds for both runs."

# ── Shared baseline parameters (matches the current run08 configuration) ──
base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,
    "process_std_beta" => 28.0,
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
)
const NPRT = 1000
const T    = 100

println("=== Experiment 01: Linear vs Nonlinear advection ===")
println("Running linear …")
linear_params = merge(base_params, Dict{String, Any}("advection_type" => "linear"))
linear = run_pf_rmse(linear_params; NPRT=NPRT, T=T)
println("  linear  RMSE final=$(round(linear.rmse_global[end], digits=1))  " *
        "mean ESS=$(round(mean(linear.ess), digits=1))")

println("Running nonlinear …")
nonlinear_params = merge(base_params, Dict{String, Any}("advection_type" => "nonlinear"))
nonlinear = run_pf_rmse(nonlinear_params; NPRT=NPRT, T=T)
println("  nonlin  RMSE final=$(round(nonlinear.rmse_global[end], digits=1))  " *
        "mean ESS=$(round(mean(nonlinear.ess), digits=1))")

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(linear.model_time, linear.rmse_global;
         xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 01 — Linear vs Nonlinear advection",
         label="Linear (v = 1)", lw=2.5, color=:steelblue,
         legend=:bottomright)
plot!(p, nonlinear.model_time, nonlinear.rmse_global;
      label="Nonlinear (v = 1 + ε·β)", lw=2.5, color=:firebrick)
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Log both rows ────────────────────────────────────────────────────────
n_sensors = length(linear.sensor_idx)
common = (;
    figure          = FIG_NAME,
    sigma_init      = base_params["init_std_beta"],
    sigma_proc      = base_params["process_std_beta"],
    length_scale_km = base_params["noise_length_scale"] / 1000,
    particles       = NPRT,
    n_obs           = n_sensors,
    obs_interval_s  = base_params["time_step"],
    sigma_obs       = base_params["obs_noise_std"],
    T_steps         = T,
    notes           = NOTES,
)
log_experiment!(; common..., model_type="Linear")
log_experiment!(; common..., model_type="Nonlinear")
println("Logged → $(LOG_PATH)")
