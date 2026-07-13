# Experiment 04 — RMSE: σ_init sweep on linear advection.
#
# Holds everything else fixed and varies init_std_beta across a range.
# Plots one RMSE-vs-time curve per σ_init on a single figure.
# Logs one row per σ_init into rmse_experiment_log.md.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_04_sigma_init_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots

const FIG_NAME = "rmse_sigma_init_sweep_exp04.png"
const NOTES    = "Linear advection; σ_init sweep with all other params fixed."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "process_std_beta" => 28.0,
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "linear",
)
const NPRT = 1000
const T    = 100

# σ_init values to sweep (β units). Prior amplitude is 2000, so:
# 50 = very tight, 150 = previous default, 300/600 = moderate, 1200 = very broad.
const SIGMA_INIT_VALUES = [50.0, 150.0, 300.0, 600.0, 1200.0]
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]

println("=== Experiment 04: σ_init sweep on linear advection ===")
results = Dict{Float64, Any}()
for σ in SIGMA_INIT_VALUES
    println("Running σ_init = $σ …")
    params = merge(base_params, Dict{String, Any}("init_std_beta" => σ))
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    results[σ] = res
    println("  σ_init=$σ  RMSE final=$(round(res.rmse_global[end], digits=1))  " *
            "mean RMSE=$(round(mean(res.rmse_global), digits=1))  " *
            "mean ESS=$(round(mean(res.ess), digits=1))")
end

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 04 — σ_init sweep (linear advection)",
         legend=:topright, lw=2.5)
for (i, σ) in enumerate(SIGMA_INIT_VALUES)
    res = results[σ]
    plot!(p, res.model_time, res.rmse_global;
          label="σ_init = $σ", lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Log one row per σ_init ───────────────────────────────────────────────
n_sensors = length(results[SIGMA_INIT_VALUES[1]].sensor_idx)
for σ in SIGMA_INIT_VALUES
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Linear",
        sigma_init      = σ,
        sigma_proc      = base_params["process_std_beta"],
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = n_sensors,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = base_params["obs_noise_std"],
        T_steps         = T,
        notes           = NOTES,
    )
end
println("Logged $(length(SIGMA_INIT_VALUES)) rows → $(LOG_PATH)")
