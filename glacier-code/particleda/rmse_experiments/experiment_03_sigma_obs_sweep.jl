# Experiment 03 — RMSE: σ_obs sweep on linear advection.
#
# Holds everything else fixed (linear advection, same seeds, same prior
# spread) and varies obs_noise_std across a range. Plots one RMSE-vs-time
# curve per σ_obs on a single figure to show how observation noise affects
# PF performance. Logs one row per σ_obs into rmse_experiment_log.md.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_03_sigma_obs_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots

const FIG_NAME = "rmse_sigma_obs_sweep_exp03.png"
const NOTES    = "Linear advection; σ_obs sweep with all other params fixed."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,
    "process_std_beta" => 28.0,
    "advection_epsilon" => 5e-4,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "linear",
)
const NPRT = 1000
const T    = 100

# σ_obs values to sweep (in ux units, since obs are ux = 1000/β).
# 0.02 ≈ 2 % rel, 0.05 ≈ 5 % rel (Mouginot 2019 InSAR), 0.10 = current default,
# 0.20 = very noisy, 0.50 = essentially uninformative.
const SIGMA_OBS_VALUES = [0.02, 0.05, 0.10, 0.20, 0.50]
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]

println("=== Experiment 03: σ_obs sweep on linear advection ===")
results = Dict{Float64, Any}()
for σ in SIGMA_OBS_VALUES
    println("Running σ_obs = $σ …")
    params = merge(base_params, Dict{String, Any}("obs_noise_std" => σ))
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    results[σ] = res
    println("  σ_obs=$σ  RMSE final=$(round(res.rmse_global[end], digits=1))  " *
            "mean RMSE=$(round(mean(res.rmse_global), digits=1))  " *
            "mean ESS=$(round(mean(res.ess), digits=1))")
end

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 03 — σ_obs sweep (linear advection)",
         legend=:topright, lw=2.5)
for (i, σ) in enumerate(SIGMA_OBS_VALUES)
    res = results[σ]
    plot!(p, res.model_time, res.rmse_global;
          label="σ_obs = $σ", lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Log one row per σ_obs ────────────────────────────────────────────────
n_sensors = length(results[SIGMA_OBS_VALUES[1]].sensor_idx)
for σ in SIGMA_OBS_VALUES
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Linear",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = base_params["process_std_beta"],
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = n_sensors,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = σ,
        T_steps         = T,
        notes           = NOTES,
    )
end
println("Logged $(length(SIGMA_OBS_VALUES)) rows → $(LOG_PATH)")
