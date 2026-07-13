# Experiment 05 — RMSE: noise correlation length ℓ sweep on linear advection.
#
# Holds everything else fixed and varies noise_length_scale (the Matérn/SE
# decorrelation length used to build the smooth noise Cholesky factor).
# Plots one RMSE-vs-time curve per ℓ on a single figure.
# Logs one row per ℓ into rmse_experiment_log.md.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_05_length_scale_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots

const FIG_NAME = "rmse_length_scale_sweep_exp05.png"
const NOTES    = "Linear advection; correlation length ℓ sweep, all other params fixed."

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
    "advection_type" => "linear",
)
const NPRT = 1000
const T    = 100

# ℓ values (m). Domain is 160 km, prior wavelength is 160/3 ≈ 53 km.
# 2 km  = near white-noise (per-cell independent)
# 5 km  = sub-wavelength
# 15 km = previous default (~wavelength/3)
# 30 km = ~wavelength/2
# 60 km = larger than wavelength (very smooth, big coherent blobs)
const LENGTH_SCALES = [2_000.0, 5_000.0, 15_000.0, 30_000.0, 60_000.0]
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]

println("=== Experiment 05: noise length-scale sweep on linear advection ===")
results = Dict{Float64, Any}()
for ℓ in LENGTH_SCALES
    println("Running ℓ = $(ℓ/1000) km …")
    params = merge(base_params, Dict{String, Any}("noise_length_scale" => ℓ))
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    results[ℓ] = res
    println("  ℓ=$(ℓ/1000) km  RMSE final=$(round(res.rmse_global[end], digits=1))  " *
            "mean RMSE=$(round(mean(res.rmse_global), digits=1))  " *
            "mean ESS=$(round(mean(res.ess), digits=1))")
end

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 05 — noise correlation length ℓ sweep (linear advection)",
         legend=:topright, lw=2.5)
for (i, ℓ) in enumerate(LENGTH_SCALES)
    res = results[ℓ]
    plot!(p, res.model_time, res.rmse_global;
          label="ℓ = $(Int(ℓ/1000)) km", lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Log one row per ℓ ────────────────────────────────────────────────────
n_sensors = length(results[LENGTH_SCALES[1]].sensor_idx)
for ℓ in LENGTH_SCALES
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Linear",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = base_params["process_std_beta"],
        length_scale_km = ℓ / 1000,
        particles       = NPRT,
        n_obs           = n_sensors,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = base_params["obs_noise_std"],
        T_steps         = T,
        notes           = NOTES,
    )
end
println("Logged $(length(LENGTH_SCALES)) rows → $(LOG_PATH)")
