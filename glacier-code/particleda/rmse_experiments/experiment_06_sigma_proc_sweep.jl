# Experiment 06 — RMSE: σ_proc (process noise) sweep on linear advection.
#
# Holds everything else fixed and varies process_std_beta across a range.
# σ_proc = 0 is included to show what happens when there is no process noise
# at all (particles evolve deterministically between obs).
# Plots one RMSE-vs-time curve per σ_proc on a single figure.
# Logs one row per σ_proc into rmse_experiment_log.md.
# Prints wall-clock time for each run and a summary at the end.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_06_sigma_proc_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots, Printf

const FIG_NAME = "rmse_sigma_proc_sweep_exp06.png"
const NOTES    = "Linear advection; σ_proc sweep with all other params fixed. σ_proc=0 = deterministic forecast."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,
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

# σ_proc values (β units). 0.0 = deterministic forecast (no process noise);
# 7 = current run10 default; 28 = previous run08; 100/300 = very large.
const SIGMA_PROC_VALUES = [0.0, 7.0, 28.0, 100.0, 300.0]
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]

println("=== Experiment 06: σ_proc sweep on linear advection ===")
results = Dict{Float64, Any}()
runtimes = Dict{Float64, Float64}()
for σ in SIGMA_PROC_VALUES
    println("Running σ_proc = $σ …")
    params = merge(base_params, Dict{String, Any}("process_std_beta" => σ))
    t0 = time()
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    runtimes[σ] = time() - t0
    results[σ] = res
    @printf("  σ_proc=%-6.1f  RMSE final=%-6.1f  mean RMSE=%-6.1f  mean ESS=%-6.1f  time=%.1fs\n",
            σ, res.rmse_global[end], mean(res.rmse_global), mean(res.ess), runtimes[σ])
end

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 06 — σ_proc sweep (linear advection)",
         legend=:topright, lw=2.5)
for (i, σ) in enumerate(SIGMA_PROC_VALUES)
    res = results[σ]
    lbl = σ == 0.0 ? "σ_proc = 0 (deterministic)" : "σ_proc = $σ"
    plot!(p, res.model_time, res.rmse_global;
          label=lbl, lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Runtime summary ─────────────────────────────────────────────────────
total_time = sum(values(runtimes))
println("\n--- Runtime summary ---")
for σ in SIGMA_PROC_VALUES
    @printf("  σ_proc=%-6.1f  %.2f s\n", σ, runtimes[σ])
end
@printf("  TOTAL     %.2f s (%.1f min)\n", total_time, total_time/60)

# ── Log one row per σ_proc ──────────────────────────────────────────────
n_sensors = length(results[SIGMA_PROC_VALUES[1]].sensor_idx)
for σ in SIGMA_PROC_VALUES
    runtime_note = @sprintf("%s (wall %.1fs)", NOTES, runtimes[σ])
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Linear",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = σ,
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = n_sensors,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = base_params["obs_noise_std"],
        T_steps         = T,
        notes           = runtime_note,
    )
end
println("Logged $(length(SIGMA_PROC_VALUES)) rows → $(LOG_PATH)")
