# Experiment 12 — RMSE: n_obs sweep with Lax–Wendroff, σ_proc = 28.
#
# Mirrors Exp 07 (n_obs sweep on linear advection, σ_proc = 28) but with
# advection_type flipped to "lax_wendroff". Purpose: see how the sweep
# behaves when we combine the amplitude-preserving stencil with real
# process noise. Exp 11 kept σ_proc = 0 and got degenerate results
# (filter can't diversify frozen particles); this reintroduces the
# process-noise "search" mechanism.
#
# All other configuration matches Exp 07: σ_init = 600, σ_obs = 0.10,
# ℓ = 15 km, same seeds, same sensor pool. Only differences vs Exp 07
# are `advection_type` and `advection_epsilon = 0.0` (LW strictly linear).
#
# Runs under --project=test.
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_12_n_obs_sweep_lw.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots, Printf, Random

const FIG_NAME = "rmse_n_obs_sweep_lw_exp12.png"
const NOTES    = "Lax–Wendroff advection; σ_proc = 28; n_obs sweep (Exp 07 rerun with LW)."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "init_std_beta"      => 600.0,
    "process_std_beta"   => 28.0,          # ← back to Exp 07 value
    "obs_noise_std"      => 0.10,
    "advection_epsilon"  => 0.0,           # strictly linear LW
    "n_integration_step" => 10,
    "time_step"          => 3600.0,
    "min_beta"           => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type"     => "lax_wendroff",
)
const NPRT = 1000
const T    = 100

const N_OBS_VALUES = [0, 1, 5, 10, 25, 50]
const COLORS = [:grey, :steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]

# Sensor pool — same seed as Exp 07 / 10 / 11 for direct comparability.
const STATION_DIR = joinpath(@__DIR__, "_tmp_stations_exp12")
mkpath(STATION_DIR)
const SENSOR_RNG = MersenneTwister(2026)
const POOL_N = maximum(N_OBS_VALUES)
const POOL = let
    nx, ny = base_params["nx"], base_params["ny"]
    xs = (rand(SENSOR_RNG, 0:nx-1, POOL_N) .+ 0.5) .* (base_params["x_length"] / nx)
    ys = (rand(SENSOR_RNG, 0:ny-1, POOL_N) .+ 0.5) .* (base_params["y_length"] / ny)
    collect(zip(xs, ys))
end

function _write_station_file(n::Int)
    path = joinpath(STATION_DIR, "stations_n$(n).txt")
    open(path, "w") do io
        println(io, "# Auto-generated for experiment_12 (n=$n).")
        println(io, "# Columns: x (m), y (m).")
        for k in 1:n
            x, y = POOL[k]
            println(io, "$x, $y")
        end
    end
    return path
end

println("=== Experiment 12: n_obs sweep, Lax–Wendroff, σ_proc = 28 ===")
results = Dict{Int, Any}()
runtimes = Dict{Int, Float64}()
for n in N_OBS_VALUES
    println("Running n_obs = $n …")
    n_for_file = max(n, 1)
    path = _write_station_file(n_for_file)
    params = merge(base_params, Dict{String, Any}(
        "station_filename"     => path,
        "disable_observations" => (n == 0),
    ))
    t0 = time()
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    runtimes[n] = time() - t0
    results[n] = res
    @printf("  n_obs=%-3d  RMSE final=%-6.1f  mean RMSE=%-6.1f  mean ESS=%-6.1f  time=%.1fs\n",
            n, res.rmse_global[end], mean(res.rmse_global), mean(res.ess), runtimes[n])
end

# ── Plot ─────────────────────────────────────────────────────────────────
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 12 — n_obs sweep (Lax–Wendroff, σ_proc = 28)",
         legend=:topright, lw=2.5)
for (i, n) in enumerate(N_OBS_VALUES)
    res = results[n]
    lbl = n == 0 ? "n_obs = 0 (no observations)" : "n_obs = $n"
    plot!(p, res.model_time, res.rmse_global;
          label=lbl, lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Runtime summary ─────────────────────────────────────────────────────
total_time = sum(values(runtimes))
println("\n--- Runtime summary ---")
for n in N_OBS_VALUES
    @printf("  n_obs=%-3d  %.2f s\n", n, runtimes[n])
end
@printf("  TOTAL     %.2f s (%.1f min)\n", total_time, total_time/60)

# ── Log one row per n_obs ───────────────────────────────────────────────
for n in N_OBS_VALUES
    runtime_note = @sprintf("%s (wall %.1fs)", NOTES, runtimes[n])
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Lax–Wendroff",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = base_params["process_std_beta"],
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = n,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = base_params["obs_noise_std"],
        T_steps         = T,
        notes           = runtime_note,
    )
end
println("Logged $(length(N_OBS_VALUES)) rows → $(LOG_PATH)")
