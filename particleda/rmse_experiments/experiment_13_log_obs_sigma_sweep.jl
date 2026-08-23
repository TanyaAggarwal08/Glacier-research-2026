# Experiment 13 — log-velocity observation operator: σ_log sweep.
#
# The default operator observes ux = 1000/β with additive Gaussian noise. This
# experiment flips obs_space to "log_velocity", i.e. h(β) = log(1000) - log(β),
# which is the log-β field up to a sign and a constant. The likelihood only
# sees the squared residual, so the misfit becomes exactly
#
#     log(β_particle) - log(β_truth) + ε,      ε ~ N(0, σ_log²)
#
# — a purely *relative* comparison. One σ_log then buys the same fractional
# accuracy at every β, whereas a single σ_v in velocity space means 10 %
# relative error at β = 1000 but 30 % at β = 3000 (since ux = 1000/β).
#
# σ_log is swept rather than assumed. The delta method (σ_log ≈ σ_v/ux =
# σ_v·β/1000) says the incumbent σ_v = 0.10 is worth σ_log ≈ 0.20 at the prior
# centre β = 2000, so the sweep brackets that on both sides. The velocity-space
# run at σ_v = 0.10 is included as the baseline; RMSE is measured in β units
# throughout, so every curve is on the same footing.
#
# All other configuration matches Exp 12 (Lax–Wendroff, σ_proc = 28, ℓ = 15 km,
# σ_init = 600, same sensor-pool seed), with n_obs fixed at 25.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_13_log_obs_sigma_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
ENV["GKSwstype"] = "100"
using Plots, Printf, Random

const FIG_NAME     = "rmse_log_obs_sigma_sweep_exp13.png"
const FIG_NAME_REL = "rmse_log_obs_relative_exp13.png"
const NOTES        = "Log-velocity obs operator; σ_log sweep vs velocity-space baseline."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "init_std_beta"      => 600.0,
    "process_std_beta"   => 28.0,
    "advection_epsilon"  => 0.0,           # strictly linear LW
    "n_integration_step" => 10,
    "time_step"          => 3600.0,
    "min_beta"           => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type"     => "lax_wendroff",
)
const NPRT  = 1000
const T     = 100
const N_OBS = 25

const SIGMA_LOG_VALUES = [0.05, 0.10, 0.20, 0.30, 0.50]
const SIGMA_V_BASELINE = 0.10

# Sensor pool — same seed as Exp 12 so the sensor layout is identical.
const STATION_DIR = joinpath(@__DIR__, "_tmp_stations_exp13")
mkpath(STATION_DIR)
const SENSOR_RNG = MersenneTwister(2026)
const POOL = let
    nx, ny = base_params["nx"], base_params["ny"]
    xs = (rand(SENSOR_RNG, 0:nx-1, 50) .+ 0.5) .* (base_params["x_length"] / nx)
    ys = (rand(SENSOR_RNG, 0:ny-1, 50) .+ 0.5) .* (base_params["y_length"] / ny)
    collect(zip(xs, ys))
end

const STATION_FILE = let
    path = joinpath(STATION_DIR, "stations_n$(N_OBS).txt")
    open(path, "w") do io
        println(io, "# Auto-generated for experiment_13 (n=$N_OBS).")
        println(io, "# Columns: x (m), y (m).")
        for k in 1:N_OBS
            x, y = POOL[k]
            println(io, "$x, $y")
        end
    end
    path
end

println("=== Experiment 13: log-velocity obs operator, σ_log sweep ===")
println("n_obs = $N_OBS, N = $NPRT, T = $T\n")

# ── Velocity-space baseline ──────────────────────────────────────────────
println("Running velocity-space baseline σ_v = $SIGMA_V_BASELINE …")
baseline = run_pf_rmse(merge(base_params, Dict{String, Any}(
    "station_filename" => STATION_FILE,
    "obs_space"        => "velocity",
    "obs_noise_std"    => SIGMA_V_BASELINE,
)); NPRT=NPRT, T=T)
@printf("  baseline  RMSE final=%-6.1f  mean RMSE=%-6.1f  mean ESS=%-6.1f\n\n",
        baseline.rmse_global[end], mean(baseline.rmse_global), mean(baseline.ess))

# ── log-velocity sweep ───────────────────────────────────────────────────
results = Dict{Float64, Any}()
for σ in SIGMA_LOG_VALUES
    println("Running σ_log = $σ …")
    params = merge(base_params, Dict{String, Any}(
        "station_filename"   => STATION_FILE,
        "obs_space"          => "log_velocity",
        "obs_noise_std_log"  => σ,
    ))
    res = run_pf_rmse(params; NPRT=NPRT, T=T)
    results[σ] = res
    @printf("  σ_log=%-5.2f  RMSE final=%-6.1f  mean RMSE=%-6.1f  mean ESS=%-6.1f\n",
            σ, res.rmse_global[end], mean(res.rmse_global), mean(res.ess))
end

# ── Pick the winner by mean RMSE over the second half (post spin-up) ─────
const HALF = T ÷ 2
_score(res) = mean(res.rmse_global[HALF:end])
best_σ = argmin(σ -> _score(results[σ]), SIGMA_LOG_VALUES)
@printf("\nBest σ_log = %.2f (mean RMSE over t > %d h = %.1f); baseline = %.1f\n",
        best_σ, HALF, _score(results[best_σ]), _score(baseline))

# ── Plot: RMSE(β) vs time ────────────────────────────────────────────────
const COLORS = [:steelblue, :seagreen, :goldenrod, :darkorange, :firebrick]
p = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
         title="Exp 13 — log-velocity obs operator, σ_log sweep",
         legend=:topright, lw=2.5)
plot!(p, baseline.model_time, baseline.rmse_global;
      label="velocity space, σ_v = $SIGMA_V_BASELINE (baseline)",
      lw=3, color=:black, linestyle=:dash)
for (i, σ) in enumerate(SIGMA_LOG_VALUES)
    res = results[σ]
    lbl = σ == best_σ ? "log space, σ_log = $σ  ★ best" : "log space, σ_log = $σ"
    plot!(p, res.model_time, res.rmse_global; label=lbl, lw=2.5, color=COLORS[i])
end
savefig(p, joinpath(RMSE_OUT, FIG_NAME))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME))")

# ── Plot: where each operator wins, as a function of β ───────────────────
# The whole argument for log space is that it weights fast (low-β) and slow
# (high-β) cells evenly. Bin cells by their true β and compare final-time
# per-cell error between the best log run and the baseline.
best = run_pf_rmse(merge(base_params, Dict{String, Any}(
    "station_filename"  => STATION_FILE,
    "obs_space"         => "log_velocity",
    "obs_noise_std_log" => best_σ,
)); NPRT=NPRT, T=T, return_trajectories=true)
base_traj = run_pf_rmse(merge(base_params, Dict{String, Any}(
    "station_filename" => STATION_FILE,
    "obs_space"        => "velocity",
    "obs_noise_std"    => SIGMA_V_BASELINE,
)); NPRT=NPRT, T=T, return_trajectories=true)

β_true_final   = base_traj.truth_states[:, end]
err_log_final  = abs.(best.ensemble_mean[:, end]      .- β_true_final)
err_vel_final  = abs.(base_traj.ensemble_mean[:, end] .- β_true_final)

edges   = range(minimum(β_true_final), maximum(β_true_final); length=9)
centers = [(edges[i] + edges[i+1]) / 2 for i in 1:length(edges)-1]
function _binned(err)
    [mean(err[(β_true_final .>= edges[i]) .& (β_true_final .< edges[i+1])])
     for i in 1:length(edges)-1]
end
p2 = plot(centers, _binned(err_vel_final);
          xlabel="true β (Pa·s/m)", ylabel="mean |error| in β at final time",
          title="Exp 13 — per-cell error vs β (final time)",
          label="velocity space, σ_v = $SIGMA_V_BASELINE",
          lw=2.5, color=:black, linestyle=:dash, marker=:circle, legend=:topleft)
plot!(p2, centers, _binned(err_log_final);
      label="log space, σ_log = $best_σ", lw=2.5, color=:firebrick, marker=:square)
savefig(p2, joinpath(RMSE_OUT, FIG_NAME_REL))
println("Saved → $(joinpath(RMSE_OUT, FIG_NAME_REL))")

# ── Log one row per configuration ───────────────────────────────────────
log_experiment!(;
    figure          = FIG_NAME,
    model_type      = "Lax–Wendroff",
    sigma_init      = base_params["init_std_beta"],
    sigma_proc      = base_params["process_std_beta"],
    length_scale_km = base_params["noise_length_scale"] / 1000,
    particles       = NPRT,
    n_obs           = N_OBS,
    obs_interval_s  = base_params["time_step"],
    sigma_obs       = SIGMA_V_BASELINE,
    T_steps         = T,
    notes           = "$NOTES Velocity-space baseline (obs = ux).",
)
for σ in SIGMA_LOG_VALUES
    star = σ == best_σ ? " BEST by mean RMSE over t > $(HALF) h." : ""
    log_experiment!(;
        figure          = FIG_NAME,
        model_type      = "Lax–Wendroff",
        sigma_init      = base_params["init_std_beta"],
        sigma_proc      = base_params["process_std_beta"],
        length_scale_km = base_params["noise_length_scale"] / 1000,
        particles       = NPRT,
        n_obs           = N_OBS,
        obs_interval_s  = base_params["time_step"],
        sigma_obs       = σ,
        T_steps         = T,
        notes           = "$NOTES σ_log = $σ (log space; σ column is σ_log, not σ_ux).$star",
    )
end
println("Logged $(length(SIGMA_LOG_VALUES) + 1) rows → $(LOG_PATH)")
