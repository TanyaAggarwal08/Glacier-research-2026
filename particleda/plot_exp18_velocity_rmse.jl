# Exp 18 — velocity RMSE at the observation points, over time.
#
# Velocity is only evaluated where the sensors are (16 cells), so this is a
# discrete series (one point per timestep), not a continuous field RMSE. It
# answers: is the estimated surface velocity getting closer to the truth as the
# filter runs?
#
# Uses the log-speeds already stored by experiment_18 (truth_logspeed = log|u|
# of the truth β, mean_logspeed = log|u| of the ensemble-mean β, both at the
# sensors, per step). No WAVI re-solve: speed = exp(log-speed).
#
# Two metrics per step, across the 16 sensors:
#   absolute : sqrt(mean( (v_est − v_truth)^2 ))            [m/yr]
#   relative : sqrt(mean( ((v_est − v_truth)/v_truth)^2 ))  [fraction]
# The relative one matters because sensor speeds span ~1.5e3–1.5e4 m/yr, so the
# absolute RMSE is dominated by the few fastest sensors.
#
# Run: julia glacier-code/particleda/plot_exp18_velocity_rmse.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)
using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const RUN_TAG = length(ARGS) >= 1 ? ARGS[1] : "experiment_18_wavi_obs"
const OUT = joinpath("glacier-code", "particleda", "results", RUN_TAG)
const TRACK = let
    local_h5 = joinpath(OUT, "tracking.h5")
    ssd_h5   = joinpath("/Volumes/ZX20/USRA 2026", RUN_TAG, "tracking.h5")
    isfile(local_h5) ? local_h5 : ssd_h5
end
@assert isfile(TRACK) "tracking.h5 not found (local or SSD)"

truth_logspeed, mean_logspeed, dt = h5open(TRACK, "r") do f
    (read(f["truth_logspeed"]), read(f["mean_logspeed"]),
     read(attributes(f["parameters"])["time_step"]))
end
n_obs, T = size(truth_logspeed)
truth_speed = exp.(truth_logspeed)      # (16, T) m/yr
mean_speed  = exp.(mean_logspeed)       # (16, T) m/yr

t_h = collect(1:T) .* dt ./ 3600        # obs exist for t = 1..T
vel_rmse = [sqrt(mean((mean_speed[:, t] .- truth_speed[:, t]).^2)) for t in 1:T]
vel_rmse_rel = [sqrt(mean(((mean_speed[:, t] .- truth_speed[:, t]) ./ truth_speed[:, t]).^2)) for t in 1:T]

println("=== Exp 18 velocity RMSE at $(n_obs) sensors ===")
@printf("%-8s %14s %14s\n", "t (h)", "abs (m/yr)", "rel (%)")
for t in 1:T
    @printf("%-8.0f %14.1f %14.1f\n", t_h[t], vel_rmse[t], 100*vel_rmse_rel[t])
end
@printf("\nabsolute: initial %.0f → final %.0f m/yr\n", vel_rmse[1], vel_rmse[end])
@printf("relative: initial %.1f%% → final %.1f%%\n", 100*vel_rmse_rel[1], 100*vel_rmse_rel[end])

# absolute (m/yr) — discrete points + connecting line
p1 = plot(t_h, vel_rmse; seriestype=:line, lw=2, color=:darkorange, label=false,
          xlabel="time (h)", ylabel="velocity RMSE at sensors (m/yr)",
          title="Exp 18 — velocity RMSE (abs) at observation points")
scatter!(p1, t_h, vel_rmse; mc=:darkorange, ms=5, msw=0, label="per-step RMSE")
savefig(p1, joinpath(OUT, "velocity_rmse_abs.png"))

# relative (%) — more interpretable across the 10× speed range
p2 = plot(t_h, 100 .* vel_rmse_rel; seriestype=:line, lw=2, color=:seagreen, label=false,
          xlabel="time (h)", ylabel="relative velocity RMSE (%)",
          title="Exp 18 — velocity RMSE (relative) at observation points")
scatter!(p2, t_h, 100 .* vel_rmse_rel; mc=:seagreen, ms=5, msw=0, label="per-step RMSE")
savefig(p2, joinpath(OUT, "velocity_rmse_rel.png"))

println("\nSaved → $(joinpath(OUT, "velocity_rmse_abs.png"))")
println("Saved → $(joinpath(OUT, "velocity_rmse_rel.png"))")
