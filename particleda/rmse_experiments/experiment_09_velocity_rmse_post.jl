# Experiment 09 (post-processing stage) — velocity RMSE via WAVI.
#
# Loads β trajectories saved by experiment_09_velocity_rmse_pf.jl. For
# each time step, runs the WAVI ice-flow forward model on:
#   1. the truth β field  → truth velocity (u_t, v_t)
#   2. the ensemble-mean β field → estimated velocity (u_m, v_m)
# and computes velocity-RMSE per step using speed |v| = √(u² + v²).
# Plots β-RMSE (continuous line) and velocity-RMSE (discrete points) on
# a shared time axis.
#
# Runs under the DEFAULT Julia env because WAVI + its HDF5/NetCDF stack
# only load cleanly there (the `test` env has an incompatible libhdf5).
#
# Run: julia glacier-code/particleda/rmse_experiments/experiment_09_velocity_rmse_post.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

# IceFlow lives in the particleda/ directory.
include(joinpath(REPO_ROOT, "glacier-code", "particleda", "ice_flow.jl"))
using .IceFlow

const RMSE_OUT = joinpath(REPO_ROOT, "glacier-code", "particleda",
                          "results", "rmse_analysis")
const IN_H5    = joinpath(RMSE_OUT, "exp09_beta_trajectories.h5")
const FIG_OUT  = joinpath(RMSE_OUT, "exp09_velocity_rmse.png")

isfile(IN_H5) || error("Missing $(IN_H5). Run experiment_09_velocity_rmse_pf.jl first.")

# ── Load trajectories ────────────────────────────────────────────────────
data = h5open(IN_H5, "r") do f
    (
        truth_beta = read(f["truth_beta"]),      # (nx*ny, T+1)
        mean_beta  = read(f["mean_beta"]),       # (nx*ny, T+1)
        model_time = read(f["model_time_h"]),    # (T+1,)
        rmse_beta  = read(f["rmse_beta"]),       # (T+1,)
        nx         = read(attributes(f["parameters"])["nx"]),
        ny         = read(attributes(f["parameters"])["ny"]),
    )
end
T_plus_1 = length(data.model_time)
println("Loaded β trajectories: $(T_plus_1) time steps, nx=$(data.nx), ny=$(data.ny).")

@assert data.nx == IceFlow.NX && data.ny == IceFlow.NY (
    "Grid mismatch: HDF5 says $(data.nx)×$(data.ny) but IceFlow expects "*
    "$(IceFlow.NX)×$(IceFlow.NY). Regenerate IceFlow with matching NX/NY.")

# ── Velocity solves per step ────────────────────────────────────────────
# The user's spec: compute velocity at every filter step (every hour by
# default). That's 2 solves × (T+1) steps.
rmse_velocity = fill(NaN, T_plus_1)
speed_truth_mean_final = (NaN, NaN)   # for a small sanity print at the end

println("Running WAVI on truth and ensemble-mean β at every step …")
t_start = time()
for t in 1:T_plus_1
    ft = IceFlow.velocity_flat(data.truth_beta[:, t])
    fm = IceFlow.velocity_flat(data.mean_beta[:, t])
    speed_t = sqrt.(ft.u .^ 2 .+ ft.v .^ 2)
    speed_m = sqrt.(fm.u .^ 2 .+ fm.v .^ 2)
    rmse_velocity[t] = sqrt(mean((speed_t .- speed_m) .^ 2))
    if t == T_plus_1
        speed_truth_mean_final = (mean(speed_t), mean(speed_m))
    end
    if t % 20 == 0 || t == T_plus_1
        @printf("  step %3d/%d  velocity-RMSE = %.2f m/yr\n",
                t, T_plus_1, rmse_velocity[t])
    end
end
elapsed = time() - t_start
@printf("WAVI stage done in %.1fs (%.1f min).\n", elapsed, elapsed/60)
@printf("Final step: mean truth-speed = %.1f m/yr, mean estimated-speed = %.1f m/yr.\n",
        speed_truth_mean_final...)

# ── Plot: β-RMSE (line) + velocity-RMSE (discrete dots) ─────────────────
p1 = plot(data.model_time, data.rmse_beta;
          xlabel="time (h)", ylabel="β RMSE (Pa·s/m)",
          title="Exp 09 — β vs velocity RMSE (linear advection)",
          label="β RMSE (continuous)", lw=2.5, color=:steelblue,
          legend=:topright)

# Velocity RMSE overlaid on a twin axis so both are visible at their
# natural scales.
p2 = twinx(p1)
scatter!(p2, data.model_time, rmse_velocity;
         ylabel="velocity RMSE (m/yr)", marker=:circle, ms=3, msw=0,
         mc=:firebrick, label="velocity RMSE (per step)",
         legend=:bottomright)

savefig(p1, FIG_OUT)
println("Saved → $FIG_OUT")

# Print a quick trend summary so the log has the numbers.
@printf("β-RMSE   : t=0 → %.1f    t=end → %.1f\n",
        data.rmse_beta[1], data.rmse_beta[end])
@printf("vel-RMSE : t=0 → %.2f    t=end → %.2f\n",
        rmse_velocity[1], rmse_velocity[end])
