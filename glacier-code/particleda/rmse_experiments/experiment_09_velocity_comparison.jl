# Experiment 09 (companion visualisation) — truth vs estimated velocity.
#
# Same input as experiment_09_velocity_rmse_post.jl (the HDF5 trajectory
# dump from stage 1). Instead of collapsing into an RMSE number, this
# script shows the truth velocity and the ensemble-mean-estimated
# velocity side-by-side so the reader can see convergence with their own
# eyes.
#
# Outputs (all in results/ice_flow_model_run/):
#   - velocity_crosssection_anim.gif  — |v|(x) at y = 76 km, truth vs mean,
#     one frame per PF step
#   - velocity_heatmap_t0.png / velocity_heatmap_final.png — 2D speed
#     heatmaps of truth (left) vs mean (right) at first and last step
#   - velocity_probe_sensor.png — |v|(t) at a fixed sensor cell,
#     truth and mean as two line series
#
# Runs under the DEFAULT Julia env (WAVI).
# Run: julia glacier-code/particleda/rmse_experiments/experiment_09_velocity_comparison.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

include(joinpath(REPO_ROOT, "glacier-code", "particleda", "ice_flow.jl"))
using .IceFlow

const IN_H5  = joinpath(REPO_ROOT, "glacier-code", "particleda", "results",
                        "rmse_analysis", "exp09_beta_trajectories.h5")
const OUT    = joinpath(REPO_ROOT, "glacier-code", "particleda", "results",
                        "ice_flow_model_run")
mkpath(OUT)

isfile(IN_H5) || error("Missing $(IN_H5). Run experiment_09_velocity_rmse_pf.jl first.")

# ── Load β trajectories ─────────────────────────────────────────────────
data = h5open(IN_H5, "r") do f
    (
        truth_beta = read(f["truth_beta"]),
        mean_beta  = read(f["mean_beta"]),
        model_time = read(f["model_time_h"]),
        nx         = read(attributes(f["parameters"])["nx"]),
        ny         = read(attributes(f["parameters"])["ny"]),
    )
end
T_plus_1 = length(data.model_time)
NX, NY = data.nx, data.ny
println("Loaded β trajectories: $(T_plus_1) time steps, $(NX)×$(NY).")

# ── WAVI solves — cache all velocity fields so we don't re-solve ────────
println("Running WAVI on truth and mean β at every step …")
t_start = time()
# Store as (NX, NY, T+1) in WAVI convention.
truth_speed_all = zeros(NX, NY, T_plus_1)
mean_speed_all  = zeros(NX, NY, T_plus_1)
for t in 1:T_plus_1
    ft = IceFlow.velocity_flat(data.truth_beta[:, t])
    fm = IceFlow.velocity_flat(data.mean_beta[:, t])
    # velocity_flat returns vectors in Glacier convention (Y, X). Reshape
    # back to (NY, NX) then transpose to (NX, NY) for plotting alongside
    # WAVI's xxh/yyh grid.
    ut_yx = reshape(ft.u, NY, NX); vt_yx = reshape(ft.v, NY, NX)
    um_yx = reshape(fm.u, NY, NX); vm_yx = reshape(fm.v, NY, NX)
    ut = Matrix(transpose(ut_yx)); vt = Matrix(transpose(vt_yx))
    um = Matrix(transpose(um_yx)); vm = Matrix(transpose(vm_yx))
    truth_speed_all[:, :, t] = sqrt.(ut .^ 2 .+ vt .^ 2)
    mean_speed_all[:, :, t]  = sqrt.(um .^ 2 .+ vm .^ 2)
    if t % 20 == 0 || t == T_plus_1
        @printf("  step %3d/%d\n", t, T_plus_1)
    end
end
@printf("WAVI stage done in %.1fs (%.1f min).\n",
        time() - t_start, (time() - t_start)/60)

# Save velocity fields too so downstream scripts can reuse them.
h5_out = joinpath(OUT, "velocity_fields.h5")
isfile(h5_out) && rm(h5_out)
h5open(h5_out, "w") do f
    f["truth_speed"] = truth_speed_all
    f["mean_speed"]  = mean_speed_all
    f["model_time_h"] = data.model_time
end
println("Saved velocity fields → $h5_out")

# ── Plot 1: cross-section anim at y = 76 km ─────────────────────────────
# Same reference row used in the β cross-section GIFs so plots are
# comparable. In WAVI's (NX, NY) layout, y is second index.
L_KM     = IceFlow.L / 1000
xs_km    = collect(0:NX-1) .* (L_KM / NX)
row_j    = clamp(round(Int, 76.0 / (L_KM / NY)) + 1, 1, NY)
row_y_km = (row_j - 1) * (L_KM / NY)

# Robust y-limits: 5th–95th percentile across all steps + a margin.
allvals = vcat(vec(truth_speed_all[:, row_j, :]), vec(mean_speed_all[:, row_j, :]))
ymin = max(0.0, quantile(filter(isfinite, allvals), 0.02))
ymax = quantile(filter(isfinite, allvals), 0.98) * 1.05

anim = @animate for t in 1:T_plus_1
    p = plot(xs_km, truth_speed_all[:, row_j, t];
             lw=3, color=:black, label="truth speed",
             xlabel="x (km)", ylabel="|v| (m/yr)",
             title=@sprintf("Ice speed at y=%.0f km  t=%.1f h",
                            row_y_km, data.model_time[t]),
             ylim=(ymin, ymax), legend=:topright)
    plot!(p, xs_km, mean_speed_all[:, row_j, t];
          lw=2.5, color=:firebrick, linestyle=:dash,
          label="ensemble-mean speed")
end
gif(anim, joinpath(OUT, "velocity_crosssection_anim.gif"), fps=6)
println("Saved velocity_crosssection_anim.gif")

# ── Plot 2: heatmaps at t=0 and t=T ─────────────────────────────────────
function heatmap_pair(t::Int, tag::String)
    clim = (
        min(minimum(truth_speed_all[:, :, t]), minimum(mean_speed_all[:, :, t])),
        max(maximum(truth_speed_all[:, :, t]), maximum(mean_speed_all[:, :, t])),
    )
    p_t = heatmap(xs_km, xs_km, transpose(truth_speed_all[:, :, t]);
                  title=@sprintf("Truth speed  t=%.1f h", data.model_time[t]),
                  xlabel="x (km)", ylabel="y (km)", c=:viridis, clims=clim)
    p_m = heatmap(xs_km, xs_km, transpose(mean_speed_all[:, :, t]);
                  title=@sprintf("Estimated speed  t=%.1f h", data.model_time[t]),
                  xlabel="x (km)", ylabel="y (km)", c=:viridis, clims=clim)
    plot(p_t, p_m; layout=(1,2), size=(1000, 420))
    savefig(joinpath(OUT, "velocity_heatmap_$(tag).png"))
end
heatmap_pair(1, "t0")
heatmap_pair(T_plus_1, "final")
println("Saved velocity_heatmap_t0.png and velocity_heatmap_final.png")

# ── Plot 3: |v|(t) at the on-cross-section sensor cell (i=21, j=20) ─────
# Same sensor index used in the β sensor-trail GIF in run10.
si, sj = 21, 20
truth_at_sensor = [truth_speed_all[si, sj, t] for t in 1:T_plus_1]
mean_at_sensor  = [mean_speed_all[si, sj, t]  for t in 1:T_plus_1]
p_sensor = plot(data.model_time, truth_at_sensor;
                lw=3, color=:black, label="truth",
                xlabel="time (h)", ylabel="|v| (m/yr)",
                title=@sprintf("Ice speed at sensor (i=%d,j=%d)", si, sj),
                legend=:topright)
plot!(p_sensor, data.model_time, mean_at_sensor;
      lw=2.5, color=:firebrick, linestyle=:dash, label="ensemble mean")
savefig(p_sensor, joinpath(OUT, "velocity_probe_sensor.png"))
println("Saved velocity_probe_sensor.png")

println("\nAll outputs in $OUT")
