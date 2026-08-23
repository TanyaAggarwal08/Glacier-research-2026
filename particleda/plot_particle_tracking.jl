# Visualisations of per-particle behaviour from tracking.h5.
#
# Produces:
#   1) crosssection_anim.gif  — β(x) along y=row_mid, animated over t.
#                                Truth thick black, ensemble mean dashed,
#                                K tracked particles as thin coloured lines.
#   2) spaghetti_probe_cells.png  — at 4 probe cells, β(t) for ALL particles
#                                   as thin α-blended lines + truth + mean.
#   3) particle_snapshots.png — at 4 timesteps, side-by-side heatmaps of
#                               tracked particles' β fields next to truth.
#   4) weight_evolution.png   — w(t) for each tracked particle (log scale).
#
# Run: julia --project=test glacier-code/particleda/plot_particle_tracking.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const TRACK = "/Volumes/ZX20/USRA 2026/run07_particle_tracking/tracking.h5"
const OUT   = joinpath("glacier-code", "particleda", "results",
                       "run07_particle_tracking")
mkpath(OUT)

# ---- Load ----
data = h5open(TRACK, "r") do f
    (
        truth        = read(f["truth"]["beta"]),                # (ny, nx, T+1)
        mean         = read(f["ensemble_mean"]["beta"]),
        particles    = read(f["particles_full"]["beta"]),       # (ny, nx, K, T+1)
        track_idx    = read(f["particles_full"]["indices"]),
        probe_beta   = read(f["probe_cells"]["beta"]),          # (nprt, T+1, n_probes)
        probe_idx    = read(f["probe_cells"]["indices"]),
        probe_ij     = read(f["probe_cells"]["ij_pairs"]),      # (2, n_probes)
        w_raw        = read(f["weights"]["raw"]),               # (nprt, T)
        ess          = read(f["weights"]["ess"]),
        sensor_idx   = read(f["sensor_indices"]),
        sensor_x     = read(f["sensor_x"]),                     # (n_obs,) metres
        sensor_y     = read(f["sensor_y"]),
        obs          = read(f["observations"]),                 # (n_obs, T) noisy ux
        dt           = read(attributes(f["parameters"])["time_step"]),
        nx           = read(attributes(f["parameters"])["nx"]),
        ny           = read(attributes(f["parameters"])["ny"]),
    )
end
T_plus_1 = size(data.truth, 3)
T        = T_plus_1 - 1
nprt     = size(data.w_raw, 1)
K        = length(data.track_idx)
t_axis   = collect(0:T) .* data.dt

println("Loaded: T=$T, nprt=$nprt, K_track=$K")

# Common helpers
clim_beta = (300.0, 1700.0)
particle_colors = palette(:tab20, max(K, 20))

# ─── 1) Cross-section animation along y=row_mid ────────────────────────────
row_mid = data.ny ÷ 2           # j index = 20  ⇒  y = 76 km on a 40-cell grid
row_y_km = (row_mid - 1) * (160.0 / data.ny)
xs_km = collect(0:data.nx-1) .* (160.0 / data.nx)
dy_m  = 160000.0 / data.ny

# Find sensors that sit ON this cross-section row (j == row_mid).
# We picked stations_crosssection.txt so exactly one does, at x = 80 km.
sensors_on_row = Int[]    # indices into the n_obs sensor list
for k in eachindex(data.sensor_y)
    j_sensor = clamp(round(Int, data.sensor_y[k] / dy_m) + 1, 1, data.ny)
    if j_sensor == row_mid
        push!(sensors_on_row, k)
    end
end
sensor_x_km_row = data.sensor_x[sensors_on_row] ./ 1000.0
println("Sensors on cross-section row y=$(round(row_y_km, digits=1)) km: ",
        length(sensors_on_row), " at x_km = ", round.(sensor_x_km_row, digits=1))

# β-equivalent observation at each on-row sensor:  β_obs = 1000 / obs_ux.
# obs has shape (n_obs, T). observations are taken AFTER step t (i.e. valid
# at the "+1" snapshot index), so we plot them at t_plus_1 frames 2..T+1.
β_obs_on_row = [1000.0 ./ data.obs[k, :] for k in sensors_on_row]  # vec of T-vectors

anim = @animate for t in 1:T_plus_1
    p = plot(xs_km, data.truth[row_mid, :, t];
             label="truth", lw=3, color=:black,
             xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("Cross-section y=%.0f km,  t=%.1f h",
                            row_y_km, (t-1)*data.dt/3600),
             ylim=(500, 3500), legend=:topright)
    plot!(p, xs_km, data.mean[row_mid, :, t];
          label="ensemble mean", lw=2, linestyle=:dash, color=:steelblue)
    for k in 1:K
        plot!(p, xs_km, data.particles[row_mid, :, k, t];
              label=k==1 ? "tracked particles" : false,
              lw=1, color=particle_colors[k], alpha=0.7)
    end
    # Mark every on-row sensor with a vertical line
    for (i_s, sx_km) in enumerate(sensor_x_km_row)
        vline!(p, [sx_km]; color=:red, linestyle=:dot, lw=1.5,
               label=(i_s==1 ? "sensor x" : false))
        # Observation marker (β-equivalent) at this sensor, for the current t
        if t >= 2     # obs valid from filter step 1 onward → snapshot t = 2
            obs_val = β_obs_on_row[i_s][t-1]
            scatter!(p, [sx_km], [obs_val];
                     marker=:xcross, ms=10, msw=2.5, mc=:red,
                     label=(i_s==1 && t==2 ? "observation (β-equivalent)" : false))
        end
    end
end
gif(anim, joinpath(OUT, "crosssection_anim.gif"), fps=6)
println("Saved crosssection_anim.gif")

# ─── 1b) ALL-particles cross-section animation ─────────────────────────────
# Plot the full ensemble on the same row. To keep memory sane we only pull
# the row_mid slice from the big particles_all dataset (40 × NPRT × T+1).
row_slice = h5open(TRACK, "r") do f
    f["particles_all"]["beta"][row_mid, :, :, :]   # → (nx, NPRT, T+1)
end
nprt_all = size(row_slice, 2)
println("Loaded row slice for all $nprt_all particles: ", size(row_slice))

anim_all = @animate for t in 1:T_plus_1
    p = plot(; xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("ALL %d particles, cross-section y=%.0f km,  t=%.1f h",
                            nprt_all, row_y_km, (t-1)*data.dt/3600),
             ylim=(500, 3500), legend=:topright)
    for ip in 1:nprt_all
        plot!(p, xs_km, row_slice[:, ip, t];
              label=false, lw=0.4, color=:gray, alpha=0.05)
    end
    plot!(p, xs_km, data.mean[row_mid, :, t];
          label="ensemble mean", lw=2.5, linestyle=:dash, color=:steelblue)
    plot!(p, xs_km, data.truth[row_mid, :, t];
          label="truth", lw=3, color=:black)
    for (i_s, sx_km) in enumerate(sensor_x_km_row)
        vline!(p, [sx_km]; color=:red, linestyle=:dot, lw=1.5,
               label=(i_s==1 ? "sensor x" : false))
        if t >= 2
            obs_val = β_obs_on_row[i_s][t-1]
            scatter!(p, [sx_km], [obs_val];
                     marker=:xcross, ms=10, msw=2.5, mc=:red,
                     label=(i_s==1 && t==2 ? "observation" : false))
        end
    end
end
gif(anim_all, joinpath(OUT, "crosssection_anim_all_particles.gif"), fps=6)
println("Saved crosssection_anim_all_particles.gif")

# Also a static snapshot at t=T+1 for the doc
p_static = plot(xs_km, data.truth[row_mid, :, end];
                label="truth", lw=3, color=:black,
                xlabel="x (km)", ylabel="β (Pa·s/m)",
                title="Cross-section y=$(round(Int, row_mid*4))km at final time",
                ylim=(500, 3500))
plot!(p_static, xs_km, data.mean[row_mid, :, end];
      label="ensemble mean", lw=2, linestyle=:dash, color=:steelblue)
for k in 1:K
    plot!(p_static, xs_km, data.particles[row_mid, :, k, end];
          label=k==1 ? "tracked particles" : false,
          lw=1, color=particle_colors[k], alpha=0.7)
end
for (i_s, sx_km) in enumerate(sensor_x_km_row)
    vline!(p_static, [sx_km]; color=:red, linestyle=:dot, lw=1.5,
           label=(i_s==1 ? "sensor x" : false))
    scatter!(p_static, [sx_km], [β_obs_on_row[i_s][end]];
             marker=:xcross, ms=10, msw=2.5, mc=:red,
             label=(i_s==1 ? "obs (β-eq) at final step" : false))
end
savefig(p_static, joinpath(OUT, "crosssection_final.png"))

# ─── 1c) Temporal-trail GIF at the on-row sensor cell ──────────────────────
# At sensor (i=21, j=20), frame at time T shows β(t) for t ∈ [0, T] for all
# 1000 particles + truth + obs accumulated so far. Lets you see particles
# propagating through time instead of through space.
sensor_i, sensor_j = 21, 20
particles_at_sensor = h5open(TRACK, "r") do f
    f["particles_all"]["beta"][sensor_j, sensor_i, :, :]    # (NPRT, T+1)
end
println("Loaded sensor trail for all $(size(particles_at_sensor,1)) particles")

truth_at_sensor = [data.truth[sensor_j, sensor_i, t] for t in 1:T_plus_1]
mean_at_sensor  = [data.mean[sensor_j, sensor_i, t]  for t in 1:T_plus_1]
sensor_obs_k = sensors_on_row[1]
β_obs_at_sensor = 1000.0 ./ data.obs[sensor_obs_k, :]      # length T

t_axis_h = collect(0:T) .* data.dt ./ 3600

anim_trail = @animate for T_now in 1:T_plus_1
    t_hist = t_axis_h[1:T_now]
    p = plot(; xlabel="time (h)", ylabel="β (Pa·s/m)",
             title=@sprintf("β(t) at sensor (i=%d,j=%d) — frame t=%.1f h",
                            sensor_i, sensor_j, t_axis_h[T_now]),
             xlim=(0, t_axis_h[end]), ylim=(500, 3500), legend=:topright)
    # All particles, thin α-blended
    for ip in 1:nprt_all
        plot!(p, t_hist, particles_at_sensor[ip, 1:T_now];
              label=false, lw=0.4, color=:gray, alpha=0.05)
    end
    plot!(p, t_hist, mean_at_sensor[1:T_now];
          label="ensemble mean", lw=2.5, linestyle=:dash, color=:steelblue)
    plot!(p, t_hist, truth_at_sensor[1:T_now];
          label="truth", lw=3, color=:black)
    # Observations arrive at filter steps t=1..T → physical hours 1..T
    if T_now >= 2
        n_obs_shown = T_now - 1
        scatter!(p, t_axis_h[2:T_now], β_obs_at_sensor[1:n_obs_shown];
                 marker=:xcross, ms=6, msw=1.5, mc=:red,
                 label="observation (β-eq)")
    end
end
gif(anim_trail, joinpath(OUT, "sensor_trail_anim.gif"), fps=6)
println("Saved sensor_trail_anim.gif")

# ─── 2) Spaghetti per probe cell ───────────────────────────────────────────
sensor_set = Set(data.sensor_idx)
n_probes = size(data.probe_ij, 2)

panels = Plots.Plot[]
for k in 1:n_probes
    ic = data.probe_ij[1, k]; jr = data.probe_ij[2, k]
    flat = data.probe_idx[k]
    is_sensor = flat in sensor_set
    title_str = @sprintf("cell (%d,%d) — %s", ic, jr, is_sensor ? "SENSOR" : "no sensor")

    # Truth at this cell
    truth_at = [data.truth[jr, ic, t] for t in 1:T_plus_1]
    mean_at  = [data.mean[jr, ic, t]  for t in 1:T_plus_1]

    p = plot(; title=title_str, xlabel="time (h)", ylabel="β",
             ylim=(500, 3500), legend=(k==1 ? :topright : false))
    # All particles, thin
    for ip in 1:nprt
        plot!(p, t_axis ./ 3600, data.probe_beta[ip, :, k];
              label=false, lw=0.4, color=:gray, alpha=0.15)
    end
    plot!(p, t_axis ./ 3600, mean_at;  label="ensemble mean", lw=2.5, color=:steelblue, linestyle=:dash)
    plot!(p, t_axis ./ 3600, truth_at; label="truth",        lw=2.5, color=:black)
    push!(panels, p)
end
p_spag = plot(panels...; layout=(2, 2), size=(1300, 950))
savefig(p_spag, joinpath(OUT, "spaghetti_probe_cells.png"))
println("Saved spaghetti_probe_cells.png")

# ─── 3) Per-particle snapshot collage ──────────────────────────────────────
snap_times = [1, T÷3+1, 2*T÷3+1, T_plus_1]    # 4 snapshot times
n_snap = length(snap_times)

# For each snap time, K tracked particles + truth = K+1 heatmaps.
# Arrange as grid: rows = snap times, cols = truth + K particles
all_panels = Plots.Plot[]
for (row, t) in enumerate(snap_times)
    p_truth = heatmap(data.truth[:, :, t];
                      clim=clim_beta, colorbar=false,
                      title=@sprintf("t=%.1fh\nTRUTH", (t-1)*data.dt/3600),
                      titlefontsize=8, axis=false, aspect_ratio=1)
    push!(all_panels, p_truth)
    for k in 1:K
        p_k = heatmap(data.particles[:, :, k, t];
                      clim=clim_beta, colorbar=false,
                      title=@sprintf("p=%d\nw=%.3f", data.track_idx[k],
                                     t > T ? NaN : data.w_raw[data.track_idx[k], min(t,T)]),
                      titlefontsize=7, axis=false, aspect_ratio=1)
        push!(all_panels, p_k)
    end
end
p_snap = plot(all_panels...; layout=(n_snap, K+1), size=(180*(K+1), 850))
savefig(p_snap, joinpath(OUT, "particle_snapshots.png"))
println("Saved particle_snapshots.png")

# ─── 4) Weight evolution for tracked particles ─────────────────────────────
p_w = plot(; xlabel="time (h)", ylabel="weight (normalised, log)",
           title="Per-particle weight evolution (tracked subset)",
           yscale=:log10, legend=:outertopright, size=(1100, 600))
hline!(p_w, [1.0 / nprt]; label="uniform 1/N", linestyle=:dash, color=:gray)
ww = t_axis[2:end] ./ 3600  # weights are at filter steps t=1..T
n_top    = 5
n_median = 5
for (k, p_idx) in enumerate(data.track_idx)
    label_k = if k <= n_top
        "top-$k (w=$(round(data.w_raw[p_idx,end],digits=4)))"
    elseif k <= n_top + n_median
        "median-$(k - n_top) (w=$(round(data.w_raw[p_idx,end],digits=4)))"
    else
        "bottom-$(k - n_top - n_median) (w=$(round(data.w_raw[p_idx,end],digits=4)))"
    end
    plot!(p_w, ww, max.(data.w_raw[p_idx, :], 1e-10);
          label=label_k, lw=1.5, color=particle_colors[k])
end
savefig(p_w, joinpath(OUT, "weight_evolution.png"))
println("Saved weight_evolution.png")

# ─── 5) ESS over time (small bonus plot for context) ──────────────────────
p_ess = plot(ww, data.ess; xlabel="time (h)", ylabel="ESS",
             title="Effective Sample Size", lw=2, color=:purple, ylim=(0, nprt))
hline!(p_ess, [0.5*nprt]; linestyle=:dash, color=:gray, label="0.5·N")
hline!(p_ess, [nprt];     linestyle=:dot,  color=:gray, label="N (uniform)")
savefig(p_ess, joinpath(OUT, "ess_tracking.png"))

println("\nAll outputs in $OUT:")
for f in ("crosssection_anim.gif", "crosssection_final.png",
          "spaghetti_probe_cells.png", "particle_snapshots.png",
          "weight_evolution.png", "ess_tracking.png")
    println("  - $f")
end
