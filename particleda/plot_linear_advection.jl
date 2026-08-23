# Visualisations for the linear-advection run.
#
# Mirrors plot_particle_tracking.jl but reads from run08_linear_advection_run.

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const TRACK = "/Volumes/ZX20/USRA 2026/run08_linear_advection_run/tracking.h5"
const OUT   = joinpath("glacier-code", "particleda", "results",
                       "run08_linear_advection_run")
mkpath(OUT)

# Reuse the entire body of plot_particle_tracking.jl. Easiest: load it as a
# string and exec it (Julia 1.x: include is path-bound). We just copy the
# core blocks here so the new dir is self-contained.

data = h5open(TRACK, "r") do f
    (
        truth        = read(f["truth"]["beta"]),
        mean         = read(f["ensemble_mean"]["beta"]),
        particles    = read(f["particles_full"]["beta"]),
        track_idx    = read(f["particles_full"]["indices"]),
        probe_beta   = read(f["probe_cells"]["beta"]),
        probe_idx    = read(f["probe_cells"]["indices"]),
        probe_ij     = read(f["probe_cells"]["ij_pairs"]),
        w_raw        = read(f["weights"]["raw"]),
        ess          = read(f["weights"]["ess"]),
        sensor_idx   = read(f["sensor_indices"]),
        sensor_x     = read(f["sensor_x"]),
        sensor_y     = read(f["sensor_y"]),
        obs          = read(f["observations"]),
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
println("Loaded (LINEAR): T=$T, nprt=$nprt, K_track=$K")

clim_beta = (300.0, 1700.0)
particle_colors = palette(:tab20, max(K, 20))

# ─── Cross-section row metadata ─────────────────────────────────────────
row_mid = data.ny ÷ 2
row_y_km = (row_mid - 1) * (160.0 / data.ny)
xs_km = collect(0:data.nx-1) .* (160.0 / data.nx)
dy_m  = 160000.0 / data.ny

sensors_on_row = Int[]
for k in eachindex(data.sensor_y)
    j_sensor = clamp(round(Int, data.sensor_y[k] / dy_m) + 1, 1, data.ny)
    if j_sensor == row_mid
        push!(sensors_on_row, k)
    end
end
sensor_x_km_row = data.sensor_x[sensors_on_row] ./ 1000.0
println("Sensors on cross-section row y=$(round(row_y_km, digits=1)) km: ",
        length(sensors_on_row), " at x_km = ", round.(sensor_x_km_row, digits=1))

normβ(x) = (x .- 1000.0) ./ 500.0
β_obs_on_row = [1000.0 ./ data.obs[k, :] for k in sensors_on_row]

# ─── 1) Cross-section anim (K subset) ───────────────────────────────────
anim = @animate for t in 1:T_plus_1
    p = plot(xs_km, data.truth[row_mid, :, t];
             label="truth", lw=3, color=:black,
             xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("LINEAR adv  y=%.0f km  t=%.1f h",
                            row_y_km, (t-1)*data.dt/3600),
             ylim=(500, 3500), legend=:topright)
    plot!(p, xs_km, data.mean[row_mid, :, t];
          label="ensemble mean", lw=2, linestyle=:dash, color=:steelblue)
    for k in 1:K
        plot!(p, xs_km, data.particles[row_mid, :, k, t];
              label=k==1 ? "tracked particles" : false,
              lw=1, color=particle_colors[k], alpha=0.7)
    end
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
gif(anim, joinpath(OUT, "crosssection_anim.gif"), fps=6)
println("Saved crosssection_anim.gif")

# ─── 1b) Cross-section anim (ALL 1000 particles) ────────────────────────
row_slice = h5open(TRACK, "r") do f
    f["particles_all"]["beta"][row_mid, :, :, :]
end
nprt_all = size(row_slice, 2)
println("Loaded row slice for all $nprt_all particles: ", size(row_slice))

anim_all = @animate for t in 1:T_plus_1
    p = plot(; xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("LINEAR adv — ALL %d particles  y=%.0f km  t=%.1f h",
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

# ─── 1c) Temporal-trail GIF at on-row sensor ────────────────────────────
sensor_i, sensor_j = 21, 20
particles_at_sensor = h5open(TRACK, "r") do f
    f["particles_all"]["beta"][sensor_j, sensor_i, :, :]
end
truth_at_sensor = [data.truth[sensor_j, sensor_i, t] for t in 1:T_plus_1]
mean_at_sensor  = [data.mean[sensor_j, sensor_i, t]  for t in 1:T_plus_1]
sensor_obs_k = sensors_on_row[1]
β_obs_at_sensor = 1000.0 ./ data.obs[sensor_obs_k, :]
t_axis_h = collect(0:T) .* data.dt ./ 3600

anim_trail = @animate for T_now in 1:T_plus_1
    t_hist = t_axis_h[1:T_now]
    p = plot(; xlabel="time (h)", ylabel="β (Pa·s/m)",
             title=@sprintf("LINEAR adv — β(t) at sensor (i=%d,j=%d)  t=%.1f h",
                            sensor_i, sensor_j, t_axis_h[T_now]),
             xlim=(0, t_axis_h[end]), ylim=(500, 3500), legend=:topright)
    for ip in 1:nprt_all
        plot!(p, t_hist, particles_at_sensor[ip, 1:T_now];
              label=false, lw=0.4, color=:gray, alpha=0.05)
    end
    plot!(p, t_hist, mean_at_sensor[1:T_now];
          label="ensemble mean", lw=2.5, linestyle=:dash, color=:steelblue)
    plot!(p, t_hist, truth_at_sensor[1:T_now];
          label="truth", lw=3, color=:black)
    if T_now >= 2
        n_obs_shown = T_now - 1
        scatter!(p, t_axis_h[2:T_now], β_obs_at_sensor[1:n_obs_shown];
                 marker=:xcross, ms=6, msw=1.5, mc=:red,
                 label="observation (β-eq)")
    end
end
gif(anim_trail, joinpath(OUT, "sensor_trail_anim.gif"), fps=6)
println("Saved sensor_trail_anim.gif")

# ─── ESS plot for context ───────────────────────────────────────────────
ww = t_axis_h[2:end]
p_ess = plot(ww, data.ess; xlabel="time (h)", ylabel="ESS",
             title="LINEAR adv — Effective Sample Size",
             lw=2, color=:purple, ylim=(0, nprt))
hline!(p_ess, [0.5*nprt]; linestyle=:dash, color=:gray, label="0.5·N")
savefig(p_ess, joinpath(OUT, "ess_tracking.png"))

println("\nAll outputs in $OUT")
