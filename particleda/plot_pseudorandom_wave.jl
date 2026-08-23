# Visualisations for the pseudo-random wave tracking run.
#
# Produces:
#   1) prior_pair.png         — truth initial field vs background initial field
#   2) crosssection_anim.gif   — β(x) along y=row_mid, animated over t
#   3) crosssection_final.png  — final cross-section snapshot
#   4) sensor_trail_anim.gif   — β(t) at one sensor, all particles + truth + obs
#   5) sensor_trail.png        — static version of the sensor trail
#   6) ess_tracking.png        — ESS over time
#   7) rmse_beta.png           — global RMSE of ensemble mean vs truth over time
#  10) rmse_beta_compare_*.png — automatic comparison plots when reference runs exist
#   8) true_beta_obs_anim.gif  — true β heatmap with observation markers
#   9) est_beta_obs_anim.gif   — estimated β heatmap with observation markers
#
# Run:
#   julia --project=test glacier-code/particleda/plot_pseudorandom_wave.jl
#   julia --project=test glacier-code/particleda/plot_pseudorandom_wave.jl glacier-code/particleda/results/run12_pseudorandom_wave_30obs

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const DEFAULT_OUT = joinpath("glacier-code", "particleda", "results",
                             "run11_pseudorandom_wave")
const OUT = if !isempty(ARGS) && isdir(ARGS[1])
    ARGS[1]
else
    DEFAULT_OUT
end
mkpath(OUT)
const TRACK = let
    local_track = joinpath(OUT, "tracking.h5")
    external = joinpath("/Volumes/ZX20/USRA 2026", basename(OUT), "tracking.h5")
    isfile(local_track) ? local_track : external
end

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
        beta_truth_prior = read(f["beta_truth_prior"]),
        beta_background_prior = read(f["beta_background_prior"]),
        dt           = read(attributes(f["parameters"])["time_step"]),
        nx           = read(attributes(f["parameters"])["nx"]),
        ny           = read(attributes(f["parameters"])["ny"]),
        obs_space    = haskey(attributes(f["parameters"]), "obs_space") ?
                       String(read(attributes(f["parameters"])["obs_space"])) :
                       "velocity",
    )
end

T_plus_1 = size(data.truth, 3)
T        = T_plus_1 - 1
nprt     = size(data.w_raw, 1)
K        = length(data.track_idx)
t_axis   = collect(0:T) .* data.dt

println("Loaded: T=$T, nprt=$nprt, K_track=$K")

# Common helpers
all_prior_vals = vcat(vec(data.beta_truth_prior), vec(data.beta_background_prior))
clim_prior = (quantile(all_prior_vals, 0.02), quantile(all_prior_vals, 0.98))
clim_beta = (quantile(vcat(vec(data.truth), vec(data.mean)), 0.02),
             quantile(vcat(vec(data.truth), vec(data.mean)), 0.98))
particle_colors = palette(:tab20, max(K, 20))

# Truth/background prior comparison
p_truth = heatmap(data.beta_truth_prior;
                  clims=clim_prior, c=:viridis, aspect_ratio=1,
                  title="Truth initial pseudo-random wave",
                  xlabel="x cell", ylabel="y cell")
p_bg = heatmap(data.beta_background_prior;
               clims=clim_prior, c=:viridis, aspect_ratio=1,
               title="Background initial wave",
               xlabel="x cell", ylabel="y cell")
p_prior = plot(p_truth, p_bg; layout=(1, 2), size=(1100, 420))
savefig(p_prior, joinpath(OUT, "prior_pair.png"))

# Cross-section metadata
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

obs_to_beta(obs_val, obs_space) = obs_space == "velocity" ? 1000.0 ./ obs_val :
                                  obs_space == "log_velocity" ? 1000.0 ./ exp.(obs_val) :
                                  exp.(obs_val)
obs_label(obs_space) = obs_space == "velocity" ? "observation (β-eq)" :
                       obs_space == "log_velocity" ? "log-ux observation (β-eq)" :
                       "log-β observation (β-eq)"

β_obs_on_row = [obs_to_beta(data.obs[k, :], data.obs_space) for k in sensors_on_row]
grid_x_km = collect(0:data.nx-1) .* (160.0 / data.nx)
grid_y_km = collect(0:data.ny-1) .* (160.0 / data.ny)
sensor_x_km = data.sensor_x ./ 1000.0
sensor_y_km = data.sensor_y ./ 1000.0

# Field animations with observation markers
anim_true_field = @animate for t in 1:T_plus_1
    p = heatmap(grid_x_km, grid_y_km, data.truth[:, :, t];
                clims=clim_beta, c=:Blues, aspect_ratio=1,
                xlabel="x (km)", ylabel="y (km)",
                title=@sprintf("True β field with observations at t = %.1f h",
                               (t-1)*data.dt/3600),
                colorbar_title="β")
    if t >= 2
        scatter!(p, sensor_x_km, sensor_y_km;
                 marker=:circle, ms=4, mc=:black, msc=:red, msw=1.5,
                 label="Observations")
    end
end
gif(anim_true_field, joinpath(OUT, "true_beta_obs_anim.gif"), fps=6)

anim_est_field = @animate for t in 1:T_plus_1
    p = heatmap(grid_x_km, grid_y_km, data.mean[:, :, t];
                clims=clim_beta, c=:Blues, aspect_ratio=1,
                xlabel="x (km)", ylabel="y (km)",
                title=@sprintf("Estimated β field with observations at t = %.1f h",
                               (t-1)*data.dt/3600),
                colorbar_title="β")
    if t >= 2
        scatter!(p, sensor_x_km, sensor_y_km;
                 marker=:circle, ms=4, mc=:black, msc=:red, msw=1.5,
                 label="Observations")
    end
end
gif(anim_est_field, joinpath(OUT, "est_beta_obs_anim.gif"), fps=6)

# Cross-section animation
anim = @animate for t in 1:T_plus_1
    p = plot(xs_km, data.truth[row_mid, :, t];
             label="truth", lw=3, color=:black,
             xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("Pseudo-random wave  y=%.0f km  t=%.1f h",
                            row_y_km, (t-1)*data.dt/3600),
             ylim=(minimum(clim_beta), maximum(clim_beta)),
             legend=:topright)
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
                     label=(i_s==1 && t==2 ? obs_label(data.obs_space) : false))
        end
    end
end
gif(anim, joinpath(OUT, "crosssection_anim.gif"), fps=6)

p_final = plot(xs_km, data.truth[row_mid, :, end];
               label="truth", lw=3, color=:black,
               xlabel="x (km)", ylabel="β (Pa·s/m)",
               title=@sprintf("Final cross-section  y=%.0f km", row_y_km),
               ylim=(minimum(clim_beta), maximum(clim_beta)), legend=:topright)
plot!(p_final, xs_km, data.mean[row_mid, :, end];
      label="ensemble mean", lw=2, linestyle=:dash, color=:steelblue)
for k in 1:K
    plot!(p_final, xs_km, data.particles[row_mid, :, k, end];
          label=k==1 ? "tracked particles" : false,
          lw=1, color=particle_colors[k], alpha=0.7)
end
for (i_s, sx_km) in enumerate(sensor_x_km_row)
    vline!(p_final, [sx_km]; color=:red, linestyle=:dot, lw=1.5,
           label=(i_s==1 ? "sensor x" : false))
    scatter!(p_final, [sx_km], [β_obs_on_row[i_s][end]];
             marker=:xcross, ms=10, msw=2.5, mc=:red,
             label=(i_s==1 ? "obs (β-eq) at final step" : false))
end
savefig(p_final, joinpath(OUT, "crosssection_final.png"))

# Sensor trail animation + static snapshot.
sensor_i, sensor_j = 21, 20
particles_at_sensor = h5open(TRACK, "r") do f
    f["particles_all"]["beta"][sensor_j, sensor_i, :, :]
end
truth_at_sensor = [data.truth[sensor_j, sensor_i, t] for t in 1:T_plus_1]
mean_at_sensor  = [data.mean[sensor_j, sensor_i, t]  for t in 1:T_plus_1]
sensor_obs_k = sensors_on_row[1]
β_obs_at_sensor = obs_to_beta(data.obs[sensor_obs_k, :], data.obs_space)
t_axis_h = collect(0:T) .* data.dt ./ 3600

anim_trail = @animate for T_now in 1:T_plus_1
    t_hist = t_axis_h[1:T_now]
    p = plot(; xlabel="time (h)", ylabel="β (Pa·s/m)",
             title=@sprintf("β(t) at sensor (i=%d,j=%d)  t=%.1f h",
                            sensor_i, sensor_j, t_axis_h[T_now]),
             xlim=(0, t_axis_h[end]), ylim=(minimum(clim_beta), maximum(clim_beta)),
             legend=:topright)
    for ip in 1:nprt
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
                 label=obs_label(data.obs_space))
    end
end
gif(anim_trail, joinpath(OUT, "sensor_trail_anim.gif"), fps=6)

p_trail = plot(; xlabel="time (h)", ylabel="β (Pa·s/m)",
               title=@sprintf("β(t) at sensor (i=%d,j=%d)", sensor_i, sensor_j),
               xlim=(0, t_axis_h[end]), ylim=(minimum(clim_beta), maximum(clim_beta)),
               legend=:topright)
for ip in 1:nprt
    plot!(p_trail, t_axis_h, particles_at_sensor[ip, :];
          label=false, lw=0.4, color=:gray, alpha=0.05)
end
plot!(p_trail, t_axis_h, mean_at_sensor;
      label="ensemble mean", lw=2.5, linestyle=:dash, color=:steelblue)
plot!(p_trail, t_axis_h, truth_at_sensor;
      label="truth", lw=3, color=:black)
scatter!(p_trail, t_axis_h[2:end], β_obs_at_sensor;
         marker=:xcross, ms=6, msw=1.5, mc=:red,
         label=obs_label(data.obs_space))
savefig(p_trail, joinpath(OUT, "sensor_trail.png"))

# ESS plot
ww = t_axis_h[2:end]
p_ess = plot(ww, data.ess; xlabel="time (h)", ylabel="ESS",
             title="Pseudo-random wave — Effective Sample Size",
             lw=2, color=:purple, ylim=(0, nprt))
hline!(p_ess, [0.5*nprt]; linestyle=:dash, color=:gray, label="0.5·N")
savefig(p_ess, joinpath(OUT, "ess_tracking.png"))

# RMSE plot
rmse_beta = zeros(T_plus_1)
for t in 1:T_plus_1
    rmse_beta[t] = sqrt(mean((data.mean[:, :, t] .- data.truth[:, :, t]) .^ 2))
end
p_rmse = plot(t_axis_h, rmse_beta;
              xlabel="time (h)", ylabel="RMSE(β)",
              title="Pseudo-random wave — Global RMSE over time",
              lw=2.5, color=:darkgreen, legend=false)
savefig(p_rmse, joinpath(OUT, "rmse_beta.png"))

function add_rmse_comparison(ref_out::AbstractString, ref_label::AbstractString, save_name::AbstractString)
    ref_track = joinpath(ref_out, "tracking.h5")
    if !isfile(ref_track)
        return
    end
    ref = h5open(ref_track, "r") do f
        (
            truth = read(f["truth"]["beta"]),
            mean  = read(f["ensemble_mean"]["beta"]),
            dt    = read(attributes(f["parameters"])["time_step"]),
        )
    end
    ref_T_plus_1 = size(ref.truth, 3)
    ref_t_axis_h = collect(0:ref_T_plus_1-1) .* ref.dt ./ 3600
    ref_rmse = [sqrt(mean((ref.mean[:, :, t] .- ref.truth[:, :, t]) .^ 2))
                for t in 1:ref_T_plus_1]
    p_cmp = plot(ref_t_axis_h, ref_rmse;
                 xlabel="time (h)", ylabel="RMSE(β)",
                 title="Pseudo-random wave — RMSE comparison",
                 lw=2.5, color=:steelblue, label=ref_label)
    plot!(p_cmp, t_axis_h, rmse_beta;
          lw=2.5, color=:darkgreen, label=basename(OUT))
    savefig(p_cmp, joinpath(OUT, save_name))
end

if OUT != DEFAULT_OUT
    add_rmse_comparison(DEFAULT_OUT, "run11: 10 obs", "rmse_beta_compare_vs_run11.png")
end

run12_out = joinpath("glacier-code", "particleda", "results", "run12_pseudorandom_wave_30obs")
if OUT != run12_out
    add_rmse_comparison(run12_out, "run12: 30 obs assimilated", "rmse_beta_compare_vs_run12.png")
end

println("All outputs in $OUT")
