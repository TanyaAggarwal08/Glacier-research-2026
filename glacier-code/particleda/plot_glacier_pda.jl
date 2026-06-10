# Generate the same diagnostic plots as particlefilteringwithiceflow.jl /
# particlefilteringnonlinear.jl, but driven by the ParticleDA HDF5 outputs.
#
# X-axes are MODEL TIME (seconds) — read from the `time_step` attribute in the
# HDF5's `parameters` group — so the plots are directly comparable to LowLevel
# regardless of how many filter steps the run had.
#
# Run: julia --project=test glacier-code/particleda/plot_glacier_pda.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5
using Statistics
using LinearAlgebra
ENV["GKSwstype"] = "100"
using Plots

# Optional first arg: directory containing glacier_obs.h5 + particle_da.h5.
# Plots will also be written into that directory.
const SAVE_DIR = if !isempty(ARGS) && isdir(ARGS[1])
    ARGS[1]
else
    joinpath("glacier-code", "particleda", "results")
end
const OBS_PATH = joinpath(SAVE_DIR, "glacier_obs.h5")
const DA_PATH  = joinpath(SAVE_DIR, "particle_da.h5")
@info "Reading from / writing to: $SAVE_DIR"

# ── Read time_step from the parameters group ────────────────────────────────
time_step = h5open(DA_PATH, "r") do f
    Float64(read(attributes(f["parameters"])["time_step"]))
end
@info "time_step from HDF5: $time_step s per filter step"

# ── Load truth, posterior mean, weights, observations ────────────────────────
beta_true = Vector{Matrix{Float64}}()
beta_est  = Vector{Matrix{Float64}}()
beta_var  = Vector{Matrix{Float64}}()
weights   = Vector{Vector{Float64}}()
obs_vec   = Vector{Vector{Float64}}()

# We load t0000 for the truth and PF mean so the *initial* divergence between
# them is visible on pointwise plots. (At t0000 PF mean ≈ prior, while truth is
# a random draw 20-30 % away — the filter then snaps to truth on the first
# update.) Weights at t0000 are NaN (uninitialised), so they're skipped.
h5open(OBS_PATH, "r") do fo
    state_keys = sort(collect(keys(fo["state"])))
    for k in state_keys
        push!(beta_true, read(fo["state"][k]["beta"]))
    end
    if haskey(fo, "observations")
        obs_keys = sort(collect(keys(fo["observations"])))[2:end]
        for k in obs_keys
            push!(obs_vec, vec(read(fo["observations"][k])))
        end
    end
end
h5open(DA_PATH, "r") do fd
    avg_keys = sort(collect(keys(fd["state_avg"])))
    for k in avg_keys
        push!(beta_est, read(fd["state_avg"][k]["beta"]))
        push!(beta_var, read(fd["state_var"][k]["beta"]))
    end
    w_keys = sort(collect(keys(fd["weights"])))[2:end]   # skip NaN
    for k in w_keys
        push!(weights, vec(read(fd["weights"][k])))
    end
end

# Two clocks: state arrays include t0000 (initial state), so their times start
# at 0. Filter arrays (weights, obs) skip t0000, so their times start at the
# first filter update (= time_step).
T_state  = min(length(beta_true), length(beta_est), length(beta_var))
T_filter = length(weights)
T_obs    = isempty(obs_vec) ? 0 : length(obs_vec)

beta_true = beta_true[1:T_state]
beta_est  = beta_est[1:T_state]
beta_var  = beta_var[1:T_state]
weights   = weights[1:T_filter]
if !isempty(obs_vec); obs_vec = obs_vec[1:T_obs]; end

@info "T_state = $T_state (incl. t0000), T_filter = $T_filter, T_obs = $T_obs"

ny, nx = size(beta_true[1])
model_time        = collect(0:T_state-1) .* time_step   # state clock (starts at 0)
model_time_filter = collect(1:T_filter) .* time_step    # filter clock (starts at time_step)
model_time_obs    = collect(1:T_obs)    .* time_step    # obs clock

# ── Grid + sensors ───────────────────────────────────────────────────────────
grid_x = h5read(DA_PATH, "grid_coordinates/x")
grid_y = h5read(DA_PATH, "grid_coordinates/y")
sx     = h5read(DA_PATH, "station_coordinates/x")
sy     = h5read(DA_PATH, "station_coordinates/y")

# Sensor linear indices: column-major (idx = (col-1)*ny + row).
dx_grid = grid_x[2] - grid_x[1]
dy_grid = grid_y[2] - grid_y[1]
sensor_indices = Int[]
for (xs, ys) in zip(sx, sy)
    i = round(Int, xs / dx_grid) + 1   # x column
    j = round(Int, ys / dy_grid) + 1   # y row
    push!(sensor_indices, (i - 1) * ny + j)
end

# ── Surrogate ux (matches model exactly) ────────────────────────────────────
ux_field(β) = 1.0e3 ./ (β .+ 1e-6)

# ── Error fields + colour limits ────────────────────────────────────────────
beta_err = [beta_est[t] .- beta_true[t] for t in 1:T_state]
climβ = (0.0, 2000.0)
emax = maximum(maximum(abs.(b)) for b in beta_err)
climE = (-emax, emax)

# ──────────────────────────────────────────────────────────────────────────────
#  1. GIFs — state evolution
# ──────────────────────────────────────────────────────────────────────────────
anim_true = @animate for t in 1:T_state
    tlabel = round(Int, model_time[t])
    heatmap(grid_x, grid_y, beta_true[t]; title="True β (t = $tlabel s)",
            aspect_ratio=1, clims=climβ, c=:Blues, colorbar=true,
            xlabel="x (m)", ylabel="y (m)")
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim_true, joinpath(SAVE_DIR, "true_beta.gif"), fps=6)

anim_est = @animate for t in 1:T_state
    tlabel = round(Int, model_time[t])
    heatmap(grid_x, grid_y, beta_est[t]; title="PF Mean β (t = $tlabel s)",
            aspect_ratio=1, clims=climβ, c=:Blues, colorbar=true,
            xlabel="x (m)", ylabel="y (m)")
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim_est, joinpath(SAVE_DIR, "est_beta.gif"), fps=6)

anim_err = @animate for t in 1:T_state
    tlabel = round(Int, model_time[t])
    heatmap(grid_x, grid_y, beta_err[t]; title="β error = est − true (t = $tlabel s)",
            aspect_ratio=1, clims=climE, c=:balance, colorbar=true,
            xlabel="x (m)", ylabel="y (m)")
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim_err, joinpath(SAVE_DIR, "error_beta.gif"), fps=6)

# ──────────────────────────────────────────────────────────────────────────────
#  2. Final β error heatmap
# ──────────────────────────────────────────────────────────────────────────────
p_final = heatmap(grid_x, grid_y, beta_err[end]; title="Final β Error",
                  aspect_ratio=1, clims=climE, c=:balance, colorbar=true,
                  xlabel="x (m)", ylabel="y (m)")
scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
savefig(p_final, joinpath(SAVE_DIR, "final_beta_error.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  3. RMSE — global (β) and at sensors (ux)
# ──────────────────────────────────────────────────────────────────────────────
rmse_beta = zeros(T_state)
rmse_vel  = zeros(T_state)
for t in 1:T_state
    rmse_beta[t] = sqrt(mean((beta_est[t] .- beta_true[t]).^2))
    ux_true_flat = vec(ux_field(beta_true[t]))
    ux_est_flat  = vec(ux_field(beta_est[t]))
    rmse_vel[t]  = sqrt(mean((ux_est_flat[sensor_indices] .- ux_true_flat[sensor_indices]).^2))
end
savefig(plot(model_time, rmse_beta; xlabel="Model time (s)", ylabel="RMSE(β)",
             title="Global RMSE of β", legend=false, lw=2),
        joinpath(SAVE_DIR, "rmse_beta.png"))
savefig(plot(model_time, rmse_vel; xlabel="Model time (s)", ylabel="RMSE(ux at sensors)",
             title="Sensor velocity RMSE", legend=false, lw=2),
        joinpath(SAVE_DIR, "rmse_vel.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  4. Single-point time series — truth, PF mean, AND observation (at sensor 1)
# ──────────────────────────────────────────────────────────────────────────────
target = sensor_indices[1]
truth_pt = [vec(beta_true[t])[target] for t in 1:T_state]
est_pt   = [vec(beta_est[t])[target]  for t in 1:T_state]
# Convert observed ux back to β so it lives on the same axis: β = 1000 / ux.
# Obs has its own clock (skips t0000) so we plot on model_time_obs, not model_time.
obs_pt_β = fill(NaN, T_obs)
if !isempty(obs_vec)
    for t in 1:T_obs
        ux_at_sensor1 = obs_vec[t][1]
        if ux_at_sensor1 > 0
            obs_pt_β[t] = 1000.0 / ux_at_sensor1
        end
    end
end
p_pt = plot(model_time, truth_pt; label="True β", lw=2,
            title="β at first sensor — truth, PF mean, observation",
            xlabel="Model time (s)", ylabel="β (Pa·s/m)")
plot!(model_time, est_pt; label="PF Mean", lw=2, linestyle=:dash, color=:orange)
scatter!(model_time_obs, obs_pt_β; label="Obs (β-equivalent)", markersize=3,
         color=:black, marker=:x, alpha=0.6)
savefig(p_pt, joinpath(SAVE_DIR, "beta_pointwise.png"))

# Normalised view: β_norm = (β - 1000) / 500.  Maps prior mean → 0, prior
# amplitude → ±1. Lets the small initial PF-mean / truth gap show up clearly
# instead of getting lost in a 500-1500 axis.
β_NORM_CENTRE = 1000.0
β_NORM_AMP    = 500.0
normβ(x) = (x .- β_NORM_CENTRE) ./ β_NORM_AMP
truth_pt_n = normβ(truth_pt)
est_pt_n   = normβ(est_pt)
obs_pt_n   = normβ(obs_pt_β)
p_pt_n = plot(model_time, truth_pt_n; label="True β (norm)", lw=2,
              title="β at first sensor — normalised to (β − 1000) / 500",
              xlabel="Model time (s)", ylabel="β_norm   (−1 to +1 = prior amp)",
              ylim=(-1.2, 1.2))
plot!(p_pt_n, model_time, est_pt_n; label="PF Mean (norm)", lw=2,
      linestyle=:dash, color=:orange)
scatter!(p_pt_n, model_time_obs, obs_pt_n; label="Obs (norm)", markersize=3,
         color=:black, marker=:x, alpha=0.6)
hline!(p_pt_n, [0.0]; linestyle=:dot, color=:gray, label="prior mean")
savefig(p_pt_n, joinpath(SAVE_DIR, "beta_pointwise_normalised.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  5a. ux at first sensor — observations in their NATIVE space (no inversion)
#      This is the view the filter actually sees. The β-equivalent plot above
#      amplifies the noise through the 1/β nonlinearity, so this plot is the
#      honest one for judging how noisy observations really are.
# ──────────────────────────────────────────────────────────────────────────────
ux_true_pt = [vec(ux_field(beta_true[t]))[target] for t in 1:T_state]
ux_est_pt  = [vec(ux_field(beta_est[t]))[target]  for t in 1:T_state]
# Obs has its own clock; obs_vec[t][2] picks sensor 2's measurement (user choice).
ux_obs_pt  = isempty(obs_vec) ? fill(NaN, T_obs) :
             [length(obs_vec[t]) >= 2 ? obs_vec[t][2] : NaN for t in 1:T_obs]
p_ux = plot(model_time, ux_true_pt; label="True ux", lw=2,
            title="ux at first sensor — truth, PF mean, observation",
            xlabel="Model time (s)", ylabel="ux (model units)")
plot!(p_ux, model_time, ux_est_pt; label="PF Mean ux", lw=2,
      linestyle=:dash, color=:orange)
scatter!(p_ux, model_time_obs, ux_obs_pt; label="Obs (ux native)", markersize=3,
         color=:black, marker=:x, alpha=0.6)
savefig(p_ux, joinpath(SAVE_DIR, "ux_first_sensor.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  5. β evolution at multiple non-sensor spatial points (a "few selected
#     grid points" view, matching beta_point_evolution.png in LowLevel)
# ──────────────────────────────────────────────────────────────────────────────
points = [(10, 10), (10, 30), (30, 10), (30, 30), (20, 20)]   # (i_col, j_row)
p_multi = plot(; xlabel="Model time (s)", ylabel="β (Pa·s/m)",
               title="β at five fixed grid points (solid=truth, dash=PF mean)",
               legend=:outertopright)
palette = [:steelblue, :darkorange, :seagreen, :purple, :firebrick]
for (k, (ic, jr)) in enumerate(points)
    truth_series = [beta_true[t][jr, ic] for t in 1:T_state]
    est_series   = [beta_est[t][jr, ic]  for t in 1:T_state]
    plot!(p_multi, model_time, truth_series;
          color=palette[k], lw=2, label="(i=$ic, j=$jr) truth")
    plot!(p_multi, model_time, est_series;
          color=palette[k], lw=2, linestyle=:dash, label="(i=$ic, j=$jr) est")
end
savefig(p_multi, joinpath(SAVE_DIR, "beta_multi_point.png"))

# Normalised version of the multi-point plot.
p_multi_n = plot(; xlabel="Model time (s)",
                 ylabel="β_norm = (β − 1000) / 500",
                 title="β at five fixed points (normalised)  solid=truth, dash=PF mean",
                 legend=:outertopright, ylim=(-1.5, 1.5))
hline!(p_multi_n, [0.0]; linestyle=:dot, color=:gray, label="prior mean")
for (k, (ic, jr)) in enumerate(points)
    truth_series_n = [(beta_true[t][jr, ic] - β_NORM_CENTRE) / β_NORM_AMP for t in 1:T_state]
    est_series_n   = [(beta_est[t][jr, ic]  - β_NORM_CENTRE) / β_NORM_AMP for t in 1:T_state]
    plot!(p_multi_n, model_time, truth_series_n;
          color=palette[k], lw=2, label="(i=$ic, j=$jr) truth")
    plot!(p_multi_n, model_time, est_series_n;
          color=palette[k], lw=2, linestyle=:dash, label="(i=$ic, j=$jr) est")
end
savefig(p_multi_n, joinpath(SAVE_DIR, "beta_multi_point_normalised.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  6. ESS over time
# ──────────────────────────────────────────────────────────────────────────────
Np = length(weights[1])
ess = [begin w = weights[t]; wn = w ./ sum(w); 1.0 / sum(wn .^ 2) end for t in 1:T_filter]
p_ess = plot(model_time_filter, ess; xlabel="Model time (s)", ylabel="ESS",
             title="Effective Sample Size", lw=2, marker=:circle,
             markersize=3, ylim=(0, Np), legend=:topright, label="ESS")
hline!(p_ess, [0.5 * Np]; linestyle=:dash, color=:red, label="0.5·Np")
savefig(p_ess, joinpath(SAVE_DIR, "ess_evolution.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  7. Weight diagnostics: max + min (extremes), variance, entropy
# ──────────────────────────────────────────────────────────────────────────────
maxw = [maximum(w ./ sum(w)) for w in weights]
minw = [minimum(w ./ sum(w)) for w in weights]
varw = [var(w ./ sum(w))     for w in weights]
ent  = [begin wn = w ./ sum(w); -sum(wn .* log.(wn .+ eps())) end for w in weights]

p_extr = plot(model_time_filter, maxw; xlabel="Model time (s)", ylabel="weight",
              title="Particle weight extremes", lw=2, label="max(weight)",
              legend=:topright, ylim=(0, 1))
plot!(p_extr, model_time_filter, minw; lw=2, label="min(weight)", color=:darkorange)
savefig(p_extr, joinpath(SAVE_DIR, "weight_extremes.png"))

savefig(plot(model_time_filter, varw; xlabel="Model time (s)",
             ylabel="Var(normalised weights)",
             title="Weight variance", lw=2, legend=false),
        joinpath(SAVE_DIR, "weight_variance.png"))

p_ent = plot(model_time_filter, ent; xlabel="Model time (s)", ylabel="Entropy",
             title="Weight entropy (higher = healthier)", lw=2, label="entropy")
hline!(p_ent, [log(Np)]; linestyle=:dash, color=:red, label="log(Np) = uniform")
savefig(p_ent, joinpath(SAVE_DIR, "weight_entropy.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  8. Posterior variance of β (proxy for ensemble spread)
# ──────────────────────────────────────────────────────────────────────────────
mean_var = [mean(beta_var[t]) for t in 1:T_state]
savefig(plot(model_time, mean_var; xlabel="Model time (s)",
             ylabel="mean posterior Var(β)",
             title="Posterior variance of β (mean over grid)",
             lw=2, legend=false),
        joinpath(SAVE_DIR, "particle_variance.png"))

# ──────────────────────────────────────────────────────────────────────────────
#  9. Snapshot collage — truth / estimate / error at three model times
# ──────────────────────────────────────────────────────────────────────────────
function _three_snapshots()
    times = [round(Int, T_state * 0.05), round(Int, T_state * 0.5), T_state]
    times = [max(1, t) for t in times]
    plots = []
    for t in times
        tlabel = round(Int, model_time[t])
        h_t = heatmap(grid_x, grid_y, beta_true[t]; title="Truth t=$tlabel s",
                      aspect_ratio=1, clims=climβ, c=:Blues,
                      xlabel="x", ylabel="y", colorbar=false)
        h_e = heatmap(grid_x, grid_y, beta_est[t]; title="PF mean t=$tlabel s",
                      aspect_ratio=1, clims=climβ, c=:Blues,
                      xlabel="x", ylabel="y", colorbar=false)
        h_d = heatmap(grid_x, grid_y, beta_err[t]; title="Error t=$tlabel s",
                      aspect_ratio=1, clims=climE, c=:balance,
                      xlabel="x", ylabel="y", colorbar=false)
        push!(plots, h_t); push!(plots, h_e); push!(plots, h_d)
    end
    plot(plots...; layout=(3, 3), size=(1200, 1100))
end
savefig(_three_snapshots(), joinpath(SAVE_DIR, "snapshots_collage.png"))

println("\nSaved to: ", abspath(SAVE_DIR))
for f in (
    "true_beta.gif", "est_beta.gif", "error_beta.gif",
    "final_beta_error.png",
    "rmse_beta.png", "rmse_vel.png",
    "beta_pointwise.png", "beta_pointwise_normalised.png",
    "ux_first_sensor.png",
    "beta_multi_point.png", "beta_multi_point_normalised.png",
    "ess_evolution.png",
    "weight_extremes.png", "weight_variance.png", "weight_entropy.png",
    "particle_variance.png",
    "snapshots_collage.png",
)
    println("  - ", f)
end
