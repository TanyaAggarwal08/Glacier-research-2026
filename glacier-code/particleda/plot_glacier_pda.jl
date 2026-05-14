# Generate the same diagnostic plots as particlefilteringwithiceflow.jl,
# but driven by the ParticleDA HDF5 outputs.
#
# Run: julia --project=test glacier-code/particleda/plot_glacier_pda.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5
using Statistics
using LinearAlgebra
ENV["GKSwstype"] = "100"
using Plots

const OBS_PATH = joinpath("glacier-code", "particleda", "results", "glacier_obs.h5")
const DA_PATH  = joinpath("glacier-code", "particleda", "results", "particle_da.h5")
const SAVE_DIR = joinpath("glacier-code", "particleda", "results")

# ── Load truth, posterior mean, weights ──────────────────────────────────────
beta_true = Vector{Matrix{Float64}}()
beta_est  = Vector{Matrix{Float64}}()
beta_var  = Vector{Matrix{Float64}}()
weights   = Vector{Vector{Float64}}()
obs_vec   = Vector{Vector{Float64}}()

h5open(OBS_PATH, "r") do fo
    h5open(DA_PATH, "r") do fd
        keys_state = sort(collect(keys(fo["state"])))
        T = length(keys_state)
        # PDA's state_avg has one more snapshot (t0000 = initial), so align by index
        keys_avg = sort(collect(keys(fd["state_avg"])))
        # observations key naming: try common variants
        obs_group = haskey(fo, "observations") ? fo["observations"] : nothing

        # truth and obs
        for k in keys_state
            push!(beta_true, read(fo["state"][k]["beta"]))
        end
        if obs_group !== nothing
            ko = sort(collect(keys(obs_group)))
            for k in ko
                push!(obs_vec, vec(read(obs_group[k])))
            end
        end

        # posterior mean + variance (skip t0000 if obs starts at t0001 — keep all and align later)
        for k in keys_avg
            push!(beta_est, read(fd["state_avg"][k]["beta"]))
            push!(beta_var, read(fd["state_var"][k]["beta"]))
        end
        kw = sort(collect(keys(fd["weights"])))
        for k in kw
            push!(weights, vec(read(fd["weights"][k])))
        end
    end
end

# Align lengths: keep min length
T = minimum((length(beta_true), length(beta_est)))
beta_true = beta_true[1:T]
beta_est  = beta_est[1:T]
beta_var  = beta_var[1:T]
weights   = weights[1:T]
@info "Aligned to T = $T snapshots"

ny, nx = size(beta_true[1])

# ── Grid + sensors ────────────────────────────────────────────────────────────
grid_x = h5read(DA_PATH, "grid_coordinates/x")
grid_y = h5read(DA_PATH, "grid_coordinates/y")
sx     = h5read(DA_PATH, "station_coordinates/x")
sy     = h5read(DA_PATH, "station_coordinates/y")

# Recompute sensor linear indices from station_coordinates (column-major: idx = (col-1)*ny + row).
dx = grid_x[2] - grid_x[1]
dy = grid_y[2] - grid_y[1]
sensor_indices = Int[]
for (xs, ys) in zip(sx, sy)
    i = round(Int, xs / dx) + 1   # x column
    j = round(Int, ys / dy) + 1   # y row
    push!(sensor_indices, (i - 1) * ny + j)
end

# ── Surrogate ux (same as model) ────────────────────────────────────────────
ux_field(β) = 1.0e3 ./ (β .+ 1e-6)

# ── Error fields + plot limits ──────────────────────────────────────────────
beta_err = [beta_est[t] .- beta_true[t] for t in 1:T]
climβ = (0.0, 2000.0)
emax = maximum(maximum(abs.(b)) for b in beta_err)
climE = (-emax, emax)

# ── GIFs ─────────────────────────────────────────────────────────────────────
anim1 = @animate for t in 1:T
    heatmap(grid_x, grid_y, beta_true[t]; title="True β (t=$t)",
            aspect_ratio=1, clims=climβ, c=:Blues, colorbar=true)
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim1, joinpath(SAVE_DIR, "true_beta.gif"), fps=6)

anim2 = @animate for t in 1:T
    heatmap(grid_x, grid_y, beta_est[t]; title="PF Mean β (t=$t)",
            aspect_ratio=1, clims=climβ, c=:Blues, colorbar=true)
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim2, joinpath(SAVE_DIR, "est_beta.gif"), fps=6)

anim3 = @animate for t in 1:T
    heatmap(grid_x, grid_y, beta_err[t]; title="β error = est − true (t=$t)",
            aspect_ratio=1, clims=climE, c=:balance, colorbar=true)
    scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
end
gif(anim3, joinpath(SAVE_DIR, "error_beta.gif"), fps=6)

# ── Final error heatmap ──────────────────────────────────────────────────────
p_final = heatmap(grid_x, grid_y, beta_err[end]; title="Final β Error",
                  aspect_ratio=1, clims=climE, c=:balance, colorbar=true)
scatter!(sx, sy; markershape=:circle, color=:red, label="Sensors", markersize=2)
savefig(p_final, joinpath(SAVE_DIR, "final_beta_error.png"))

# ── RMSE traces ──────────────────────────────────────────────────────────────
rmse_beta = zeros(T)
rmse_vel  = zeros(T)
for t in 1:T
    rmse_beta[t] = sqrt(mean((beta_est[t] .- beta_true[t]).^2))
    ux_true_flat = vec(ux_field(beta_true[t]))
    ux_est_flat  = vec(ux_field(beta_est[t]))
    rmse_vel[t]  = sqrt(mean((ux_est_flat[sensor_indices] .- ux_true_flat[sensor_indices]).^2))
end
savefig(plot(1:T, rmse_beta; xlabel="Time step", ylabel="RMSE(β)",
             title="Global RMSE of β", legend=false, lw=2),
        joinpath(SAVE_DIR, "rmse_beta.png"))
savefig(plot(1:T, rmse_vel; xlabel="Time step", ylabel="RMSE(ux @ sensors)",
             title="Sensor velocity RMSE", legend=false, lw=2),
        joinpath(SAVE_DIR, "rmse_vel.png"))

# ── Pointwise β at first sensor ──────────────────────────────────────────────
target = sensor_indices[1]
truth_pt = [vec(beta_true[t])[target] for t in 1:T]
est_pt   = [vec(beta_est[t])[target]  for t in 1:T]
p_pt = plot(1:T, truth_pt; label="True β", lw=2, title="β at first sensor",
            xlabel="Time step", ylabel="β")
plot!(1:T, est_pt; label="PF Mean", lw=2, linestyle=:dash)
savefig(p_pt, joinpath(SAVE_DIR, "beta_pointwise.png"))

# ── ESS over time ────────────────────────────────────────────────────────────
Np = length(weights[1])
ess = [begin
    w = weights[t]
    wn = w ./ sum(w)
    1.0 / sum(wn .^ 2)
end for t in 1:T]
p_ess = plot(1:T, ess; xlabel="Time step", ylabel="ESS",
             title="Effective Sample Size", lw=2, marker=:circle,
             markersize=3, ylim=(0, Np), legend=:topright, label="ESS")
hline!(p_ess, [0.5 * Np]; linestyle=:dash, color=:red, label="0.5·Np")
savefig(p_ess, joinpath(SAVE_DIR, "ess_evolution.png"))

# ── Max weight + entropy (collapse diagnostics) ──────────────────────────────
maxw = [maximum(w ./ sum(w)) for w in weights]
ent  = [begin wn = w ./ sum(w); -sum(wn .* log.(wn .+ eps())) end for w in weights]
savefig(plot(1:T, maxw; xlabel="Time step", ylabel="max(weight)",
             title="Max normalised weight", lw=2, legend=false),
        joinpath(SAVE_DIR, "max_weight.png"))
p_ent = plot(1:T, ent; xlabel="Time step", ylabel="Entropy",
             title="Weight entropy", lw=2, label="entropy")
hline!(p_ent, [log(Np)]; linestyle=:dash, color=:red, label="log(Np)")
savefig(p_ent, joinpath(SAVE_DIR, "weight_entropy.png"))

# ── Posterior variance ≈ particle variance (since we used MeanAndVarSummaryStat)
mean_var = [mean(beta_var[t]) for t in 1:T]
savefig(plot(1:T, mean_var; xlabel="Time step", ylabel="mean posterior Var(β)",
             title="Posterior variance (mean over grid)", lw=2, legend=false),
        joinpath(SAVE_DIR, "particle_variance.png"))

println("\nSaved to: ", abspath(SAVE_DIR))
for f in ("true_beta.gif", "est_beta.gif", "error_beta.gif",
          "final_beta_error.png", "rmse_beta.png", "rmse_vel.png",
          "beta_pointwise.png", "ess_evolution.png", "max_weight.png",
          "weight_entropy.png", "particle_variance.png")
    println("  - ", f)
end
