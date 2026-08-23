# Side-by-side comparison plotter for the observation-ablation experiment.
#
# Overlays RMSE, ESS, max-weight, and pointwise β-at-sensor-1 from both runs:
#   A) Baseline:     results/run06_obs_baseline/
#   B) No-obs:       results/run06_no_observations/
#
# Outputs land in results/run06_ablation_compare/.
#
# Run: julia --project=test glacier-code/particleda/plot_obs_vs_no_obs.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const A_DIR = joinpath("glacier-code", "particleda", "results", "run06_obs_baseline")
const B_DIR = joinpath("glacier-code", "particleda", "results", "run06_no_observations")
const OUT   = joinpath("glacier-code", "particleda", "results", "run06_ablation_compare")
mkpath(OUT)

struct CaseData
    label::String
    time_step::Float64
    model_time::Vector{Float64}         # state clock (starts at 0)
    model_time_filter::Vector{Float64}  # filter clock (starts at time_step)
    beta_true::Vector{Matrix{Float64}}
    beta_est::Vector{Matrix{Float64}}
    weights::Vector{Vector{Float64}}
    rmse_beta::Vector{Float64}
    ess::Vector{Float64}
    maxw::Vector{Float64}
end

function load_case(dir::String, label::String)
    obs_path = joinpath(dir, "glacier_obs.h5")
    da_path  = joinpath(dir, "particle_da.h5")

    time_step = h5open(da_path, "r") do f
        Float64(read(attributes(f["parameters"])["time_step"]))
    end

    beta_true = Vector{Matrix{Float64}}()
    beta_est  = Vector{Matrix{Float64}}()
    weights   = Vector{Vector{Float64}}()

    h5open(obs_path, "r") do fo
        for k in sort(collect(keys(fo["state"])))
            push!(beta_true, read(fo["state"][k]["beta"]))
        end
    end
    h5open(da_path, "r") do fd
        for k in sort(collect(keys(fd["state_avg"])))
            push!(beta_est, read(fd["state_avg"][k]["beta"]))
        end
        for k in sort(collect(keys(fd["weights"])))[2:end]   # skip NaN t0000
            push!(weights, vec(read(fd["weights"][k])))
        end
    end

    T_state = min(length(beta_true), length(beta_est))
    beta_true = beta_true[1:T_state]
    beta_est  = beta_est[1:T_state]

    model_time        = collect(0:T_state-1) .* time_step
    model_time_filter = collect(1:length(weights)) .* time_step

    rmse_beta = [sqrt(mean((beta_est[t] .- beta_true[t]).^2)) for t in 1:T_state]
    ess  = [begin w=weights[t]; wn=w./sum(w); 1.0/sum(wn.^2) end for t in eachindex(weights)]
    maxw = [maximum(w ./ sum(w)) for w in weights]

    return CaseData(label, time_step, model_time, model_time_filter,
                    beta_true, beta_est, weights, rmse_beta, ess, maxw)
end

# Split RMSE into sensor-cell and unsensored-cell components.
function rmse_split(case::CaseData, sensor_flat_idx::Vector{Int})
    T = length(case.beta_true)
    n_cells = length(case.beta_true[1])
    sensor_set = Set(sensor_flat_idx)
    other_idx = [i for i in 1:n_cells if i ∉ sensor_set]
    sens = Float64[]; unsens = Float64[]
    for t in 1:T
        bt = vec(case.beta_true[t]); be = vec(case.beta_est[t])
        push!(sens,   sqrt(mean((be[sensor_flat_idx] .- bt[sensor_flat_idx]).^2)))
        push!(unsens, sqrt(mean((be[other_idx]       .- bt[other_idx]      ).^2)))
    end
    return sens, unsens
end

A = load_case(A_DIR, "with obs (baseline)")
B = load_case(B_DIR, "no obs (ablation)")

# ── RMSE comparison ─────────────────────────────────────────────────────────
p_rmse = plot(A.model_time, A.rmse_beta;
              xlabel="Model time (s)", ylabel="RMSE(β)",
              title="Global RMSE — observations on vs off",
              label=A.label, lw=2, color=:steelblue)
plot!(p_rmse, B.model_time, B.rmse_beta;
      label=B.label, lw=2, color=:firebrick)
savefig(p_rmse, joinpath(OUT, "rmse_compare.png"))

# ── RMSE in log-y to make the gap obvious if convergence is dramatic ────────
p_rmse_log = plot(A.model_time, A.rmse_beta;
                  xlabel="Model time (s)", ylabel="RMSE(β)  (log scale)",
                  title="Global RMSE — log scale",
                  label=A.label, lw=2, color=:steelblue, yscale=:log10)
plot!(p_rmse_log, B.model_time, B.rmse_beta;
      label=B.label, lw=2, color=:firebrick)
savefig(p_rmse_log, joinpath(OUT, "rmse_compare_log.png"))

# ── Sensor-cell vs unsensored-cell RMSE (the diagnostic split) ──────────────
# Global RMSE averages over all 1600 cells but only ~1% are sensored, so the
# observation signal gets swamped. Split it: do observations actually help
# *where they land*?
sensor_x = h5read(joinpath(A_DIR, "particle_da.h5"), "station_coordinates/x")
sensor_y = h5read(joinpath(A_DIR, "particle_da.h5"), "station_coordinates/y")
ny, nx = size(A.beta_true[1])
dx = 160000.0 / nx
dy = 160000.0 / ny
sensor_flat = Int[]
for k in eachindex(sensor_x)
    i_col = clamp(round(Int, sensor_x[k] / dx) + 1, 1, nx)
    j_row = clamp(round(Int, sensor_y[k] / dy) + 1, 1, ny)
    push!(sensor_flat, (i_col - 1) * ny + j_row)
end
sensor_flat = unique(sensor_flat)

A_sens, A_unsens = rmse_split(A, sensor_flat)
B_sens, B_unsens = rmse_split(B, sensor_flat)

p_split = plot(A.model_time, A_sens;
               xlabel="Model time (s)", ylabel="RMSE(β)",
               title="RMSE split — sensor cells vs unsensored cells",
               label="with obs — sensor cells", lw=2, color=:steelblue)
plot!(p_split, A.model_time, A_unsens;
      label="with obs — unsensored cells", lw=2, linestyle=:dash, color=:steelblue)
plot!(p_split, B.model_time, B_sens;
      label="no obs — sensor cells", lw=2, color=:firebrick)
plot!(p_split, B.model_time, B_unsens;
      label="no obs — unsensored cells", lw=2, linestyle=:dash, color=:firebrick)
savefig(p_split, joinpath(OUT, "rmse_split_sensor_vs_unsensored.png"))

# ── ESS comparison ──────────────────────────────────────────────────────────
Np = length(A.weights[1])
p_ess = plot(A.model_time_filter, A.ess;
             xlabel="Model time (s)", ylabel="ESS",
             title="Effective Sample Size",
             label=A.label, lw=1.5, alpha=0.85, color=:steelblue,
             ylim=(0, Np))
plot!(p_ess, B.model_time_filter, B.ess;
      label=B.label, lw=1.5, alpha=0.85, color=:firebrick)
hline!(p_ess, [0.5 * Np]; linestyle=:dash, color=:gray, label="0.5·Np")
savefig(p_ess, joinpath(OUT, "ess_compare.png"))

# ── Max-weight comparison ───────────────────────────────────────────────────
p_mw = plot(A.model_time_filter, A.maxw;
            xlabel="Model time (s)", ylabel="max(weight)",
            title="Max normalised weight",
            label=A.label, lw=1.5, alpha=0.85, color=:steelblue,
            ylim=(0, 1))
plot!(p_mw, B.model_time_filter, B.maxw;
      label=B.label, lw=1.5, alpha=0.85, color=:firebrick)
savefig(p_mw, joinpath(OUT, "maxweight_compare.png"))

# ── Pointwise β at sensor 1 (or first interior cell if sensor 1 is at corner)
i_col = round(Int, sensor_x[1] / dx) + 1
j_row = round(Int, sensor_y[1] / dy) + 1
sensor_idx_flat = (i_col - 1) * ny + j_row

# Normalised view so both runs share a clean [-1, 1] reference.
normβ(x) = (x .- 1000.0) ./ 500.0

# Load the noisy observations for sensor 1 (in ux units) and convert to β.
# obs_ux ≈ 1000/β + ε  ⇒  β_obs ≈ 1000/obs_ux  (point estimate, no error bar).
obs_ux_A = Float64[]
obs_times_A = Float64[]
h5open(joinpath(A_DIR, "glacier_obs.h5"), "r") do fo
    for k in sort(collect(keys(fo["observations"])))
        v = read(fo["observations"][k])
        push!(obs_ux_A, vec(v)[1])
        # observations are taken at the END of each filter step (t=1..n)
        push!(obs_times_A, parse(Int, replace(k, "t"=>"")) * A.time_step)
    end
end
beta_obs_A_n = normβ(1000.0 ./ obs_ux_A)

truth_pt_A_n = normβ([vec(A.beta_true[t])[sensor_idx_flat] for t in 1:length(A.model_time)])
est_pt_A_n   = normβ([vec(A.beta_est[t])[sensor_idx_flat]  for t in 1:length(A.model_time)])
est_pt_B_n   = normβ([vec(B.beta_est[t])[sensor_idx_flat]  for t in 1:length(B.model_time)])

p_pt = plot(A.model_time, truth_pt_A_n;
            xlabel="Model time (s)", ylabel="β_norm = (β − 1000) / 500",
            title="β at first sensor (normalised) — truth vs PF mean, with vs without obs",
            label="truth (shared)", lw=2.5, color=:black, ylim=(-1.5, 1.5))
plot!(p_pt, A.model_time, est_pt_A_n;
      label="PF mean (with obs)", lw=2, linestyle=:dash, color=:steelblue)
plot!(p_pt, B.model_time, est_pt_B_n;
      label="PF mean (no obs)", lw=2, linestyle=:dashdot, color=:firebrick)
scatter!(p_pt, obs_times_A, beta_obs_A_n;
         label="obs (β-equivalent)", marker=:xcross, ms=4, mc=:black, msw=1.5)
hline!(p_pt, [0.0]; linestyle=:dot, color=:gray, label="prior mean")
savefig(p_pt, joinpath(OUT, "beta_pointwise_compare.png"))

# ── Multi-point view at NON-DEGENERATE cells (proves dynamics are running) ──
# These cells sit on strong prior gradients so the advection has work to do.
# If dynamics were dead, the no-obs lines would be flat. They're not.
probe_cells = [(10, 10), (10, 30), (30, 10), (5, 15)]   # (i_col, j_row)
ω = 2π / 160000.0
function prior_at(i, j)
    x = (i - 1) * (160000.0 / nx); y = (j - 1) * (160000.0 / ny)
    return 1000 + 500 * sin(ω*x) * sin(ω*y)
end

panels = []
for (ic, jr) in probe_cells
    flat_idx = (ic - 1) * ny + jr
    truth_series = [vec(A.beta_true[t])[flat_idx] for t in 1:length(A.model_time)]
    est_A_series = [vec(A.beta_est[t])[flat_idx]  for t in 1:length(A.model_time)]
    est_B_series = [vec(B.beta_est[t])[flat_idx]  for t in 1:length(B.model_time)]
    prior_val = prior_at(ic, jr)
    title_str = "cell (i=$ic, j=$jr), prior β = $(round(Int, prior_val))"
    p = plot(A.model_time, truth_series; label="truth", lw=2.5,
             color=:black, xlabel="Model time (s)", ylabel="β (Pa·s/m)",
             title=title_str, legend=:topright)
    plot!(p, A.model_time, est_A_series; label="PF mean (with obs)", lw=2,
          linestyle=:dash, color=:steelblue)
    plot!(p, B.model_time, est_B_series; label="PF mean (no obs)", lw=2,
          linestyle=:dashdot, color=:firebrick)
    hline!(p, [prior_val]; linestyle=:dot, color=:gray, label="prior β")
    push!(panels, p)
end
p_dyn = plot(panels...; layout=(2, 2), size=(1300, 950))
savefig(p_dyn, joinpath(OUT, "dynamics_verification.png"))

# ── Numerical summary ──────────────────────────────────────────────────────
println("\n=== Observation-ablation summary ===")
@printf "%-22s  %-12s  %-12s\n" "metric" "with obs" "no obs"
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE(β) initial"   A.rmse_beta[1]      B.rmse_beta[1]
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE(β) final"     mean(A.rmse_beta[end-9:end]) mean(B.rmse_beta[end-9:end])
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE(β) min"        minimum(A.rmse_beta) minimum(B.rmse_beta)
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE(β) max"        maximum(A.rmse_beta) maximum(B.rmse_beta)
@printf "%-22s  %-12.1f  %-12.1f\n" "mean ESS"          mean(A.ess)         mean(B.ess)
@printf "%-22s  %-12.3f  %-12.3f\n" "frac ESS > 0.5·Np" mean(A.ess .> 0.5 * Np) mean(B.ess .> 0.5 * Np)
@printf "%-22s  %-12.3f  %-12.3f\n" "mean max(weight)"  mean(A.maxw)        mean(B.maxw)
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE sensor (final)"  mean(A_sens[end-9:end])   mean(B_sens[end-9:end])
@printf "%-22s  %-12.2f  %-12.2f\n" "RMSE unsens (final)"  mean(A_unsens[end-9:end]) mean(B_unsens[end-9:end])

# Persist CSV
open(joinpath(OUT, "summary.csv"), "w") do io
    println(io, "metric,with_obs,no_obs")
    println(io, "rmse_initial,$(A.rmse_beta[1]),$(B.rmse_beta[1])")
    println(io, "rmse_final,$(mean(A.rmse_beta[end-9:end])),$(mean(B.rmse_beta[end-9:end]))")
    println(io, "rmse_min,$(minimum(A.rmse_beta)),$(minimum(B.rmse_beta))")
    println(io, "rmse_max,$(maximum(A.rmse_beta)),$(maximum(B.rmse_beta))")
    println(io, "mean_ess,$(mean(A.ess)),$(mean(B.ess))")
    println(io, "frac_ess_above_half_np,$(mean(A.ess .> 0.5*Np)),$(mean(B.ess .> 0.5*Np))")
    println(io, "mean_max_weight,$(mean(A.maxw)),$(mean(B.maxw))")
end

println("\nSaved to: ", abspath(OUT))
for f in ("rmse_compare.png", "rmse_compare_log.png",
          "rmse_split_sensor_vs_unsensored.png",
          "ess_compare.png", "maxweight_compare.png",
          "beta_pointwise_compare.png", "dynamics_verification.png",
          "summary.csv")
    println("  - ", f)
end
