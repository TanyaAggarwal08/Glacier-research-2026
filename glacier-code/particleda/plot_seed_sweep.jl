# Aggregate the seed sweep: overlay ESS / RMSE(β) / max-weight curves
# across seeds and dump a summary CSV.
#
# Run: julia --project=test glacier-code/particleda/plot_seed_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const SWEEP_DIR = joinpath("glacier-code", "particleda", "results", "seed_sweep_v2")
const SEEDS     = [42, 7, 13, 99, 2024]

struct SeedResult
    seed::Int
    ess::Vector{Float64}
    maxw::Vector{Float64}
    rmse_beta::Vector{Float64}
end

function load_seed(seed::Int)
    sd = joinpath(SWEEP_DIR, "seed$(lpad(seed, 4, '0'))")
    obs_path = joinpath(sd, "glacier_obs.h5")
    da_path  = joinpath(sd, "particle_da.h5")

    beta_true = Vector{Matrix{Float64}}()
    beta_est  = Vector{Matrix{Float64}}()
    weights   = Vector{Vector{Float64}}()

    h5open(obs_path, "r") do fo
        # Skip t0000 placeholder
        for k in sort(collect(keys(fo["state"])))[2:end]
            push!(beta_true, read(fo["state"][k]["beta"]))
        end
    end
    h5open(da_path, "r") do fd
        for k in sort(collect(keys(fd["state_avg"])))[2:end]
            push!(beta_est, read(fd["state_avg"][k]["beta"]))
        end
        for k in sort(collect(keys(fd["weights"])))[2:end]
            push!(weights, vec(read(fd["weights"][k])))
        end
    end

    T = minimum((length(beta_true), length(beta_est), length(weights)))
    beta_true = beta_true[1:T]; beta_est = beta_est[1:T]; weights = weights[1:T]

    ess = [begin w = weights[t]; wn = w ./ sum(w); 1.0 / sum(wn .^ 2) end for t in 1:T]
    maxw = [maximum(w ./ sum(w)) for w in weights]
    rmse_beta = [sqrt(mean((beta_est[t] .- beta_true[t]) .^ 2)) for t in 1:T]

    return SeedResult(seed, ess, maxw, rmse_beta)
end

results = [load_seed(s) for s in SEEDS]
T = length(results[1].ess)
Np = 1000  # from YAML

# ── Overlaid ESS ─────────────────────────────────────────────────────────────
p_ess = plot(; xlabel="Time step", ylabel="ESS", title="ESS — 5 seeds",
             ylim=(0, Np), legend=:bottomright)
for r in results
    plot!(p_ess, 1:T, r.ess; label="seed $(r.seed)", lw=1.2, alpha=0.8)
end
hline!(p_ess, [0.5 * Np]; linestyle=:dash, color=:red, label="0.5·Np")
savefig(p_ess, joinpath(SWEEP_DIR, "ess_all_seeds.png"))

# ── Overlaid max(weight) ─────────────────────────────────────────────────────
p_mw = plot(; xlabel="Time step", ylabel="max(weight)",
            title="Max normalised weight — 5 seeds", legend=:topright)
for r in results
    plot!(p_mw, 1:T, r.maxw; label="seed $(r.seed)", lw=1.2, alpha=0.8)
end
savefig(p_mw, joinpath(SWEEP_DIR, "maxweight_all_seeds.png"))

# ── Overlaid RMSE(β) ─────────────────────────────────────────────────────────
p_r = plot(; xlabel="Time step", ylabel="RMSE(β)",
           title="Global RMSE(β) — 5 seeds", legend=:topright)
for r in results
    plot!(p_r, 1:T, r.rmse_beta; label="seed $(r.seed)", lw=1.2, alpha=0.8)
end
savefig(p_r, joinpath(SWEEP_DIR, "rmse_all_seeds.png"))

# ── Median ± min/max envelope plot ───────────────────────────────────────────
function envelope(series_per_seed)
    M = hcat(series_per_seed...)  # T × n_seeds
    return (
        median = [median(M[t, :]) for t in 1:T],
        lo     = [minimum(M[t, :]) for t in 1:T],
        hi     = [maximum(M[t, :]) for t in 1:T],
    )
end

ess_env  = envelope([r.ess        for r in results])
rmse_env = envelope([r.rmse_beta  for r in results])

p_env1 = plot(1:T, ess_env.median; ribbon=(ess_env.median .- ess_env.lo,
                                            ess_env.hi .- ess_env.median),
              xlabel="Time step", ylabel="ESS",
              title="ESS — median ± min/max over 5 seeds",
              label="median", lw=2, fillalpha=0.25, ylim=(0, Np))
hline!(p_env1, [0.5 * Np]; linestyle=:dash, color=:red, label="0.5·Np")
savefig(p_env1, joinpath(SWEEP_DIR, "ess_envelope.png"))

p_env2 = plot(1:T, rmse_env.median; ribbon=(rmse_env.median .- rmse_env.lo,
                                             rmse_env.hi .- rmse_env.median),
              xlabel="Time step", ylabel="RMSE(β)",
              title="RMSE(β) — median ± min/max over 5 seeds",
              label="median", lw=2, fillalpha=0.25)
savefig(p_env2, joinpath(SWEEP_DIR, "rmse_envelope.png"))

# ── Summary table ────────────────────────────────────────────────────────────
println("\nSummary (per seed):")
@printf "%-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n" "seed" "mean_ESS" "min_ESS" "frac>0.5Np" "RMSE_init" "RMSE_final"
csv_lines = ["seed,mean_ess,min_ess,frac_above_half_np,rmse_init,rmse_final"]
for r in results
    me  = mean(r.ess)
    mi  = minimum(r.ess)
    frac = mean(r.ess .> 0.5 * Np)
    ri  = r.rmse_beta[1]
    rf  = mean(r.rmse_beta[max(end-9, 1):end])  # last 10 steps avg
    @printf "%-8d  %-10.1f  %-10.1f  %-10.3f  %-10.2f  %-10.2f\n" r.seed me mi frac ri rf
    push!(csv_lines, "$(r.seed),$(me),$(mi),$(frac),$(ri),$(rf)")
end

write(joinpath(SWEEP_DIR, "summary.csv"), join(csv_lines, "\n") * "\n")
println("\nSaved: ess_all_seeds.png, maxweight_all_seeds.png, rmse_all_seeds.png,")
println("       ess_envelope.png, rmse_envelope.png, summary.csv")
println("       in $(abspath(SWEEP_DIR))")
