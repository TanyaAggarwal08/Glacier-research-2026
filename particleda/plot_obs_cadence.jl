# Compare run-04 (every-1s) vs run-05 (every-2s) on the same plots.
# X-axis is always MODEL TIME (seconds), so the two runs are directly comparable
# even though run-04 has 200 filter steps and run-05 has only 100.
#
# Run: julia --project=test glacier-code/particleda/plot_obs_cadence.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const CADENCE_DIR = joinpath("glacier-code", "particleda", "results", "obs_cadence")

struct CaseResult
    label::String
    time_step::Float64           # seconds per filter step
    model_time::Vector{Float64}  # cumulative model time at each filter step
    ess::Vector{Float64}
    maxw::Vector{Float64}
    rmse_beta::Vector{Float64}
    beta_var_mean::Vector{Float64}
end

function load_case(label::String, time_step::Float64)
    sd = joinpath(CADENCE_DIR, label)
    obs_path = joinpath(sd, "glacier_obs.h5")
    da_path  = joinpath(sd, "particle_da.h5")

    beta_true = Vector{Matrix{Float64}}()
    beta_est  = Vector{Matrix{Float64}}()
    beta_var  = Vector{Matrix{Float64}}()
    weights   = Vector{Vector{Float64}}()

    h5open(obs_path, "r") do fo
        # Match the DA file: skip t0000 placeholder.
        for k in sort(collect(keys(fo["state"])))[2:end]
            push!(beta_true, read(fo["state"][k]["beta"]))
        end
    end
    h5open(da_path, "r") do fd
        # Skip t0000 — ParticleDA writes an initial placeholder snapshot before
        # any observation update; its weights are uninitialised (NaN).
        for k in sort(collect(keys(fd["state_avg"])))[2:end]
            push!(beta_est, read(fd["state_avg"][k]["beta"]))
            push!(beta_var, read(fd["state_var"][k]["beta"]))
        end
        for k in sort(collect(keys(fd["weights"])))[2:end]
            push!(weights, vec(read(fd["weights"][k])))
        end
    end

    T = minimum((length(beta_true), length(beta_est), length(weights)))
    beta_true = beta_true[1:T]; beta_est = beta_est[1:T]
    beta_var  = beta_var[1:T];  weights  = weights[1:T]

    ess  = [begin w=weights[t]; wn=w./sum(w); 1.0/sum(wn.^2) end for t in 1:T]
    maxw = [maximum(w ./ sum(w)) for w in weights]
    rmse = [sqrt(mean((beta_est[t] .- beta_true[t]).^2)) for t in 1:T]
    bvar = [mean(beta_var[t]) for t in 1:T]
    mtime = collect(0:(T-1)) .* time_step

    return CaseResult(label, time_step, mtime, ess, maxw, rmse, bvar)
end

cases = [
    load_case("run04_every1c", 400.0),
    load_case("run05_every2c", 800.0),
]
Np = 1000

function overlay(metric_fn::Function, ylab::String, title::String, fname::String;
                 hline_at::Union{Nothing,Float64}=nothing, ylim=nothing,
                 cases=cases)
    p = plot(; xlabel="Model time (s)", ylabel=ylab, title=title,
             legend=:topright, ylim=ylim)
    for (i, c) in enumerate(cases)
        plot!(p, c.model_time, metric_fn(c); label=c.label, lw=1.5, alpha=0.85)
    end
    if hline_at !== nothing
        hline!(p, [hline_at]; linestyle=:dash, color=:red, label=nothing)
    end
    savefig(p, joinpath(CADENCE_DIR, fname))
end

overlay(c -> c.ess, "ESS", "Effective Sample Size — run-04 vs run-05",
        "ess_cadence.png"; hline_at=0.5 * Np, ylim=(0, Np))
overlay(c -> c.maxw, "max(weight)", "Max normalised weight",
        "maxweight_cadence.png"; ylim=(0, 1))
overlay(c -> c.rmse_beta, "RMSE(β)", "Global RMSE of β",
        "rmse_cadence.png")
overlay(c -> c.beta_var_mean, "mean posterior Var(β)",
        "Posterior variance (mean over grid)", "variance_cadence.png")

# Summary numbers
println("\nSummary:")
@printf "%-18s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s\n" "label" "ts(s)" "n_steps" "mean_ESS" "min_ESS" "frac>0.5Np" "RMSE_final"
csv_lines = ["label,time_step,n_steps,mean_ess,min_ess,frac_above_half_np,rmse_final"]
for c in cases
    me = mean(c.ess); mi = minimum(c.ess); fr = mean(c.ess .> 0.5 * Np)
    rf = mean(c.rmse_beta[max(end-9, 1):end])
    @printf "%-18s  %-9.2f  %-9d  %-9.1f  %-9.1f  %-9.3f  %-9.2f\n" c.label c.time_step length(c.ess) me mi fr rf
    push!(csv_lines, "$(c.label),$(c.time_step),$(length(c.ess)),$me,$mi,$fr,$rf")
end
write(joinpath(CADENCE_DIR, "summary.csv"), join(csv_lines, "\n") * "\n")

println("\nSaved: ess_cadence.png, maxweight_cadence.png, rmse_cadence.png, variance_cadence.png, summary.csv")
println("       in $(abspath(CADENCE_DIR))")
