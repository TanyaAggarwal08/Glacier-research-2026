# Experiment 15 (Goal 2) — does "log beats velocity" survive error bars?
#
# run14 vs run12 showed log β beating velocity by ~6.6 % on mean RMSE, but from
# a SINGLE seed pair, with both filters collapsing (min ESS 5.6 and 1.0). That
# is not evidence. This script repeats both operators across 10 seed pairs at
# the σ_log chosen in Exp 14 (Goal 1) and reports spread.
#
# Noise levels are delta-method equivalent at the prior centre β = 2000
# (ux = 0.5):   σ_v = σ_log · ux = 0.40 × 0.5 = 0.20
# Pitting σ_log = 0.40 against the old σ_v = 0.10 would compare a healthy log
# filter to a velocity filter still tuned to the sharp likelihood that caused
# the run14 collapse — exactly the confound this experiment exists to remove.
#
# SEED DESIGN — both seeds vary, and the design is PAIRED:
#   SEED_PF  = 42  + s   (particle init, process noise, resampling, jitter)
#   SEED_OBS = 123 + s   (truth process noise + observation noise)
# Varying SEED_PF alone would give error bars conditional on one truth
# realisation, which cannot speak to generalisation. Varying both captures
# truth-trajectory + obs-noise + filter noise. This costs no power because both
# operators consume exactly n_obs normals per step, so at a given SEED_OBS the
# truth is bit-identical across the two arms — each seed is a matched pair, and
# the paired difference cancels truth-realisation variance. That identity is
# asserted at runtime below rather than assumed.
#
# The truth FIELD (prior_truth_seed = 11) is held fixed throughout: we vary the
# noise realisations around one β field, not the field itself.
#
# Everything else matches run14 exactly (pseudo-random wave prior, 30 sensors,
# N = 1000, T = 100, SIGMA_JITTER = 20.0, process_std_beta = 10.0).
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_15_seed_sweep_operator_comparison.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf, HDF5, Distributions
include(joinpath(REPO_ROOT, "glacier-code", "particleda", "glacier_model.jl"))
using .Glacier
ENV["GKSwstype"] = "100"
using Plots

const RMSE_OUT = joinpath("glacier-code", "particleda", "results", "rmse_analysis")
mkpath(RMSE_OUT)

const NPRT = 1000
const T    = 100
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0
const SEEDS   = 1:10
const SIGMA_LOG = 0.40      # chosen in Exp 14 (Goal 1)
const SIGMA_V   = 0.20      # delta-method equivalent at β = 2000

base_glacier = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection_30.txt",
    "prior_mode" => "pseudo_random_wave",
    "prior_center_beta" => 2000.0,
    "prior_signal_scale_beta" => 300.0,
    "background_std_beta" => 300.0,
    "prior_max_wavenumber" => 2,
    "prior_truth_seed" => 11,
    "prior_background_seed" => 29,
    "init_std_beta" => 200.0,
    "process_std_beta" => 10.0,
    "advection_epsilon" => 0.0,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "lax_wendroff",
)

function systematic_resample(w::AbstractVector, rng::AbstractRNG)
    N = length(w)
    c = cumsum(w)
    u0 = rand(rng) / N
    idx = Vector{Int}(undef, N)
    j = 1
    for i in 1:N
        u = u0 + (i-1)/N
        while j < N && c[j] < u
            j += 1
        end
        idx[i] = j
    end
    return idx
end

function run_pf(; obs_space::String, sigma_log::Float64, sigma_v::Float64,
                  seed_pf::Int, seed_obs::Int)
    Glacier._CFL_WARNED[] = false
    params = merge(base_glacier, Dict{String, Any}(
        "obs_space"         => obs_space,
        "obs_noise_std"     => sigma_v,
        "obs_noise_std_log" => sigma_log,
    ))
    model = Glacier.init(Dict("glacier" => params))
    n_state = ParticleDA.get_state_dimension(model)
    n_obs   = ParticleDA.get_observation_dimension(model)

    rng_truth = MersenneTwister(seed_obs)
    truth_states = zeros(n_state, T+1)
    observations = zeros(n_obs, T)
    s_truth = copy(model.truth_prior_mean)
    @inbounds for i in eachindex(s_truth)
        s_truth[i] = max(s_truth[i], model.parameters.min_beta)
    end
    truth_states[:, 1] = s_truth
    for t in 1:T
        ParticleDA.update_state_deterministic!(s_truth, model, t)
        ParticleDA.update_state_stochastic!(s_truth, model, rng_truth)
        truth_states[:, t+1] = s_truth
        y = zeros(n_obs)
        ParticleDA.sample_observation_given_state!(y, s_truth, model, rng_truth)
        observations[:, t] = y
    end

    rng_pf = MersenneTwister(seed_pf)
    particles = zeros(n_state, NPRT)
    for p in 1:NPRT
        ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
    end
    ensemble_mean = zeros(n_state, T+1)
    ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]
    ess_series  = zeros(T)
    log_weights = zeros(NPRT)
    n_resample  = 0

    for t in 1:T
        for p in 1:NPRT
            sp = view(particles, :, p)
            ParticleDA.update_state_deterministic!(sp, model, t)
            ParticleDA.update_state_stochastic!(sp, model, rng_pf)
        end
        y_t = view(observations, :, t)
        for p in 1:NPRT
            log_weights[p] += ParticleDA.get_log_density_observation_given_state(
                y_t, view(particles, :, p), model)
        end
        lmax = maximum(log_weights)
        w = exp.(log_weights .- lmax); w ./= sum(w)
        ess_series[t] = 1.0 / sum(w .^ 2)
        ensemble_mean[:, t+1] = particles * w
        if ess_series[t] < ESS_THRESHOLD
            idx = systematic_resample(w, rng_pf)
            particles = particles[:, idx]
            if SIGMA_JITTER > 0
                for p in 1:NPRT
                    Glacier._apply_noise!(view(particles, :, p), model, rng_pf,
                                          SIGMA_JITTER, 1)
                end
            end
            log_weights .= 0.0
            n_resample += 1
        end
    end

    rmse = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2)) for t in 1:T+1]
    return (ess=ess_series, rmse=rmse, n_resample=n_resample, truth=truth_states)
end

println("=== Experiment 15 (Goal 2): seed-repeat operator comparison ===")
println("Pseudo-random wave prior, 30 sensors, N=$NPRT, T=$T")
println("log_beta arm : σ_log = $SIGMA_LOG")
println("velocity arm : σ_v   = $SIGMA_V   (delta-method equivalent at β=2000)")
println("Seeds        : SEED_PF = 42+s, SEED_OBS = 123+s for s = $(collect(SEEDS))")
println("Held fixed   : SIGMA_JITTER=$SIGMA_JITTER, process_std_beta=$(base_glacier["process_std_beta"])\n")

res_log = Dict{Int, Any}()
res_vel = Dict{Int, Any}()
for s in SEEDS
    spf, sobs = 42 + s, 123 + s
    t0 = time()
    rl = run_pf(obs_space="log_beta", sigma_log=SIGMA_LOG, sigma_v=SIGMA_V,
                seed_pf=spf, seed_obs=sobs)
    rv = run_pf(obs_space="velocity", sigma_log=SIGMA_LOG, sigma_v=SIGMA_V,
                seed_pf=spf, seed_obs=sobs)
    # The pairing claim, verified rather than assumed.
    @assert rl.truth == rv.truth "truth differs between arms at seed s=$s — pairing broken"
    res_log[s] = rl; res_vel[s] = rv
    @printf("s=%-3d (PF=%d,OBS=%d)  log: final=%-6.1f last10=%-6.1f minESS=%-6.1f | vel: final=%-6.1f last10=%-6.1f minESS=%-6.1f  (%.0fs)\n",
            s, spf, sobs,
            rl.rmse[end], mean(rl.rmse[end-9:end]), minimum(rl.ess),
            rv.rmse[end], mean(rv.rmse[end-9:end]), minimum(rv.ess),
            time()-t0)
end

final_log  = [res_log[s].rmse[end] for s in SEEDS]
final_vel  = [res_vel[s].rmse[end] for s in SEEDS]
last10_log = [mean(res_log[s].rmse[end-9:end]) for s in SEEDS]
last10_vel = [mean(res_vel[s].rmse[end-9:end]) for s in SEEDS]

_ms(v) = (mean(v), std(v))

println("\n=== Unpaired summary across $(length(SEEDS)) seeds (mean ± std) ===\n")
@printf("%-22s %18s %18s\n", "metric", "log_beta", "velocity")
for (name, vl, vv) in [("final RMSE", final_log, final_vel),
                       ("last-10-mean RMSE", last10_log, last10_vel)]
    ml, sl = _ms(vl); mv, sv = _ms(vv)
    @printf("%-22s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", name, ml, sl, mv, sv)
end

println("\n=== ESS health across seeds (mean ± std) ===\n")
minl = [minimum(res_log[s].ess) for s in SEEDS]
minv = [minimum(res_vel[s].ess) for s in SEEDS]
medl = [median(res_log[s].ess) for s in SEEDS]
medv = [median(res_vel[s].ess) for s in SEEDS]
rsl  = [100 * res_log[s].n_resample / T for s in SEEDS]
rsv  = [100 * res_vel[s].n_resample / T for s in SEEDS]
for (name, vl, vv) in [("min ESS", minl, minv), ("median ESS", medl, medv),
                       ("resample %", rsl, rsv)]
    ml, sl = _ms(vl); mv, sv = _ms(vv)
    @printf("%-22s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", name, ml, sl, mv, sv)
end

# ── Paired analysis: d = velocity − log, positive means log wins ─────────
println("\n=== Paired difference (velocity − log_beta); positive ⇒ log wins ===\n")
for (name, dl, dv) in [("final RMSE", final_log, final_vel),
                       ("last-10-mean RMSE", last10_log, last10_vel)]
    d = dv .- dl
    md, sd = mean(d), std(d)
    sem = sd / sqrt(length(d))
    tstat = md / sem
    pval = 2 * ccdf(TDist(length(d) - 1), abs(tstat))
    wins = count(>(0), d)
    @printf("%s:\n", name)
    @printf("  mean diff = %+.1f ± %.1f (sd)   sem = %.1f\n", md, sd, sem)
    @printf("  log wins on %d / %d seeds\n", wins, length(d))
    @printf("  paired t(%d) = %+.2f   p = %.3f   →  %s\n\n",
            length(d)-1, tstat, pval,
            pval < 0.05 ? "significant at 0.05" : "NOT significant at 0.05")
end

# ── Plots ────────────────────────────────────────────────────────────────
p1 = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
          title="Exp 15 — RMSE across $(length(SEEDS)) seeds", legend=:topright)
for (i, s) in enumerate(SEEDS)
    plot!(p1, 0:T, res_log[s].rmse; color=:firebrick, alpha=0.35, lw=1,
          label=(i==1 ? "log_beta, σ_log=$SIGMA_LOG" : false))
    plot!(p1, 0:T, res_vel[s].rmse; color=:steelblue, alpha=0.35, lw=1,
          label=(i==1 ? "velocity, σ_v=$SIGMA_V" : false))
end
plot!(p1, 0:T, [mean([res_log[s].rmse[t] for s in SEEDS]) for t in 1:T+1];
      color=:firebrick, lw=3.5, label="log_beta mean")
plot!(p1, 0:T, [mean([res_vel[s].rmse[t] for s in SEEDS]) for t in 1:T+1];
      color=:steelblue, lw=3.5, label="velocity mean")
savefig(p1, joinpath(RMSE_OUT, "rmse_seed_sweep_operator_exp15.png"))

d_last10 = last10_vel .- last10_log
p2 = bar(collect(SEEDS), d_last10;
         xlabel="seed s", ylabel="RMSE(velocity) − RMSE(log_beta)",
         title="Exp 15 — paired difference, last-10-mean RMSE",
         label="positive ⇒ log wins",
         color=[d > 0 ? :seagreen : :firebrick for d in d_last10])
hline!(p2, [0]; color=:black, lw=2, label=false)
hline!(p2, [mean(d_last10)]; color=:goldenrod, lw=2.5, linestyle=:dash,
       label=@sprintf("mean = %+.1f", mean(d_last10)))
savefig(p2, joinpath(RMSE_OUT, "paired_diff_operator_exp15.png"))

h5open(joinpath(RMSE_OUT, "exp15_seed_sweep.h5"), "w") do f
    for (arm, res) in [("log_beta", res_log), ("velocity", res_vel)]
        g = create_group(f, arm)
        for s in SEEDS
            gs = create_group(g, "seed_$s")
            gs["rmse"] = res[s].rmse
            gs["ess"]  = res[s].ess
        end
    end
    attributes(f)["sigma_log"] = SIGMA_LOG
    attributes(f)["sigma_v"]   = SIGMA_V
end
println("Saved → $(joinpath(RMSE_OUT, "rmse_seed_sweep_operator_exp15.png"))")
println("Saved → $(joinpath(RMSE_OUT, "paired_diff_operator_exp15.png"))")
println("Saved → $(joinpath(RMSE_OUT, "exp15_seed_sweep.h5"))")
println("\nGoal 2 complete.")
