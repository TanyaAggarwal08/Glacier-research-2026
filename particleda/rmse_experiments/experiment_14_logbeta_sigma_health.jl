# Experiment 14 (Goal 1) — σ_log sweep to cure the run14 ESS collapse.
#
# run14 (log β obs, σ_log = 0.20) collapses: min ESS 5.6 at t=5, 87 % of steps
# below N/2, degenerate on the very first assimilation. ESS reflects the JOINT
# weight — 30 sensors summed, accumulated across steps — so it is the
# diagnostic to trust, not the per-sensor ratio test in
# glacier-notes/25_observation_error_analysis.md. A collapsing ESS means the
# joint likelihood is too SHARP, so σ_log goes UP, not down.
#
# Sweeps σ_log ∈ {0.20, 0.30, 0.40, 0.50}. 0.20 is the run14 baseline and acts
# as a fidelity check: it must reproduce run14's published numbers exactly
# (mean ESS 321.1, min ESS 5.6 @ t=5, final RMSE 169.7).
#
# This mirrors run_pseudorandom_wave_30obs_logbeta.jl line for line, EXCEPT it
# does not retain all_particles / weights_raw (that is what made run14's
# tracking.h5 1.3 GB). It deliberately does NOT reuse _helpers.jl's
# run_pf_rmse, which differs from the run14 driver in two ways that would
# break comparability: it samples the truth initial state (run14 starts the
# truth deterministically at truth_prior_mean, consuming no RNG) and it
# defaults to sigma_jitter = 30.0 (run14 uses 20.0).
#
# SIGMA_JITTER and process_std_beta are held at the run14 values on purpose —
# this experiment isolates the σ_log effect.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_14_logbeta_sigma_health.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf, HDF5
include(joinpath(REPO_ROOT, "glacier-code", "particleda", "glacier_model.jl"))
using .Glacier
ENV["GKSwstype"] = "100"
using Plots

const RMSE_OUT = joinpath("glacier-code", "particleda", "results", "rmse_analysis")
mkpath(RMSE_OUT)
const FIG_ESS  = "ess_logbeta_sigma_sweep_exp14.png"
const FIG_RMSE = "rmse_logbeta_sigma_sweep_exp14.png"

const NPRT     = 1000
const T        = 100
const SEED_PF  = 42
const SEED_OBS = 123
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0          # run14 value — held fixed, not tuned here

const SIGMA_LOG_VALUES = [0.20, 0.30, 0.40, 0.50]

# Exactly run14's yaml_params, with obs_noise_std_log left to the sweep.
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
    "process_std_beta" => 10.0,      # run14 value — held fixed, not tuned here
    "obs_space" => "log_beta",
    "obs_noise_std" => 0.10,
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

# One full run14-equivalent PF run at a given σ_log.
function run_logbeta(sigma_log::Float64; seed_pf::Int=SEED_PF, seed_obs::Int=SEED_OBS,
                     obs_space::String="log_beta", sigma_v::Float64=0.10)
    Glacier._CFL_WARNED[] = false
    params = merge(base_glacier, Dict{String, Any}(
        "obs_space"         => obs_space,
        "obs_noise_std"     => sigma_v,
        "obs_noise_std_log" => sigma_log,
    ))
    model = Glacier.init(Dict("glacier" => params))
    n_state = ParticleDA.get_state_dimension(model)
    n_obs   = ParticleDA.get_observation_dimension(model)

    # ── truth + observations (run14: truth starts deterministic) ─────────
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

    # ── PF ──────────────────────────────────────────────────────────────
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
    return (ess=ess_series, rmse=rmse, n_resample=n_resample, n_obs=n_obs)
end

println("=== Experiment 14 (Goal 1): σ_log sweep on the run14 log_beta setup ===")
println("Pseudo-random wave prior, 30 sensors, N=$NPRT, T=$T, SEED_PF=$SEED_PF, SEED_OBS=$SEED_OBS")
println("Held fixed (NOT tuned here): SIGMA_JITTER=$SIGMA_JITTER, process_std_beta=$(base_glacier["process_std_beta"]), init_std_beta=$(base_glacier["init_std_beta"])")
println("Resample trigger: ESS < $(ESS_THRESHOLD) (= N/2)\n")

results = Dict{Float64, Any}()
for σ in SIGMA_LOG_VALUES
    t0 = time()
    res = run_logbeta(σ)
    results[σ] = res
    @printf("σ_log=%.2f done (%.0fs)  n_obs=%d\n", σ, time()-t0, res.n_obs)
end

# ── Report ───────────────────────────────────────────────────────────────
println("\n=== ESS health ===\n")
@printf("%-8s %10s %10s %10s %12s %12s %12s\n",
        "σ_log", "min ESS", "@ step", "median", "steps<N/2", "steps<N/10", "resample %")
for σ in SIGMA_LOG_VALUES
    r = results[σ]
    mn, am = findmin(r.ess)
    @printf("%-8.2f %10.1f %10d %10.1f %12s %12s %12.0f\n",
            σ, mn, am, median(r.ess),
            "$(count(<(NPRT/2), r.ess))/$T", "$(count(<(NPRT/10), r.ess))/$T",
            100 * r.n_resample / T)
end

println("\n=== RMSE(β) tradeoff ===\n")
@printf("%-8s %12s %12s %12s %12s\n",
        "σ_log", "final", "last-10 mean", "run mean", "mean ESS")
for σ in SIGMA_LOG_VALUES
    r = results[σ]
    @printf("%-8.2f %12.1f %12.1f %12.1f %12.1f\n",
            σ, r.rmse[end], mean(r.rmse[end-9:end]), mean(r.rmse), mean(r.ess))
end

# Fidelity check against the published run14 numbers.
r020 = results[0.20]
mn020, am020 = findmin(r020.ess)
println("\n=== Fidelity check: σ_log=0.20 must reproduce run14 ===")
@printf("  mean ESS   : %.1f  (run14 published 321.1)\n", mean(r020.ess))
@printf("  min ESS    : %.1f @ t=%d  (run14 published 5.6 @ t=5)\n", mn020, am020)
@printf("  final RMSE : %.1f  (run14 published 169.7)\n", r020.rmse[end])

# ── Plots ────────────────────────────────────────────────────────────────
const COLORS = [:firebrick, :goldenrod, :seagreen, :steelblue]
p_ess = plot(; xlabel="timestep", ylabel="ESS", ylim=(0, NPRT),
             title="Exp 14 — ESS vs σ_log (log β obs, 30 sensors)",
             legend=:topright)
for (i, σ) in enumerate(SIGMA_LOG_VALUES)
    r = results[σ]
    plot!(p_ess, 1:T, r.ess;
          label=@sprintf("σ_log = %.2f  (min %.0f)", σ, minimum(r.ess)),
          lw=2, color=COLORS[i])
end
hline!(p_ess, [NPRT/2]; linestyle=:dash, color=:gray, label="N/2 (resample trigger)")
hline!(p_ess, [NPRT/10]; linestyle=:dot, color=:black, label="N/10")
savefig(p_ess, joinpath(RMSE_OUT, FIG_ESS))
println("\nSaved → $(joinpath(RMSE_OUT, FIG_ESS))")

p_rmse = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
              title="Exp 14 — RMSE vs σ_log (log β obs, 30 sensors)",
              legend=:topright)
for (i, σ) in enumerate(SIGMA_LOG_VALUES)
    r = results[σ]
    plot!(p_rmse, 0:T, r.rmse;
          label=@sprintf("σ_log = %.2f  (final %.0f)", σ, r.rmse[end]),
          lw=2.5, color=COLORS[i])
end
savefig(p_rmse, joinpath(RMSE_OUT, FIG_RMSE))
println("Saved → $(joinpath(RMSE_OUT, FIG_RMSE))")

h5open(joinpath(RMSE_OUT, "exp14_logbeta_sigma_sweep.h5"), "w") do f
    for σ in SIGMA_LOG_VALUES
        g = create_group(f, "sigma_$(replace(string(σ), "." => "p"))")
        g["ess"] = results[σ].ess
        g["rmse"] = results[σ].rmse
        attributes(g)["sigma_log"] = σ
        attributes(g)["n_resample"] = results[σ].n_resample
    end
end
println("Saved → $(joinpath(RMSE_OUT, "exp14_logbeta_sigma_sweep.h5"))")
println("\nGoal 1 complete — measurement only, SIGMA_JITTER and process noise untouched.")
