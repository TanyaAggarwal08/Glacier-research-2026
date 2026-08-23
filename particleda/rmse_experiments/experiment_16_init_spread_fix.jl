# Experiment 16 — spread fix: does widening the prior lift worst-case min ESS?
#
# Exp 14/15 diagnosis: σ_log tuning lifts *median* ESS but leaves the
# worst-case collapse intact (min ESS down to 9–11 on bad seeds), and the
# minimum lands on the FIRST assimilation. Cause is a prior-mismatch, not an
# observation-noise problem: background_std_beta = 300 > init_std_beta = 200,
# so the swarm is centred ~300 off truth but only spreads 200 — the truth
# starts ~1.5σ OUTSIDE the ensemble and no σ_log can rescue an ensemble that
# doesn't bracket the truth.
#
# Fix under test: make the swarm at least as wide as it is off-centre, by
# raising init_std_beta to ≥ background_std_beta. Sweep init_std ∈ {300, 400}
# (= background, and wider-than-background) across the same 10 seeds as Exp 15.
# init_std = 200 baseline is REUSED from exp15_seed_sweep.h5 (its log arm is
# exactly this config at init_std = 200), so only the new widths are run.
#
# Why this is a clean test: the truth starts deterministically from
# truth_prior_mean (copy + clamp), so init_std_beta touches only the particle
# cloud, never the truth. Widening the prior cannot move the target, and the
# seed pairing is unaffected.
#
# Operator + all else pinned to Exp 15: log_beta, σ_log = 0.40, N = 1000,
# T = 100, 30 sensors, SIGMA_JITTER = 20.0, process_std_beta = 10.0,
# background_std_beta = 300. Only init_std_beta varies.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_16_init_spread_fix.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf, HDF5
include(joinpath(REPO_ROOT, "glacier-code", "particleda", "glacier_model.jl"))
using .Glacier
ENV["GKSwstype"] = "100"
using Plots

const RMSE_OUT = joinpath("glacier-code", "particleda", "results", "rmse_analysis")
const EXP15_H5 = joinpath(RMSE_OUT, "exp15_seed_sweep.h5")

const NPRT = 1000
const T    = 100
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0
const SEEDS   = 1:10
const SIGMA_LOG = 0.40
const BACKGROUND_STD = 300.0
const INIT_STD_NEW = [300.0, 400.0]   # 200 reused from exp15
const INIT_STD_BASELINE = 200.0

base_glacier = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection_30.txt",
    "prior_mode" => "pseudo_random_wave",
    "prior_center_beta" => 2000.0,
    "prior_signal_scale_beta" => 300.0,
    "background_std_beta" => BACKGROUND_STD,
    "prior_max_wavenumber" => 2,
    "prior_truth_seed" => 11,
    "prior_background_seed" => 29,
    "process_std_beta" => 10.0,
    "obs_space" => "log_beta",
    "obs_noise_std" => 0.10,
    "obs_noise_std_log" => SIGMA_LOG,
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

function run_logbeta(init_std::Float64; seed_pf::Int, seed_obs::Int)
    Glacier._CFL_WARNED[] = false
    params = merge(base_glacier, Dict{String, Any}("init_std_beta" => init_std))
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
    return (ess=ess_series, rmse=rmse, n_resample=n_resample,
            ess_first=ess_series[1])
end

# ── Baseline init_std = 200: reuse the Exp 15 log arm ────────────────────
results = Dict{Float64, Dict{Int, Any}}()
results[INIT_STD_BASELINE] = Dict{Int, Any}()
h5open(EXP15_H5, "r") do f
    for s in SEEDS
        ess  = read(f["log_beta"]["seed_$s"]["ess"])
        rmse = read(f["log_beta"]["seed_$s"]["rmse"])
        results[INIT_STD_BASELINE][s] = (ess=ess, rmse=rmse,
            n_resample=count(<(ESS_THRESHOLD), ess), ess_first=ess[1])
    end
end
println("Reused init_std=200 baseline (log arm) from exp15_seed_sweep.h5\n")

println("=== Experiment 16: init_std_beta spread fix (log_beta, σ_log=$SIGMA_LOG) ===")
println("background_std_beta = $BACKGROUND_STD held fixed; init_std swept.")
println("Seeds: SEED_PF=42+s, SEED_OBS=123+s for s=$(collect(SEEDS))\n")

for init_std in INIT_STD_NEW
    results[init_std] = Dict{Int, Any}()
    for s in SEEDS
        t0 = time()
        r = run_logbeta(init_std; seed_pf=42+s, seed_obs=123+s)
        results[init_std][s] = r
        @printf("init_std=%-4.0f s=%-3d  final=%-6.1f last10=%-6.1f minESS=%-6.1f ESS(t=1)=%-6.1f  (%.0fs)\n",
                init_std, s, r.rmse[end], mean(r.rmse[end-9:end]),
                minimum(r.ess), r.ess_first, time()-t0)
    end
    println()
end

# ── Summary: the worst-case min ESS is the headline metric ───────────────
const ALL_INIT = vcat(INIT_STD_BASELINE, INIT_STD_NEW)
_ms(v) = (mean(v), std(v))
println("=== Summary across $(length(SEEDS)) seeds (mean ± std) ===\n")
@printf("%-10s %14s %14s %14s %14s %14s\n",
        "init_std", "WORST minESS", "mean minESS", "mean ESS(t=1)",
        "final RMSE", "last10 RMSE")
for init_std in ALL_INIT
    R = results[init_std]
    miness  = [minimum(R[s].ess) for s in SEEDS]
    first   = [R[s].ess_first for s in SEEDS]
    finals  = [R[s].rmse[end] for s in SEEDS]
    last10  = [mean(R[s].rmse[end-9:end]) for s in SEEDS]
    mf, sf = _ms(finals); ml, sl = _ms(last10)
    mfe, _ = _ms(first)
    @printf("%-10.0f %14.1f %9.1f±%-5.1f %14.1f %8.1f±%-5.1f %8.1f±%-5.1f\n",
            init_std, minimum(miness), mean(miness), std(miness),
            mfe, mf, sf, ml, sl)
end

println("\nWorst-case min ESS is min over seeds — the single number that says")
println("whether the collapse is cured. Off single digits ⇒ swarm now brackets truth.")

# ── Plots ────────────────────────────────────────────────────────────────
# Per-init min-ESS across seeds (strip plot), to show worst case + spread.
p1 = plot(; xlabel="init_std_beta (β units)", ylabel="min ESS over run",
          title="Exp 16 — worst-case ESS vs prior width (log β, σ_log=$SIGMA_LOG)",
          legend=:topright, xlim=(150, 450))
for init_std in ALL_INIT
    miness = [minimum(results[init_std][s].ess) for s in SEEDS]
    scatter!(p1, fill(init_std, length(SEEDS)), miness;
             label=false, color=:steelblue, alpha=0.6, ms=5)
    scatter!(p1, [init_std], [minimum(miness)];
             label=(init_std==ALL_INIT[1] ? "worst seed" : false),
             color=:firebrick, ms=8, marker=:diamond)
end
plot!(p1, ALL_INIT, [mean([minimum(results[i][s].ess) for s in SEEDS]) for i in ALL_INIT];
      label="mean over seeds", lw=2.5, color=:black, marker=:square)
hline!(p1, [NPRT/10]; linestyle=:dot, color=:gray, label="N/10 = 100")
vline!(p1, [BACKGROUND_STD]; linestyle=:dash, color=:goldenrod,
       label="background_std = $BACKGROUND_STD")
savefig(p1, joinpath(RMSE_OUT, "min_ess_vs_init_std_exp16.png"))

p2 = plot(; xlabel="time (h)", ylabel="global RMSE(β)",
          title="Exp 16 — seed-mean RMSE vs prior width", legend=:topright)
const CLR = Dict(200.0=>:firebrick, 300.0=>:goldenrod, 400.0=>:seagreen)
for init_std in ALL_INIT
    mean_rmse = [mean([results[init_std][s].rmse[t] for s in SEEDS]) for t in 1:T+1]
    plot!(p2, 0:T, mean_rmse; lw=3, color=CLR[init_std],
          label=@sprintf("init_std=%.0f (final %.0f)", init_std, mean_rmse[end]))
end
savefig(p2, joinpath(RMSE_OUT, "rmse_vs_init_std_exp16.png"))

h5open(joinpath(RMSE_OUT, "exp16_init_spread.h5"), "w") do f
    for init_std in INIT_STD_NEW
        g = create_group(f, "init_$(Int(init_std))")
        for s in SEEDS
            gs = create_group(g, "seed_$s")
            gs["rmse"] = results[init_std][s].rmse
            gs["ess"]  = results[init_std][s].ess
        end
    end
end
println("\nSaved → min_ess_vs_init_std_exp16.png, rmse_vs_init_std_exp16.png, exp16_init_spread.h5")
println("Done.")
