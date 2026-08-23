# Experiment 17 — likelihood tempering on the 16-obs log_beta setup.
#
# Tests one specific prediction. The worst-case min ESS has stayed pinned at
# t = 1 in single digits through every knob so far (σ_log, prior width, obs
# count). Two readings are possible:
#   (A) t = 1 is a separate coverage problem (truth outside the cloud), OR
#   (B) t = 1 is the SAME weight-degeneracy mechanism as general collapse,
#       biting hardest at t = 1 because the ensemble is at its widest there
#       (raw prior, before any resampling has concentrated it) → largest
#       spread of log(β) across particles → largest joint log-likelihood
#       variance → sharpest one-shot weight collapse.
# Exp 16 already ruled out (A) as the ESS lever: widening the prior fixed
# bracketing yet made ESS worse. Reading (B) predicts that tempering — which
# applies the likelihood through graduated sub-steps and so never presents the
# ensemble with the full sharp update at once — should specifically lift the
# t = 1 min ESS, because t = 1 is exactly where the un-tempered update is
# sharpest.
#
# PREDICTION: tempering lifts t = 1 min ESS off single digits, and the min
# moves away from t = 1 / rises substantially ⇒ confirms (B). If min ESS stays
# pinned at t = 1 in single digits despite tempering ⇒ (B) is wrong, something
# genuinely separate happens at t = 1 — reported plainly either way.
#
# TEMPERING (Del Moral, Doucet & Jasra 2006, "Sequential Monte Carlo
# samplers"; the "tempering and mutation steps" named as future work in Giles
# et al. 2024, the ParticleDA paper). At each assimilation step the likelihood
# g(y|x) is introduced through K stages with exponents
#     0 = φ_0 < φ_1 < ... < φ_K = 1,
# incremental weight g(y|x)^{φ_k − φ_{k−1}} per stage, resampling (+ RPF jitter
# as a crude mutation) between stages when ESS < N/2. Fixed LINEAR schedule
# φ_k = k/K (equal increments 1/K). First fixed-K attempt only, K = 10; the
# schedule is deliberately NOT tuned yet.
#
# KEY IDENTITY: K = 1 (φ = [0,1], one full-likelihood stage) reproduces the
# exp15/16 filter exactly, so baseline and tempered run through one code path
# and differ only in K — no confound.
#
# CORRECTNESS: after any intra-step resample the particle positions change, so
# the log-likelihood is recomputed at the new positions before the next stage's
# increment (a stale-likelihood increment would be wrong). This is the source
# of the tempering compute cost.
#
# HONEST ESS: ess_series[t] = min over the K stages of the pre-resample ESS —
# the worst degeneracy actually experienced during the step. For K = 1 this is
# the single post-update ESS, identical to the baseline metric.
#
# Held fixed vs the 16-obs setup: N = 1000, T = 100, stations_grid_16.txt,
# init_std = 200, process σ = 10 / ℓ = 15 km, Lax–Wendroff, SIGMA_JITTER = 20,
# and σ_log = 0.40 (the ESS-stable value from Exp 15 — NOT run16's 0.20).
# Same 10 seeds as the earlier sweeps.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_17_tempering.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf, HDF5
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
const SIGMA_LOG = 0.40
const K_TEMPER  = 10        # first fixed-K attempt; K=1 is the baseline

base_glacier = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_grid_16.txt",
    "prior_mode" => "pseudo_random_wave",
    "prior_center_beta" => 2000.0,
    "prior_signal_scale_beta" => 300.0,
    "background_std_beta" => 300.0,
    "prior_max_wavenumber" => 2,
    "prior_truth_seed" => 11,
    "prior_background_seed" => 29,
    "init_std_beta" => 200.0,
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

# One PF run with K-stage likelihood tempering. K = 1 is the un-tempered
# baseline. Returns ess (min-over-stages per step), rmse, argmin timestep,
# wall time, and total resample-event count.
function run_pf_tempered(; K::Int, seed_pf::Int, seed_obs::Int)
    Glacier._CFL_WARNED[] = false
    model = Glacier.init(Dict("glacier" => base_glacier))
    n_state = ParticleDA.get_state_dimension(model)
    n_obs   = ParticleDA.get_observation_dimension(model)
    phi = collect(0:K) ./ K                       # φ_0..φ_K, φ_K = 1

    # ── truth + observations (identical for all K at a given seed) ────────
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

    # ── PF with tempering ────────────────────────────────────────────────
    wall_t0 = time()
    rng_pf = MersenneTwister(seed_pf)
    particles = zeros(n_state, NPRT)
    for p in 1:NPRT
        ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
    end
    ensemble_mean = zeros(n_state, T+1)
    ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]
    ess_series  = zeros(T)
    log_weights = zeros(NPRT)
    loglik      = zeros(NPRT)
    n_resample  = 0

    for t in 1:T
            for p in 1:NPRT
                sp = view(particles, :, p)
                ParticleDA.update_state_deterministic!(sp, model, t)
                ParticleDA.update_state_stochastic!(sp, model, rng_pf)
            end
            y_t = view(observations, :, t)
            for p in 1:NPRT
                loglik[p] = ParticleDA.get_log_density_observation_given_state(
                    y_t, view(particles, :, p), model)
            end

            stage_min_ess = Inf
            for k in 1:K
                dphi = phi[k+1] - phi[k]                 # = 1/K
                log_weights .+= dphi .* loglik
                lmax = maximum(log_weights)
                w = exp.(log_weights .- lmax); w ./= sum(w)
                ess_k = 1.0 / sum(w .^ 2)
                stage_min_ess = min(stage_min_ess, ess_k)

                if k == K
                    # posterior estimate with full-likelihood weights,
                    # before the final-stage resample (matches baseline).
                    ensemble_mean[:, t+1] = particles * w
                end

                if ess_k < ESS_THRESHOLD
                    idx = systematic_resample(w, rng_pf)
                    particles = particles[:, idx]
                    if SIGMA_JITTER > 0
                        for p in 1:NPRT
                            Glacier._apply_noise!(view(particles, :, p), model,
                                                  rng_pf, SIGMA_JITTER, 1)
                        end
                    end
                    log_weights .= 0.0
                    n_resample += 1
                    if k < K                              # positions moved →
                        for p in 1:NPRT                   # refresh likelihood
                            loglik[p] = ParticleDA.get_log_density_observation_given_state(
                                y_t, view(particles, :, p), model)
                        end
                    end
                end
            end
            ess_series[t] = stage_min_ess
        end
    wall = time() - wall_t0

    rmse = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2))
            for t in 1:T+1]
    ess_min, ess_argmin = findmin(ess_series)
    return (ess=ess_series, rmse=rmse, n_resample=n_resample, wall=wall,
            min_ess=ess_min, argmin_t=ess_argmin)
end

println("=== Experiment 17: likelihood tempering, 16-obs log_beta, σ_log=$SIGMA_LOG ===")
println("Baseline K=1 (no tempering) vs tempered K=$K_TEMPER (linear schedule φ_k=k/K)")
println("Seeds: SEED_PF=42+s, SEED_OBS=123+s for s=$(collect(SEEDS))")
println("Held fixed: N=$NPRT, T=$T, 16 sensors, init_std=200, σ_proc=10, ℓ=15km, SIGMA_JITTER=$SIGMA_JITTER\n")

res = Dict(1 => Dict{Int,Any}(), K_TEMPER => Dict{Int,Any}())
for s in SEEDS
    for K in (1, K_TEMPER)
        r = run_pf_tempered(K=K, seed_pf=42+s, seed_obs=123+s)
        res[K][s] = r
        @printf("K=%-2d s=%-2d  minESS=%-6.1f @t=%-3d  final=%-6.1f last10=%-6.1f  %.0fs\n",
                K, s, r.min_ess, r.argmin_t, r.rmse[end],
                mean(r.rmse[end-9:end]), r.wall)
    end
end

_ms(v) = (mean(v), std(v))
function summarize(K)
    R = res[K]
    miness   = [R[s].min_ess for s in SEEDS]
    argmins  = [R[s].argmin_t for s in SEEDS]
    meaness  = [mean(R[s].ess) for s in SEEDS]
    below2   = [count(<(NPRT/2), R[s].ess) for s in SEEDS]
    below10  = [count(<(NPRT/10), R[s].ess) for s in SEEDS]
    finals   = [R[s].rmse[end] for s in SEEDS]
    last10   = [mean(R[s].rmse[end-9:end]) for s in SEEDS]
    walls    = [R[s].wall for s in SEEDS]
    return (; miness, argmins, meaness, below2, below10, finals, last10, walls)
end
b = summarize(1); tt = summarize(K_TEMPER)

println("\n=== Baseline (K=1) vs Tempered (K=$K_TEMPER) ===\n")
@printf("%-26s %18s %18s\n", "metric", "K=1 (baseline)", "K=$K_TEMPER (tempered)")
@printf("%-26s %18.1f %18.1f\n", "WORST min ESS (over seeds)", minimum(b.miness), minimum(tt.miness))
mb,sb=_ms(b.miness); mt,st=_ms(tt.miness)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "mean min ESS", mb,sb,mt,st)
mb,sb=_ms(b.meaness); mt,st=_ms(tt.meaness)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "mean ESS", mb,sb,mt,st)
mb,sb=_ms(b.below2); mt,st=_ms(tt.below2)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "steps < N/2", mb,sb,mt,st)
mb,sb=_ms(b.below10); mt,st=_ms(tt.below10)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "steps < N/10", mb,sb,mt,st)
mb,sb=_ms(b.finals); mt,st=_ms(tt.finals)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "final RMSE", mb,sb,mt,st)
mb,sb=_ms(b.last10); mt,st=_ms(tt.last10)
@printf("%-26s %9.1f ± %-6.1f %9.1f ± %-6.1f\n", "last-10-mean RMSE", mb,sb,mt,st)

println("\n=== The t=1 prediction ===\n")
@printf("Baseline  : WORST min ESS = %.1f, occurring at t = %d\n",
        minimum(b.miness), b.argmins[argmin(b.miness)])
@printf("            min-ESS timestep per seed: %s\n", string(b.argmins))
@printf("            seeds with min at t=1: %d/%d\n", count(==(1), b.argmins), length(SEEDS))
@printf("Tempered  : WORST min ESS = %.1f, occurring at t = %d\n",
        minimum(tt.miness), tt.argmins[argmin(tt.miness)])
@printf("            min-ESS timestep per seed: %s\n", string(tt.argmins))
@printf("            seeds with min at t=1: %d/%d\n", count(==(1), tt.argmins), length(SEEDS))

worst_lift = minimum(tt.miness) - minimum(b.miness)
moved_off_t1 = count(==(1), tt.argmins) < count(==(1), b.argmins)
println()
if minimum(tt.miness) > NPRT/10 || (worst_lift > 20 && moved_off_t1)
    println("VERDICT: prediction (B) HELD — tempering lifted the worst-case min ESS")
    println("  and/or moved it off t=1. t=1 collapse is peak-τ² weight degeneracy,")
    println("  the same mechanism as general collapse, not a separate coverage problem.")
elseif minimum(tt.miness) <= 12 && count(==(1), tt.argmins) >= count(==(1), b.argmins)
    println("VERDICT: prediction (B) did NOT hold — min ESS stays pinned at t=1 in")
    println("  single digits despite tempering. Something genuinely separate happens")
    println("  at t=1; the peak-τ² reinterpretation is not sufficient.")
else
    println("VERDICT: MIXED — worst min ESS lift = $(round(worst_lift,digits=1)); inspect")
    println("  the per-seed timesteps above before concluding.")
end

# ── Compute cost ─────────────────────────────────────────────────────────
println("\n=== Compute cost ===\n")
@printf("mean wall/run  : K=1 %.0fs   K=%d %.0fs   ratio %.2fx\n",
        mean(b.walls), K_TEMPER, mean(tt.walls), mean(tt.walls)/mean(b.walls))
@printf("mean resamples : K=1 %.1f    K=%d %.1f\n",
        mean([res[1][s].n_resample for s in SEEDS]),
        K_TEMPER, mean([res[K_TEMPER][s].n_resample for s in SEEDS]))

# ── Plots ────────────────────────────────────────────────────────────────
# ESS trace on the seed with the worst BASELINE min ESS, baseline vs tempered.
worst_seed = SEEDS[argmin(b.miness)]
rb = res[1][worst_seed]; rt = res[K_TEMPER][worst_seed]
p1 = plot(; xlabel="timestep", ylabel="min-stage ESS", ylim=(0, NPRT),
          title="Exp 17 — ESS, worst baseline seed (s=$worst_seed)", legend=:topright)
plot!(p1, 1:T, rb.ess; lw=2, color=:firebrick, label="K=1 baseline")
plot!(p1, 1:T, rt.ess; lw=2, color=:seagreen, label="K=$K_TEMPER tempered")
hline!(p1, [ESS_THRESHOLD]; ls=:dash, color=:gray, label="N/2")
hline!(p1, [NPRT/10]; ls=:dot, color=:black, label="N/10")
scatter!(p1, [rb.argmin_t], [rb.min_ess]; color=:firebrick, ms=7, marker=:diamond,
         label=@sprintf("baseline min %.0f @t=%d", rb.min_ess, rb.argmin_t))
scatter!(p1, [rt.argmin_t], [rt.min_ess]; color=:seagreen, ms=7, marker=:diamond,
         label=@sprintf("tempered min %.0f @t=%d", rt.min_ess, rt.argmin_t))
savefig(p1, joinpath(RMSE_OUT, "ess_tempering_exp17.png"))

# Per-seed worst min ESS, baseline vs tempered (paired).
p2 = plot(; xlabel="seed s", ylabel="min ESS over run",
          title="Exp 17 — per-seed min ESS: tempering lift", legend=:topright)
scatter!(p2, collect(SEEDS), b.miness; color=:firebrick, ms=6, label="K=1 baseline")
scatter!(p2, collect(SEEDS), tt.miness; color=:seagreen, ms=6, label="K=$K_TEMPER tempered")
for s in SEEDS
    plot!(p2, [s, s], [b.miness[s], tt.miness[s]]; color=:gray, alpha=0.5, label=false)
end
hline!(p2, [NPRT/10]; ls=:dot, color=:black, label="N/10 = 100")
savefig(p2, joinpath(RMSE_OUT, "min_ess_tempering_by_seed_exp17.png"))

h5open(joinpath(RMSE_OUT, "exp17_tempering.h5"), "w") do f
    for (K, tag) in ((1, "baseline_K1"), (K_TEMPER, "tempered_K$(K_TEMPER)"))
        g = create_group(f, tag)
        for s in SEEDS
            gs = create_group(g, "seed_$s")
            gs["ess"]  = res[K][s].ess
            gs["rmse"] = res[K][s].rmse
        end
    end
end
println("\nSaved → ess_tempering_exp17.png, min_ess_tempering_by_seed_exp17.png, exp17_tempering.h5")
println("Done.")
