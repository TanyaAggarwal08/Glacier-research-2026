# run16 — log-β observation operator, pseudo-random wave, 16 observations.
#
# Identical to run14 (run_pseudorandom_wave_30obs_logbeta.jl) EXCEPT the sensor
# count drops 30 -> 16 via stations_grid_16.txt (an exact 4x4 subset of run14's
# grid). σ_log = 0.20 is held at the run14 value so this is a clean A/B that
# isolates the observation-count effect. Motivation: the joint log-likelihood
# sums squared residuals over all sensors, so its variance across particles —
# which is what collapses ESS in a bootstrap PF — scales with the effective
# observation count. Fewer near-independent observations ⇒ less weight
# degeneracy ⇒ higher ESS is the prediction under test.
#
# Produces ESS-vs-time and RMSE(β)-vs-time plots in
# results/run16_pseudorandom_wave_16obs_logbeta/. Lightweight: keeps only the
# ensemble mean, so no multi-GB particle dump.
#
# Run: julia --project=test glacier-code/particleda/run_pseudorandom_wave_16obs_logbeta.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA, HDF5, Random, Statistics, LinearAlgebra, PDMats, Printf
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier
ENV["GKSwstype"] = "100"
using Plots

const NPRT     = 1000
const T        = 100
const SEED_PF  = 42
const SEED_OBS = 123
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0

const OUT = joinpath("glacier-code", "particleda", "results",
                     "run16_pseudorandom_wave_16obs_logbeta")
mkpath(OUT)

# Exactly run14's parameters, with the 16-sensor station file swapped in.
params = Dict("glacier" => Dict(
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
    "obs_noise_std_log" => 0.20,
    "advection_epsilon" => 0.0,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "lax_wendroff",
))

model = Glacier.init(params)
n_state = ParticleDA.get_state_dimension(model)
n_obs   = ParticleDA.get_observation_dimension(model)
nx, ny  = model.parameters.nx, model.parameters.ny
@assert n_obs == 16 "expected 16 sensors, got $n_obs"
println("run16 (PSEUDORANDOM WAVE, 16 OBS, LOG-BETA): N=$NPRT, T=$T, n_obs=$n_obs, σ_log=$(model.parameters.obs_noise_std_log)")

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

# ── truth + observations (run14: truth starts deterministic) ─────────────
rng_truth = MersenneTwister(SEED_OBS)
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

# ── PF ───────────────────────────────────────────────────────────────────
rng_pf = MersenneTwister(SEED_PF)
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
    global particles, log_weights, n_resample
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
    t % 10 == 0 && println("  step $t/$T  ESS=$(round(ess_series[t], digits=1))")
end

# ── metrics ──────────────────────────────────────────────────────────────
rmse_beta = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2)) for t in 1:T+1]
t_axis_h  = collect(0:T) .* model.parameters.time_step ./ 3600.0
ess_min, ess_argmin = findmin(ess_series)

println("\n=== run16 summary (16 obs, log β, σ_log=0.20) ===")
@printf("  mean ESS   = %.1f\n", mean(ess_series))
@printf("  min  ESS   = %.1f @ step %d\n", ess_min, ess_argmin)
@printf("  steps<N/2  = %d/%d\n", count(<(ESS_THRESHOLD), ess_series), T)
@printf("  steps<N/10 = %d/%d\n", count(<(NPRT/10), ess_series), T)
@printf("  final RMSE = %.1f   mean RMSE = %.1f\n", rmse_beta[end], mean(rmse_beta))
println("  (run14 reference, 30 obs same seeds: mean ESS 321.1, min ESS 5.6 @ t=5, final RMSE 169.7)")

# ── plots ────────────────────────────────────────────────────────────────
p_ess = plot(1:T, ess_series;
             xlabel="timestep", ylabel="ESS", ylim=(0, NPRT),
             title="run16 — ESS (16 obs, log β, σ_log=0.20)",
             lw=2, color=:purple, label="ESS")
hline!(p_ess, [ESS_THRESHOLD]; linestyle=:dash, color=:gray, label="N/2 (resample trigger)")
hline!(p_ess, [mean(ess_series)]; linestyle=:dot, color=:darkgreen,
       label=@sprintf("mean = %.1f", mean(ess_series)))
scatter!(p_ess, [ess_argmin], [ess_min]; color=:red, ms=6,
         label=@sprintf("min = %.1f @ t=%d", ess_min, ess_argmin))
savefig(p_ess, joinpath(OUT, "ess_tracking.png"))

p_rmse = plot(t_axis_h, rmse_beta;
              xlabel="time (h)", ylabel="RMSE(β)",
              title="run16 — global RMSE (16 obs, log β, σ_log=0.20)",
              lw=2.5, color=:darkgreen, legend=false)
savefig(p_rmse, joinpath(OUT, "rmse_beta.png"))

h5open(joinpath(OUT, "metrics.h5"), "w") do f
    f["ess"] = ess_series
    f["rmse_beta"] = rmse_beta
    f["time_h"] = t_axis_h
    f["sensor_indices"] = collect(model.sensor_indices)
    attributes(f)["n_obs"] = n_obs
    attributes(f)["sigma_log"] = model.parameters.obs_noise_std_log
    attributes(f)["n_resample"] = n_resample
end

println("\nSaved → $(joinpath(OUT, "ess_tracking.png"))")
println("Saved → $(joinpath(OUT, "rmse_beta.png"))")
println("Saved → $(joinpath(OUT, "metrics.h5"))")
