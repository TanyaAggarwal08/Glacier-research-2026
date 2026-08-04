# Shared helpers for RMSE tuning experiments.
#
# Provides:
#   run_pf_rmse(params; NPRT, T, SEED_PF, SEED_OBS)
#       → NamedTuple with model_time, rmse_global, ess, sensor_indices, …
#   log_experiment!(; figure, model_type, …, notes)
#       → appends a row to glacier-notes/rmse_experiment_log.md
#
# Light variant of run_particle_tracking.jl: keeps only the ensemble mean
# (per cell, per timestep) — no per-particle history dump, so it runs ~5×
# faster and needs no external SSD.

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf

const REPO_ROOT_HELPERS = realpath(joinpath(@__DIR__, "..", "..", ".."))
const LOG_PATH = joinpath(REPO_ROOT_HELPERS, "glacier-notes",
                          "rmse_experiment_log.md")
const RMSE_OUT = joinpath(REPO_ROOT_HELPERS, "glacier-code", "particleda",
                          "results", "rmse_analysis")
mkpath(RMSE_OUT)

# Load the Glacier model only once across multiple experiment scripts.
if !isdefined(Main, :Glacier)
    include(joinpath(REPO_ROOT_HELPERS, "glacier-code", "particleda",
                     "glacier_model.jl"))
    using .Glacier
end

# Systematic resampling (Douc-Cappé).
function _systematic_resample(w::AbstractVector, rng::AbstractRNG)
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

"""
    run_pf_rmse(model_params; NPRT, T, SEED_PF, SEED_OBS)

Run a bootstrap PF on the Glacier model with the given parameter dict.
Returns a NamedTuple with:
  - model_time   :: Vector{Float64}  (length T+1, hours, starts at 0)
  - rmse_global  :: Vector{Float64}  (length T+1, RMSE of ensemble mean vs truth)
  - ess          :: Vector{Float64}  (length T)
  - sensor_idx   :: Vector{Int}
  - meta         :: Dict             (the param dict for downstream logging)
"""
function run_pf_rmse(model_params::Dict;
                     NPRT::Int=1000, T::Int=100,
                     SEED_PF::Int=42, SEED_OBS::Int=123,
                     return_trajectories::Bool=false,
                     ess_threshold_frac::Float64=0.5,
                     sigma_jitter::Float64=30.0)
    # ParticleDA expects the params under the "glacier" key.
    Glacier._CFL_WARNED[] = false       # let CFL warning fire once per run
    model = Glacier.init(Dict("glacier" => model_params))
    n_state = ParticleDA.get_state_dimension(model)
    n_obs   = ParticleDA.get_observation_dimension(model)

    # ── truth + observations ─────────────────────────────────────────────
    rng_truth = MersenneTwister(SEED_OBS)
    truth_states = zeros(n_state, T+1)
    observations = zeros(n_obs, T)
    s_truth = zeros(n_state)
    ParticleDA.sample_initial_state!(s_truth, model, rng_truth)
    truth_states[:, 1] = s_truth
    for t in 1:T
        ParticleDA.update_state_deterministic!(s_truth, model, t)
        ParticleDA.update_state_stochastic!(s_truth, model, rng_truth)
        truth_states[:, t+1] = s_truth
        y = zeros(n_obs)
        ParticleDA.sample_observation_given_state!(y, s_truth, model, rng_truth)
        observations[:, t] = y
    end

    # ── PF loop ──────────────────────────────────────────────────────────
    rng_pf = MersenneTwister(SEED_PF)
    particles = zeros(n_state, NPRT)
    for p in 1:NPRT
        ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
    end
    ensemble_mean = zeros(n_state, T+1)
    ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]
    ess_series = zeros(T)

    # ── run10 PF recipe ──────────────────────────────────────────────────
    # - Cumulative log-weights carried across steps (Liu & Chen 1998).
    # - Resample only when ESS drops below `ess_threshold_frac · N`.
    # - Post-resample RPF jitter (Musso et al. 2001) with marginal std
    #   `sigma_jitter` (β units), using the model's smooth Cholesky noise.
    log_weights   = zeros(NPRT)
    ess_threshold = ess_threshold_frac * NPRT
    n_resample    = 0

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
        ensemble_mean[:, t+1] = (particles * w)
        if ess_series[t] < ess_threshold
            idx = _systematic_resample(w, rng_pf)
            particles = particles[:, idx]
            if sigma_jitter > 0
                for p in 1:NPRT
                    Glacier._apply_noise!(view(particles, :, p), model, rng_pf,
                                          sigma_jitter, 1)
                end
            end
            log_weights .= 0.0
            n_resample += 1
        end
    end

    # ── global RMSE of ensemble mean vs truth, per timestep ──────────────
    rmse_global = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2))
                   for t in 1:T+1]
    model_time_h = collect(0:T) .* model.parameters.time_step ./ 3600.0

    if return_trajectories
        return (
            model_time    = model_time_h,
            rmse_global   = rmse_global,
            ess           = ess_series,
            sensor_idx    = copy(model.sensor_indices),
            meta          = model_params,
            truth_states  = truth_states,
            ensemble_mean = ensemble_mean,
        )
    end
    return (
        model_time  = model_time_h,
        rmse_global = rmse_global,
        ess         = ess_series,
        sensor_idx  = copy(model.sensor_indices),
        meta        = model_params,
    )
end

# ─── Markdown log ───────────────────────────────────────────────────────
const LOG_HEADER = """
# RMSE experiment log

Every figure in `results/rmse_analysis/` is documented here with the exact
parameters used to produce it. Each table row corresponds to one data series
on a plot (so a "linear vs nonlinear" comparison contributes two rows that
share the same `Figure Name`).

| Figure Name | Model Type | σ_init (β) | σ_proc (β) | Correlation ℓ (km) | Particles N | # Observations | Obs Interval (s) | σ_obs (ux) | Prior (base + amp) | Modes | T steps | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
"""

function _ensure_log_initialised()
    if !isfile(LOG_PATH)
        open(LOG_PATH, "w") do io
            print(io, LOG_HEADER)
        end
    end
end

"""
    log_experiment!(; figure, model_type, sigma_init, sigma_proc, length_scale_km,
                     particles, n_obs, obs_interval_s, sigma_obs,
                     prior="2000 + 2000·sin·sin", modes=3, T_steps=100, notes="")

Append a single row to `glacier-notes/rmse_experiment_log.md`. Creates the
file with header if it doesn't yet exist.
"""
function log_experiment!(;
        figure::String, model_type::String,
        sigma_init::Real, sigma_proc::Real, length_scale_km::Real,
        particles::Int, n_obs::Int, obs_interval_s::Real, sigma_obs::Real,
        prior::String="2000 + 2000·sin·sin", modes::Int=3, T_steps::Int=100,
        notes::String="")
    _ensure_log_initialised()
    row = @sprintf("| `%s` | %s | %.1f | %.2f | %.1f | %d | %d | %.0f | %.3f | %s | %d | %d | %s |\n",
        figure, model_type, sigma_init, sigma_proc, length_scale_km,
        particles, n_obs, obs_interval_s, sigma_obs,
        prior, modes, T_steps, notes)
    open(LOG_PATH, "a") do io
        print(io, row)
    end
    return row
end
