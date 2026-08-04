# Particle-tracking driver for the pseudo-random wave prior/background setup.
#
# This mirrors run_particle_tracking.jl, but swaps the sinusoidal double-bump
# prior for an Evensen-style pseudo-random wave field:
#   - truth initial state = one normalized random wave realization
#   - particle initial mean = an independent background realization
#   - process noise = smooth correlated perturbations on top
#
# Outputs go to run11_pseudorandom_wave/.
#
# Run: julia --project=test glacier-code/particleda/run_pseudorandom_wave.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA, HDF5, Random, Statistics, LinearAlgebra, PDMats, Printf
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier

# ---- knobs ----
const NPRT      = 1000
const T         = 100
const K_TRACK   = 15
const SEED_PF   = 42
const SEED_OBS  = 123
const PROBE_CELLS_IJ = [(20, 20), (10, 10), (10, 30), (5, 15)]

const OUT = joinpath("glacier-code", "particleda", "results",
                     "run11_pseudorandom_wave")
mkpath(OUT)
const _SSD_BASE = "/Volumes/ZX20/USRA 2026"
const EXT_OUT = let candidate = joinpath(_SSD_BASE, "run11_pseudorandom_wave")
    try
        mkpath(candidate)
        candidate
    catch
        OUT
    end
end
EXT_OUT != OUT && mkpath(EXT_OUT)

yaml_params = Dict("glacier" => Dict(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "prior_mode" => "pseudo_random_wave",
    "prior_center_beta" => 2000.0,
    "prior_signal_scale_beta" => 300.0,
    "background_std_beta" => 300.0,
    "prior_max_wavenumber" => 2,
    "prior_truth_seed" => 11,
    "prior_background_seed" => 29,
    "init_std_beta" => 200.0,        # ensemble spread around the background
    "process_std_beta" => 10.0,      # ~2% of the signal scale per step
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 0.0,      # strictly linear pseudo-random wave advection
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "lax_wendroff",
))
model = Glacier.init(yaml_params)
n_state = ParticleDA.get_state_dimension(model)
n_obs   = ParticleDA.get_observation_dimension(model)
nx, ny  = model.parameters.nx, model.parameters.ny

println("Setup (PSEUDORANDOM WAVE): nprt=$NPRT, T=$T, n_state=$n_state, n_obs=$n_obs")

probe_flat = [(ic-1)*ny + jr for (ic, jr) in PROBE_CELLS_IJ]
n_probes = length(probe_flat)

# ---- 1) Truth path ----
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

# ---- 2) Bootstrap PF ----
rng_pf = MersenneTwister(SEED_PF)
particles = zeros(n_state, NPRT)
for p in 1:NPRT
    ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
end

all_particles = zeros(n_state, NPRT, T+1)
all_particles[:, :, 1] = particles
weights_raw  = zeros(NPRT, T)
ess_series   = zeros(T)

# Cumulative log-weights between resampling events (Liu & Chen 1998).
log_weights = zeros(NPRT)
const ESS_THRESHOLD = 0.5 * NPRT

# Regularized PF jitter after resampling. Keeps the pseudo-random field
# ensemble from collapsing to repeated copies of one wave realisation.
const SIGMA_JITTER = 20.0

ensemble_mean = zeros(n_state, T+1)
ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]

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

for t in 1:T
    global particles, log_weights
    # Predict.
    for p in 1:NPRT
        sp = view(particles, :, p)
        ParticleDA.update_state_deterministic!(sp, model, t)
        ParticleDA.update_state_stochastic!(sp, model, rng_pf)
    end

    # Update weights.
    y_t = view(observations, :, t)
    for p in 1:NPRT
        log_weights[p] += ParticleDA.get_log_density_observation_given_state(
            y_t, view(particles, :, p), model)
    end

    # Normalise + ESS.
    lmax = maximum(log_weights)
    w = exp.(log_weights .- lmax)
    w ./= sum(w)
    weights_raw[:, t] = w
    ess_series[t] = 1.0 / sum(w .^ 2)

    # Snapshot + weighted mean.
    all_particles[:, :, t+1] = particles
    ensemble_mean[:, t+1] = particles * w

    # ESS-gated resampling + regularisation.
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
    end

    if t % 10 == 0
        println("  step $t / $T  ESS=$(round(ess_series[t], digits=1))")
    end
end

println("Total resampling threshold = $(ESS_THRESHOLD).")

# Track winners / median / losers
final_w = weights_raw[:, end]
order = sortperm(final_w; rev=true)
top_k    = order[1:5]
mid_lo   = NPRT ÷ 2 - 2
med_k    = order[mid_lo : mid_lo + 4]
bottom_k = order[end-4 : end]
tracked  = vcat(top_k, med_k, bottom_k)

println("\nTracked particles (idx, final-step weight):")
for p in tracked
    @printf "  idx=%3d  w_final=%.4f\n" p final_w[p]
end

# ---- 3) Save ----
out_path = joinpath(EXT_OUT, "tracking.h5")
isfile(out_path) && rm(out_path)
h5open(out_path, "w") do f
    g_truth = create_group(f, "truth")
    g_truth["beta"] = reshape(truth_states, ny, nx, T+1)

    g_mean = create_group(f, "ensemble_mean")
    g_mean["beta"] = reshape(ensemble_mean, ny, nx, T+1)

    g_all = create_group(f, "particles_all")
    g_all["beta"] = reshape(all_particles, ny, nx, NPRT, T+1)

    g_part = create_group(f, "particles_full")
    g_part["beta"] = reshape(all_particles[:, tracked, :], ny, nx, length(tracked), T+1)
    g_part["indices"] = collect(tracked)

    g_probe = create_group(f, "probe_cells")
    probe_beta = zeros(NPRT, T+1, n_probes)
    for k in 1:n_probes
        probe_beta[:, :, k] = all_particles[probe_flat[k], :, :]
    end
    g_probe["beta"] = probe_beta
    g_probe["indices"] = collect(probe_flat)
    g_probe["ij_pairs"] = reshape([x for ij in PROBE_CELLS_IJ for x in ij], 2, n_probes)

    g_w = create_group(f, "weights")
    g_w["raw"] = weights_raw
    g_w["ess"] = ess_series

    sensor_xs = Float64[]; sensor_ys = Float64[]
    dx = model.parameters.x_length / nx
    dy = model.parameters.y_length / ny
    for idx in model.sensor_indices
        j = ((idx - 1) % ny) + 1
        i = ((idx - 1) ÷ ny) + 1
        push!(sensor_xs, (i - 1) * dx)
        push!(sensor_ys, (j - 1) * dy)
    end
    f["sensor_indices"] = collect(model.sensor_indices)
    f["sensor_x"] = sensor_xs
    f["sensor_y"] = sensor_ys
    f["observations"] = observations

    f["beta_truth_prior"] = reshape(model.truth_prior_mean, ny, nx)
    f["beta_background_prior"] = reshape(model.beta_prior_mean, ny, nx)

    g_par = create_group(f, "parameters")
    attributes(g_par)["nprt"] = NPRT
    attributes(g_par)["T"]    = T
    attributes(g_par)["time_step"] = model.parameters.time_step
    attributes(g_par)["nx"]   = nx
    attributes(g_par)["ny"]   = ny
    attributes(g_par)["advection_type"] = "lax_wendroff"
    attributes(g_par)["prior_mode"] = model.parameters.prior_mode
end
println("\nSaved tracking data → $out_path")
