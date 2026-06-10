# Particle-tracking driver.
#
# Runs a custom bootstrap PF (predict → weight → systematic resample) so we
# can dump per-particle state, not just the ensemble mean. ParticleDA's
# MeanAndVarSummaryStat only writes mean/var to HDF5 — useful for stats,
# useless for "what is particle 47 actually doing".
#
# What we save (results/run07_particle_tracking/tracking.h5):
#   /truth/beta                 (T+1, ny, nx)
#   /ensemble_mean/beta         (T+1, ny, nx)
#   /particles_full/beta        (K_track, T+1, ny, nx)   # K_track particles
#   /probe_cells/beta           (nprt, T+1, n_probes)    # all particles, few cells
#   /probe_cells/indices        (n_probes,) flat indices
#   /weights/raw                (nprt, T)                # right before resample
#   /weights/ess                (T,)
#   /tracked_indices/initial    (K_track,) — particle indices we chose to track
#   /sensor_indices             (n_sensors,)
#
# We pick which particles to track by running once with all-particle full
# fields cached in RAM (K=nprt=200, T=100, n=1600 → 256 MB, OK), then at end
# of run pick top-K and bottom-K by final-time weight and save THOSE only.

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA, HDF5, Random, Statistics, LinearAlgebra, PDMats, Printf
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier

# ---- knobs ----
const NPRT      = 1000         # canonical scale; all_particles takes ~1.3 GB RAM
const T         = 100          # 100 hourly steps ≈ 4 days
const K_TRACK   = 15           # how many particles to dump full fields for
const SEED_PF   = 42
const SEED_OBS  = 123
const PROBE_CELLS_IJ = [(20, 20), (10, 10), (10, 30), (5, 15)]   # (i_col, j_row)

const OUT = joinpath("glacier-code", "particleda", "results",
                     "run07_particle_tracking")
mkpath(OUT)
# Big HDF5 (~1.3 GB for 1000 particles × 1600 cells × 101 steps) goes to the
# external SSD so the repo stays small.
const EXT_OUT = "/Volumes/ZX20/USRA 2026/run07_particle_tracking"
mkpath(EXT_OUT)

# Build model.
yaml_params = Dict("glacier" => Dict(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 150.0,
    "process_std_beta" => 7.0,
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
))
model = Glacier.init(yaml_params)
n_state = ParticleDA.get_state_dimension(model)
n_obs   = ParticleDA.get_observation_dimension(model)
nx, ny  = model.parameters.nx, model.parameters.ny

println("Setup: nprt=$NPRT, T=$T, n_state=$n_state, n_obs=$n_obs, K_track=$K_TRACK")

# Probe cells: flat indices (column-major: idx = (i-1)*ny + j)
probe_flat = [(ic-1)*ny + jr for (ic, jr) in PROBE_CELLS_IJ]
n_probes = length(probe_flat)

# ---- 1) Simulate the TRUTH path ----
rng_truth = MersenneTwister(SEED_OBS)
truth_states = zeros(n_state, T+1)
observations = zeros(n_obs, T)        # obs at step t = 1..T (taken AFTER advection)

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

# ---- 2) Run the bootstrap PF, keeping ALL particle states ----
rng_pf = MersenneTwister(SEED_PF)
particles = zeros(n_state, NPRT)           # current particle states
for p in 1:NPRT
    ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
end

# Storage: full β fields per particle per timestep.
all_particles = zeros(n_state, NPRT, T+1)  # 1600*200*101 = 32M floats = 256 MB
all_particles[:, :, 1] = particles
weights_raw  = zeros(NPRT, T)
ess_series   = zeros(T)

# Systematic resampling (Douc–Cappé). Returns indices.
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
    global particles
    # Predict each particle (advect + process noise)
    for p in 1:NPRT
        sp = view(particles, :, p)
        ParticleDA.update_state_deterministic!(sp, model, t)
        ParticleDA.update_state_stochastic!(sp, model, rng_pf)
    end

    # Score (log-likelihood) for each particle
    y_t = view(observations, :, t)
    logw = zeros(NPRT)
    for p in 1:NPRT
        logw[p] = ParticleDA.get_log_density_observation_given_state(
            y_t, view(particles, :, p), model)
    end
    # Normalise: w = exp(logw - max) / sum
    lmax = maximum(logw)
    w = exp.(logw .- lmax)
    w ./= sum(w)
    weights_raw[:, t] = w
    ess_series[t] = 1.0 / sum(w .^ 2)

    # Store BEFORE resampling so trajectories per-particle make sense.
    # (After resampling, particle indices get scrambled.)
    all_particles[:, :, t+1] = particles

    # Resample.
    idx = systematic_resample(w, rng_pf)
    particles = particles[:, idx]

    if t % 10 == 0
        println("  step $t / $T  ESS=$(round(ess_series[t], digits=1))")
    end
end

# Ensemble mean (across particles), per timestep
ensemble_mean = zeros(n_state, T+1)
for t in 1:T+1
    ensemble_mean[:, t] = mean(all_particles[:, :, t]; dims=2)[:, 1]
end

# ---- 3) Pick "important" particles to dump full fields for ----
# Top-3 by final-step weight (winners), bottom-3 (losers), plus 2 median weight.
final_w = weights_raw[:, end]
order = sortperm(final_w; rev=true)
# K_TRACK = 15: 5 top, 5 median, 5 bottom
top_k    = order[1:5]
mid_lo   = NPRT ÷ 2 - 2
med_k    = order[mid_lo : mid_lo + 4]
bottom_k = order[end-4 : end]
tracked  = vcat(top_k, med_k, bottom_k)

println("\nTracked particles (idx, final-step weight):")
for p in tracked
    @printf "  idx=%3d  w_final=%.4f\n" p final_w[p]
end

# ---- 4) Save everything ----
out_path = joinpath(EXT_OUT, "tracking.h5")
isfile(out_path) && rm(out_path)
h5open(out_path, "w") do f
    # Truth
    g_truth = create_group(f, "truth")
    g_truth["beta"] = reshape(truth_states, ny, nx, T+1)

    # Ensemble mean
    g_mean = create_group(f, "ensemble_mean")
    g_mean["beta"] = reshape(ensemble_mean, ny, nx, T+1)

    # ALL particles' full fields (the big one — ~1.3 GB).
    g_all = create_group(f, "particles_all")
    g_all["beta"] = reshape(all_particles, ny, nx, NPRT, T+1)

    # Subset selection (top/median/bottom by final-step weight) for legacy plots.
    g_part = create_group(f, "particles_full")
    g_part["beta"] = reshape(all_particles[:, tracked, :], ny, nx, length(tracked), T+1)
    g_part["indices"] = collect(tracked)

    # Probe-cell trajectories for ALL particles
    g_probe = create_group(f, "probe_cells")
    probe_beta = zeros(NPRT, T+1, n_probes)
    for k in 1:n_probes
        probe_beta[:, :, k] = all_particles[probe_flat[k], :, :]
    end
    g_probe["beta"] = probe_beta
    g_probe["indices"] = collect(probe_flat)
    g_probe["ij_pairs"] = reshape([x for ij in PROBE_CELLS_IJ for x in ij], 2, n_probes)

    # Weights & ESS
    g_w = create_group(f, "weights")
    g_w["raw"] = weights_raw
    g_w["ess"] = ess_series

    # Sensors (flat indices + x,y in metres + the noisy observations themselves)
    sensor_xs = Float64[]; sensor_ys = Float64[]
    dx = model.parameters.x_length / nx
    dy = model.parameters.y_length / ny
    for idx in model.sensor_indices
        j = ((idx - 1) % ny) + 1            # row
        i = ((idx - 1) ÷ ny) + 1            # col
        push!(sensor_xs, (i - 1) * dx)
        push!(sensor_ys, (j - 1) * dy)
    end
    f["sensor_indices"] = collect(model.sensor_indices)
    f["sensor_x"] = sensor_xs
    f["sensor_y"] = sensor_ys
    f["observations"] = observations        # (n_obs, T)

    # Parameters (for reproducibility)
    g_par = create_group(f, "parameters")
    attributes(g_par)["nprt"] = NPRT
    attributes(g_par)["T"]    = T
    attributes(g_par)["time_step"] = model.parameters.time_step
    attributes(g_par)["nx"]   = nx
    attributes(g_par)["ny"]   = ny
end
println("\nSaved tracking data → $out_path")
println("  /truth/beta            shape (ny, nx, T+1)")
println("  /ensemble_mean/beta    shape (ny, nx, T+1)")
println("  /particles_full/beta   shape (ny, nx, K=$(length(tracked)), T+1)")
println("  /probe_cells/beta      shape (nprt=$NPRT, T+1, n_probes=$n_probes)")
println("  /weights/raw           shape (nprt, T)")
