# Particle-tracking driver — LINEAR ADVECTION + NO OBSERVATIONS.
#
# Same as run_linear_advection.jl except disable_observations: true is set,
# so get_log_density returns 0 → uniform weights → filter ignores the data.
# Truth path is unaffected. Outputs go to run09_no_obs_linear_adv/.
#
# Run: julia --project=test glacier-code/particleda/run_no_obs_linear_adv.jl

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
                     "run09_no_obs_linear_adv")
mkpath(OUT)
const EXT_OUT = "/Volumes/ZX20/USRA 2026/run09_no_obs_linear_adv"
mkpath(EXT_OUT)

yaml_params = Dict("glacier" => Dict(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,        # scaled with prior amplitude (4× old)
    "process_std_beta" => 28.0,      # scaled with prior amplitude (4× old)
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,       # unused by the linear branch
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "disable_observations" => true,    # ← ABLATION: filter ignores all data
))
model = Glacier.init(yaml_params)
n_state = ParticleDA.get_state_dimension(model)
n_obs   = ParticleDA.get_observation_dimension(model)
nx, ny  = model.parameters.nx, model.parameters.ny

println("Setup (LINEAR ADVECTION): nprt=$NPRT, T=$T, n_state=$n_state, n_obs=$n_obs")

probe_flat = [(ic-1)*ny + jr for (ic, jr) in PROBE_CELLS_IJ]
n_probes = length(probe_flat)

# ---- 1) Truth path ----
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
    for p in 1:NPRT
        sp = view(particles, :, p)
        ParticleDA.update_state_deterministic!(sp, model, t)
        ParticleDA.update_state_stochastic!(sp, model, rng_pf)
    end

    y_t = view(observations, :, t)
    logw = zeros(NPRT)
    for p in 1:NPRT
        logw[p] = ParticleDA.get_log_density_observation_given_state(
            y_t, view(particles, :, p), model)
    end
    lmax = maximum(logw)
    w = exp.(logw .- lmax)
    w ./= sum(w)
    weights_raw[:, t] = w
    ess_series[t] = 1.0 / sum(w .^ 2)

    all_particles[:, :, t+1] = particles

    idx = systematic_resample(w, rng_pf)
    particles = particles[:, idx]

    if t % 10 == 0
        println("  step $t / $T  ESS=$(round(ess_series[t], digits=1))")
    end
end

ensemble_mean = zeros(n_state, T+1)
for t in 1:T+1
    ensemble_mean[:, t] = mean(all_particles[:, :, t]; dims=2)[:, 1]
end

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

    g_par = create_group(f, "parameters")
    attributes(g_par)["nprt"] = NPRT
    attributes(g_par)["T"]    = T
    attributes(g_par)["time_step"] = model.parameters.time_step
    attributes(g_par)["nx"]   = nx
    attributes(g_par)["ny"]   = ny
    attributes(g_par)["advection_type"] = "linear"
end
println("\nSaved tracking data → $out_path")
