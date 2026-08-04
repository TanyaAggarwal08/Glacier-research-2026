# run17 — showcase of likelihood tempering: 16-obs log_beta, K=10 tempering.
#
# Reproduces the Exp 17 tempered filter (K=10 linear schedule, σ_log = 0.40,
# 16 sensors) for the canonical seed (SEED_PF=42, SEED_OBS=123, the run14/run16
# seed) and stores the FULL particle trajectory so we can draw the β-vs-x
# cross-section. The particle dump (≈1.3 GB) goes to the external SSD; only the
# plots are kept in the local results folder.
#
# Tempering = Del Moral, Doucet & Jasra (2006) SMC samplers; the incremental
# likelihood g(y|x)^{Δφ} is applied through K stages φ_k = k/K with resampling
# (+ RPF jitter as a crude mutation) between stages when ESS < N/2. After any
# intra-step resample the log-likelihood is recomputed at the moved particle
# positions before the next stage's increment.
#
# Run: julia --project=test glacier-code/particleda/run_pseudorandom_wave_16obs_tempered_logbeta.jl

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
const K_TEMPER = 10
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0

const RUN_NAME = "run17_pseudorandom_wave_16obs_tempered_logbeta"
const OUT = joinpath("glacier-code", "particleda", "results", RUN_NAME)
mkpath(OUT)
# Full particle dump → external SSD; fall back to local only if unavailable.
const _SSD_BASE = "/Volumes/ZX20/USRA 2026"
const EXT_OUT = let candidate = joinpath(_SSD_BASE, RUN_NAME)
    try; mkpath(candidate); candidate; catch; OUT; end
end
println("Particle dump target: $EXT_OUT")

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
    "obs_noise_std_log" => 0.40,
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
phi = collect(0:K_TEMPER) ./ K_TEMPER
println("run17 (16 OBS, LOG-BETA, K=$K_TEMPER TEMPERING): N=$NPRT, T=$T, σ_log=0.40, seed=($SEED_PF,$SEED_OBS)")

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

# ── truth + observations ─────────────────────────────────────────────────
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

# ── tempered PF, storing every particle ──────────────────────────────────
rng_pf = MersenneTwister(SEED_PF)
particles = zeros(n_state, NPRT)
for p in 1:NPRT
    ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
end
all_particles = zeros(n_state, NPRT, T+1)     # ≈1.3 GB
all_particles[:, :, 1] = particles
ensemble_mean = zeros(n_state, T+1)
ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]
ess_series  = zeros(T)
log_weights = zeros(NPRT)
loglik      = zeros(NPRT)
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
        loglik[p] = ParticleDA.get_log_density_observation_given_state(
            y_t, view(particles, :, p), model)
    end

    stage_min_ess = Inf
    for k in 1:K_TEMPER
        dphi = phi[k+1] - phi[k]
        log_weights .+= dphi .* loglik
        lmax = maximum(log_weights)
        w = exp.(log_weights .- lmax); w ./= sum(w)
        ess_k = 1.0 / sum(w .^ 2)
        stage_min_ess = min(stage_min_ess, ess_k)
        if k == K_TEMPER
            ensemble_mean[:, t+1] = particles * w
        end
        if ess_k < ESS_THRESHOLD
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
            if k < K_TEMPER
                for p in 1:NPRT
                    loglik[p] = ParticleDA.get_log_density_observation_given_state(
                        y_t, view(particles, :, p), model)
                end
            end
        end
    end
    ess_series[t] = stage_min_ess
    all_particles[:, :, t+1] = particles        # post-analysis ensemble
    t % 10 == 0 && println("  step $t/$T  min-stage ESS=$(round(ess_series[t], digits=1))")
end

rmse_beta = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2)) for t in 1:T+1]
ess_min, ess_argmin = findmin(ess_series)
println("\nmean ESS=$(round(mean(ess_series),digits=1))  min ESS=$(round(ess_min,digits=1)) @t=$ess_argmin  final RMSE=$(round(rmse_beta[end],digits=1))")

# ── save full particle trajectory to SSD ─────────────────────────────────
out_path = joinpath(EXT_OUT, "tracking.h5")
isfile(out_path) && rm(out_path)
h5open(out_path, "w") do f
    f["truth/beta"]         = reshape(truth_states, ny, nx, T+1)
    f["ensemble_mean/beta"] = reshape(ensemble_mean, ny, nx, T+1)
    f["particles_all/beta"] = reshape(all_particles, ny, nx, NPRT, T+1)
    f["weights/ess"]        = ess_series
    f["observations"]       = observations
    f["beta_truth_prior"]      = reshape(model.truth_prior_mean, ny, nx)
    f["beta_background_prior"] = reshape(model.beta_prior_mean, ny, nx)
    sxs = Float64[]; sys = Float64[]
    dx = model.parameters.x_length / nx; dy = model.parameters.y_length / ny
    for idx in model.sensor_indices
        j = ((idx - 1) % ny) + 1; i = ((idx - 1) ÷ ny) + 1
        push!(sxs, (i-1)*dx); push!(sys, (j-1)*dy)
    end
    f["sensor_indices"] = collect(model.sensor_indices)
    f["sensor_x"] = sxs; f["sensor_y"] = sys
    g = create_group(f, "parameters")
    attributes(g)["nprt"]=NPRT; attributes(g)["T"]=T; attributes(g)["nx"]=nx; attributes(g)["ny"]=ny
    attributes(g)["time_step"]=model.parameters.time_step
    attributes(g)["obs_space"]="log_beta"; attributes(g)["obs_noise_std_log"]=0.40
    attributes(g)["K_temper"]=K_TEMPER; attributes(g)["seed_pf"]=SEED_PF; attributes(g)["seed_obs"]=SEED_OBS
end
println("Saved particle trajectory → $out_path  ($(round(filesize(out_path)/1e9, digits=2)) GB)")

# ── β-vs-x cross-section diagram ─────────────────────────────────────────
# Choose the grid row carrying the most sensors so the section shows where
# observations constrain β.
sensor_rows = [((idx - 1) % ny) + 1 for idx in model.sensor_indices]
row_counts  = Dict{Int,Int}()
for r in sensor_rows; row_counts[r] = get(row_counts, r, 0) + 1; end
row_mid = argmax(row_counts)                       # row with most sensors
row_y_km = (row_mid - 1) * (model.parameters.y_length / ny) / 1000
xs_km = collect(0:nx-1) .* (model.parameters.x_length / nx) / 1000
sensors_on_row_x = [ (((idx-1) ÷ ny))*(model.parameters.x_length/nx)/1000
                     for idx in model.sensor_indices if (((idx-1) % ny)+1) == row_mid ]
println("Cross-section row = $row_mid (y ≈ $(round(row_y_km,digits=1)) km), $(length(sensors_on_row_x)) sensors on it")

truth_grid = reshape(truth_states, ny, nx, T+1)
mean_grid  = reshape(ensemble_mean, ny, nx, T+1)
part_grid  = reshape(all_particles, ny, nx, NPRT, T+1)
clim = (quantile(vec(truth_grid), 0.02), quantile(vec(truth_grid), 0.98))

# Particle spread bands along the row, per time.
function section_bands(t)
    P = part_grid[row_mid, :, :, t]                # (nx, NPRT)
    lo90 = [quantile(P[i, :], 0.05) for i in 1:nx]
    hi90 = [quantile(P[i, :], 0.95) for i in 1:nx]
    lo50 = [quantile(P[i, :], 0.25) for i in 1:nx]
    hi50 = [quantile(P[i, :], 0.75) for i in 1:nx]
    return lo90, hi90, lo50, hi50
end

function draw_section(t)
    lo90, hi90, lo50, hi50 = section_bands(t)
    p = plot(xs_km, hi90; fillrange=lo90, fillalpha=0.18, color=:steelblue,
             lw=0, label="particles 5–95%",
             xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("run17 tempered — β cross-section  y=%.0f km  t=%.0f h",
                            row_y_km, (t-1)*model.parameters.time_step/3600),
             ylim=(minimum(clim), maximum(clim)), legend=:topright)
    plot!(p, xs_km, hi50; fillrange=lo50, fillalpha=0.30, color=:steelblue,
          lw=0, label="particles 25–75%")
    plot!(p, xs_km, mean_grid[row_mid, :, t]; lw=2, ls=:dash, color=:navy,
          label="ensemble mean")
    plot!(p, xs_km, truth_grid[row_mid, :, t]; lw=3, color=:black, label="truth")
    for (i_s, sx) in enumerate(sensors_on_row_x)
        vline!(p, [sx]; color=:red, ls=:dot, lw=1.2, label=(i_s==1 ? "sensor x" : false))
    end
    return p
end

savefig(draw_section(T+1), joinpath(OUT, "crosssection_final_tempered.png"))
println("Saved → $(joinpath(OUT, "crosssection_final_tempered.png"))")

anim = @animate for t in 1:T+1
    draw_section(t)
end
gif(anim, joinpath(OUT, "crosssection_anim_tempered.gif"), fps=6)
println("Saved → $(joinpath(OUT, "crosssection_anim_tempered.gif"))")

# ESS trace alongside, for context.
p_ess = plot(1:T, ess_series; xlabel="timestep", ylabel="min-stage ESS",
             title="run17 tempered — ESS (16 obs, K=$K_TEMPER, σ_log=0.40)",
             lw=2, color=:seagreen, ylim=(0, NPRT), label="ESS")
hline!(p_ess, [ESS_THRESHOLD]; ls=:dash, color=:gray, label="N/2")
scatter!(p_ess, [ess_argmin], [ess_min]; color=:red, ms=6,
         label=@sprintf("min %.0f @t=%d", ess_min, ess_argmin))
savefig(p_ess, joinpath(OUT, "ess_tracking.png"))
println("Saved → $(joinpath(OUT, "ess_tracking.png"))")
println("Done.")
