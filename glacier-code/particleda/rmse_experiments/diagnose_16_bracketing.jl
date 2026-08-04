# Diagnostic — does the truth actually sit OUTSIDE the particle cloud?
#
# Exp 16 refuted the bracketing hypothesis: widening init_std_beta made ESS
# worse, not better. That is only surprising if the truth was genuinely
# outside the cloud. This measures it directly instead of asserting it.
#
# For each init_std ∈ {200, 300, 400} and seed s, at t = 1 (the step where the
# collapse bites) compute, at every sensor cell, the standardised distance
#     z_k = (truth_k − mean_particles_k) / std_particles_k
# in log(β) space (the observation space). |z| < 1 means the truth is inside
# the 1σ envelope of the forecast cloud at that sensor. If most sensors have
# |z| ≲ 1, the cloud brackets the truth and the bracketing story is wrong; if
# |z| ≫ 1, it is right and something else explains the Exp 16 result.
#
# Cheap: one init + one propagation step per (init_std, seed), no full run.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/diagnose_16_bracketing.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using ParticleDA, Random, Statistics, LinearAlgebra, PDMats, Printf
include(joinpath(REPO_ROOT, "glacier-code", "particleda", "glacier_model.jl"))
using .Glacier

const NPRT = 1000
const SEEDS = 1:10
const INIT_STDS = [200.0, 300.0, 400.0]
const MIN_BETA = 10.0

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
    "process_std_beta" => 10.0,
    "obs_space" => "log_beta",
    "obs_noise_std" => 0.10,
    "obs_noise_std_log" => 0.40,
    "advection_epsilon" => 0.0,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => MIN_BETA,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "lax_wendroff",
)

# Forecast ensemble + truth at t = 1, at the sensor cells, in log(β).
function bracketing_at_t1(init_std::Float64; seed_pf::Int, seed_obs::Int)
    Glacier._CFL_WARNED[] = false
    params = merge(base_glacier, Dict{String, Any}("init_std_beta" => init_std))
    model = Glacier.init(Dict("glacier" => params))
    n_state = ParticleDA.get_state_dimension(model)
    sensors = model.sensor_indices

    # Truth at t = 1 (deterministic start, one propagation + process step).
    rng_truth = MersenneTwister(seed_obs)
    s_truth = copy(model.truth_prior_mean)
    @inbounds for i in eachindex(s_truth)
        s_truth[i] = max(s_truth[i], MIN_BETA)
    end
    ParticleDA.update_state_deterministic!(s_truth, model, 1)
    ParticleDA.update_state_stochastic!(s_truth, model, rng_truth)
    truth_log = log.(max.(s_truth[sensors], MIN_BETA))

    # Forecast ensemble at t = 1.
    rng_pf = MersenneTwister(seed_pf)
    particles = zeros(n_state, NPRT)
    for p in 1:NPRT
        ParticleDA.sample_initial_state!(view(particles, :, p), model, rng_pf)
    end
    for p in 1:NPRT
        sp = view(particles, :, p)
        ParticleDA.update_state_deterministic!(sp, model, 1)
        ParticleDA.update_state_stochastic!(sp, model, rng_pf)
    end
    sens_log = log.(max.(particles[sensors, :], MIN_BETA))   # (n_obs, NPRT)
    mean_log = vec(mean(sens_log; dims=2))
    std_log  = vec(std(sens_log; dims=2))
    z = (truth_log .- mean_log) ./ std_log                    # per sensor
    return z, std_log
end

println("=== Bracketing diagnostic at t=1 (log β obs space) ===")
println("z_k = (truth − ensemble_mean)/ensemble_std per sensor; |z|<1 ⇒ inside 1σ\n")
@printf("%-10s %12s %12s %12s %12s %12s\n",
        "init_std", "mean |z|", "median |z|", "max |z|",
        "frac |z|<1", "frac |z|<2")
for init_std in INIT_STDS
    all_absz = Float64[]
    frac1 = Float64[]; frac2 = Float64[]
    for s in SEEDS
        z, _ = bracketing_at_t1(init_std; seed_pf=42+s, seed_obs=123+s)
        append!(all_absz, abs.(z))
        push!(frac1, mean(abs.(z) .< 1))
        push!(frac2, mean(abs.(z) .< 2))
    end
    @printf("%-10.0f %12.2f %12.2f %12.2f %12.2f %12.2f\n",
            init_std, mean(all_absz), median(all_absz), maximum(all_absz),
            mean(frac1), mean(frac2))
end
println("\nInterpretation:")
println("  frac|z|<1 high  ⇒ cloud already brackets truth; widening only sharpens weights.")
println("  frac|z|<1 low   ⇒ truth outside cloud; bracketing hypothesis supported.")
