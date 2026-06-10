# Multi-seed robustness check: rerun run-03 config at 5 different seeds.
# Each seed varies *both* simulate_observations.seed (different truth + obs
# realisations) and filter.seed (different particle randomness) so each run
# is a fully independent draw.
#
# Run: julia --project=test glacier-code/particleda/run_seed_sweep.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier

const SEEDS = [42, 7, 13, 99, 2024]
const SWEEP_DIR = joinpath("glacier-code", "particleda", "results", "seed_sweep_v2")
mkpath(SWEEP_DIR)

const YAML_TEMPLATE = """
filter:
  nprt: 1000
  verbose: true
  output_filename: "PLACEHOLDER_DA"
  seed: PLACEHOLDER_SEED

model:
  glacier:
    nx: 40
    ny: 40
    x_length: 160000.0
    y_length: 160000.0
    sensor_stride: 100
    init_std_theta: 0.30
    process_std_theta: 0.007
    obs_noise_std: 0.10
    advection_epsilon: 0.0005
    n_integration_step: 1
    time_step: 400.0

simulate_observations:
  seed: PLACEHOLDER_SEED
  n_time_step: 200
"""

for seed in SEEDS
    seed_dir = joinpath(SWEEP_DIR, "seed$(lpad(seed, 4, '0'))")
    mkpath(seed_dir)
    obs_path  = joinpath(seed_dir, "glacier_obs.h5")
    da_path   = joinpath(seed_dir, "particle_da.h5")
    yaml_path = joinpath(seed_dir, "glacier.yaml")
    isfile(obs_path) && rm(obs_path)
    isfile(da_path)  && rm(da_path)

    yaml_str = replace(YAML_TEMPLATE,
                       "PLACEHOLDER_DA"   => da_path,
                       "PLACEHOLDER_SEED" => string(seed))
    write(yaml_path, yaml_str)

    println("\n=== seed = $seed → $seed_dir ===")
    simulate_observations_from_model(Glacier.init, yaml_path, obs_path)
    run_particle_filter(
        Glacier.init, yaml_path, obs_path,
        ParticleDA.BootstrapFilter,
        ParticleDA.MeanAndVarSummaryStat,
    )
end

println("\nDone. Per-seed outputs in: ", abspath(SWEEP_DIR))
