# Observation cadence experiment.
#
# Run-04 (every-1 cadence): time_step=400, n_integration_step=1, 200 filter steps.
#   → internal dt = 400 s, total model time = 80 000 s, observation every 400 s.
#
# Run-05 (every-2 cadence): time_step=800, n_integration_step=2, 100 filter steps.
#   → internal dt = 400 s (SAME physics per sub-step), total model time = 80 000 s,
#     but observations only every 800 s → HALF as many obs over the same physical span.
#
# Why 400 s? Our toy's CFL stability ceiling is 0.2 * dx / max_speed
# ≈ 0.2 * 4000 / 1.75 ≈ 457 s, so dt = 400 s sits just below with safety margin.
# At dt = 1 s the advection per step (~1.5 m) is much smaller than dx (4 km),
# so the truth would barely move and the filter has nothing to track — that
# was our first attempt and it produced near-flat RMSE.
#
# Run: julia --project=test glacier-code/particleda/run_obs_cadence.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier

const RESULTS = joinpath("glacier-code", "particleda", "results")
const CADENCE_DIR = joinpath(RESULTS, "obs_cadence")
mkpath(CADENCE_DIR)

const COMMON_MODEL = """
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
    n_integration_step: PLACEHOLDER_NINT
    time_step: PLACEHOLDER_TS
"""

function run_case(label::String; time_step::Float64, n_int::Int, n_time_step::Int, seed::Int=42)
    out_dir = joinpath(CADENCE_DIR, label)
    mkpath(out_dir)
    obs_path = joinpath(out_dir, "glacier_obs.h5")
    da_path  = joinpath(out_dir, "particle_da.h5")
    yaml_path = joinpath(out_dir, "glacier.yaml")
    isfile(obs_path) && rm(obs_path)
    isfile(da_path)  && rm(da_path)

    yaml_str = """
filter:
  nprt: 1000
  verbose: true
  output_filename: "$da_path"
  seed: $seed

model:
$(replace(COMMON_MODEL, "PLACEHOLDER_NINT" => string(n_int), "PLACEHOLDER_TS" => string(time_step)))

simulate_observations:
  seed: $seed
  n_time_step: $n_time_step
"""
    write(yaml_path, yaml_str)

    println("\n=== $label (time_step=$time_step, n_int=$n_int, T=$n_time_step) ===")
    simulate_observations_from_model(Glacier.init, yaml_path, obs_path)
    run_particle_filter(
        Glacier.init, yaml_path, obs_path,
        ParticleDA.BootstrapFilter,
        ParticleDA.MeanAndVarSummaryStat,
    )
end

run_case("run04_every1c"; time_step=400.0, n_int=1, n_time_step=200)
run_case("run05_every2c"; time_step=800.0, n_int=2, n_time_step=100)

println("\nDone. Per-case outputs in: ", abspath(CADENCE_DIR))
