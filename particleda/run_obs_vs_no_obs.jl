# Observation ablation experiment.
#
# Runs the same particle filter under TWO configurations:
#   A) Canonical    (glacier.yaml)        → filter assimilates observations
#   B) No-obs       (glacier_no_obs.yaml) → filter ignores observations
#
# Same seeds in both → identical truth and observations. The only difference
# is whether the filter uses the data. Outputs land in
#   results/run06_obs_baseline/   (a fresh copy of the canonical)
#   results/run06_no_observations/
#
# Run: julia --project=test glacier-code/particleda/run_obs_vs_no_obs.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using ParticleDA
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier
using YAML

function run_case(yaml_path::String, label::String)
    @info "\n=== Running $label ($yaml_path) ==="
    yaml_dict = YAML.load_file(yaml_path)
    da_path = yaml_dict["filter"]["output_filename"]
    obs_path = joinpath(dirname(da_path), "glacier_obs.h5")
    mkpath(dirname(da_path))
    isfile(obs_path) && rm(obs_path)
    isfile(da_path)  && rm(da_path)

    simulate_observations_from_model(Glacier.init, yaml_path, obs_path)
    run_particle_filter(
        Glacier.init, yaml_path, obs_path,
        ParticleDA.BootstrapFilter,
        ParticleDA.MeanAndVarSummaryStat,
    )
    return (obs=obs_path, da=da_path)
end

# A) Canonical (with observations) — we re-run it into a dedicated folder
#    so it sits alongside the no-obs run for clean comparison.
baseline_yaml = joinpath(@__DIR__, "glacier_obs_baseline.yaml")
# Generate a one-off YAML that's a clone of canonical but writes into
# run06_obs_baseline/ instead of overwriting the canonical results.
const BASELINE_YAML_CONTENT = """
filter:
  nprt: 1000
  verbose: true
  output_filename: "glacier-code/particleda/results/run06_obs_baseline/particle_da.h5"
  seed: 42

model:
  glacier:
    nx: 40
    ny: 40
    x_length: 160000.0
    y_length: 160000.0
    sensor_stride: 100
    init_std_beta: 300.0
    process_std_beta: 7.0
    obs_noise_std: 0.10
    advection_epsilon: 0.0005
    n_integration_step: 10
    time_step: 3600.0
    min_beta: 10.0
    noise_length_scale: 30000.0
    disable_observations: false

simulate_observations:
  seed: 123
  n_time_step: 200
"""
write(baseline_yaml, BASELINE_YAML_CONTENT)

run_case(baseline_yaml, "Baseline (with observations)")
run_case(joinpath(@__DIR__, "glacier_no_obs.yaml"), "No observations")

println("\nDone. Outputs:")
println("  Baseline (with obs):     glacier-code/particleda/results/run06_obs_baseline/")
println("  No observations:         glacier-code/particleda/results/run06_no_observations/")
