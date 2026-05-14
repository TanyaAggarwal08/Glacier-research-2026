# Run from anywhere:
#   julia --project=test glacier-code/particleda/run_glacier_pda.jl
# The script cd's to the repo root, so relative paths in glacier.yaml resolve.

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)
@info "Working directory set to $(pwd())"

using ParticleDA
include(joinpath(@__DIR__, "glacier_model.jl"))
using .Glacier
using HDF5

const YAML_PATH = joinpath("glacier-code", "particleda", "glacier.yaml")
const OBS_PATH  = joinpath("glacier-code", "particleda", "results", "glacier_obs.h5")
const DA_PATH   = joinpath("glacier-code", "particleda", "results", "particle_da.h5")

mkpath(dirname(OBS_PATH))

isfile(OBS_PATH) && rm(OBS_PATH)
isfile(DA_PATH)  && rm(DA_PATH)

println("=== Step 1: Simulating observations ===")
simulate_observations_from_model(Glacier.init, YAML_PATH, OBS_PATH)
println("Observations written to: $OBS_PATH")

println("\n=== Step 2: Bootstrap particle filter ===")
run_particle_filter(
    Glacier.init,
    YAML_PATH,
    OBS_PATH,
    ParticleDA.BootstrapFilter,
    ParticleDA.MeanAndVarSummaryStat,
)
println("Filter output written to: $DA_PATH")

# ── Verification ──────────────────────────────────────────────────────────────
println("\n=== Verification ===")
@assert isfile(OBS_PATH) "glacier_obs.h5 missing"
@assert isfile(DA_PATH)  "particle_da.h5 missing"
println("  [OK] both HDF5 files exist")

h5open(DA_PATH, "r") do fh
    nw = length(keys(fh["weights"]))
    println("  [INFO] weights group has $nw timestep keys")
    @assert nw >= 10 "Expected ≥10 weight timesteps, got $nw"

    final_key = sort(collect(keys(fh["state_avg"])))[end]
    log_beta = read(fh["state_avg"][final_key]["log_beta"])
    println("  [OK] state_avg/$final_key/log_beta shape = $(size(log_beta))")
    @assert size(log_beta) == (40, 40) "Expected (40,40), got $(size(log_beta))"
end

println("\nRun complete.")
