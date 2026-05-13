# Run from the repo root:
#   julia --project=test bootstrap-pf-experiments/code/benchmark_run.jl
# (the script `cd`s to the repo root itself, so launching from any directory works)

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)
@info "Working directory set to $(pwd())"

using ParticleDA
include(joinpath(REPO_ROOT, "test", "models", "llw2d.jl"))
using .LLW2d
using HDF5

const YAML_PATH = joinpath("bootstrap-pf-experiments", "code", "benchmark.yaml")
const OBS_PATH  = joinpath("bootstrap-pf-experiments", "results", "benchmark_obs.h5")
const DA_PATH   = joinpath("bootstrap-pf-experiments", "results", "particle_da.h5")

mkpath(dirname(OBS_PATH))

# Remove stale output
isfile(OBS_PATH) && rm(OBS_PATH)
isfile(DA_PATH)  && rm(DA_PATH)

println("=== Step 1: Simulating observations (seed=123, T=260) ===")
simulate_observations_from_model(LLW2d.init, YAML_PATH, OBS_PATH)
println("Observations written to: $OBS_PATH")

println("\n=== Step 2: Bootstrap particle filter (N=200, T=260) ===")
run_particle_filter(
    LLW2d.init,
    YAML_PATH,
    OBS_PATH,
    ParticleDA.BootstrapFilter,
    ParticleDA.MeanAndVarSummaryStat,
)
println("Filter output written to: $DA_PATH")

# ── Verification ──────────────────────────────────────────────────────────────
println("\n=== Verification ===")
@assert isfile(DA_PATH) "particle_da.h5 missing"
println("  [OK] particle_da.h5 exists")

h5open(DA_PATH, "r") do fh
    nw = length(keys(fh["weights"]))
    println("  [INFO] weights group has $nw timestep keys (t0000..t$(lpad(nw-1,4,'0')))")
    @assert nw >= 50 "Expected ≥50 weight timesteps, got $nw"
    println("  [OK] weights contains ≥50 timesteps")

    final_key = "t" * lpad(string(nw - 1), 4, '0')   # last available timestep
    h = read(fh["state_avg"][final_key]["height"])
    @assert size(h) == (51, 51) "Expected (51,51), got $(size(h))"
    println("  [OK] state_avg/$final_key/height has shape $(size(h))")
end

println("\nRun complete. Now execute benchmark_plot.jl to generate figures.")
