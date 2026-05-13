# Bootstrap-PF Experiments on LLW2d

All ParticleDA Bootstrap-filter experiments from the tsunami benchmark live here. Outputs are written to sibling folders, so the original ParticleDA repo stays clean.

## Folder layout

```
bootstrap-pf-experiments/
├── code/                              ← you are here
│   ├── benchmark.yaml                 (paper-showcase config, N=200, T=260)
│   ├── benchmark_run.jl               (simulate obs + run BootstrapFilter)
│   ├── benchmark_plot.jl              (5-row truth/assimilated grid + ESS + weights)
│   ├── benchmark_sweep.jl             (N × σ_obs × n_stations sweep)
│   └── README.md                      (this file)
├── results/                           ← outputs of benchmark_run + benchmark_plot
│   ├── benchmark_obs.h5
│   ├── particle_da.h5
│   ├── benchmark_fields.png           (5 timesteps × Truth/Assimilated)
│   ├── benchmark_ess.png
│   └── benchmark_weights.png
└── sweep/                             ← outputs of benchmark_sweep
    ├── sweep_*.h5                     (per-config HDF5 — 11 files)
    ├── benchmark_sweep_ess.png
    ├── benchmark_sweep_rmse.png
    ├── benchmark_sweep_summary.png
    ├── benchmark_sweep_results.csv
    └── benchmark_sweep.log
```

## How to run

The scripts auto-locate the repo root via `@__DIR__`, so they work from any working directory.

```bash
# 1. Reproduce the paper's wave-propagation showcase figure (N=200, T=260)
julia --project=test bootstrap-pf-experiments/code/benchmark_run.jl
julia --project=test bootstrap-pf-experiments/code/benchmark_plot.jl

# 2. Run the parameter sweep (11 configurations, ~2–3 hours total)
julia --project=test bootstrap-pf-experiments/code/benchmark_sweep.jl
```

## What each script does

### `benchmark_run.jl` (~5 min single-rank)

- Reads `code/benchmark.yaml`
- Simulates 260 timesteps of truth + observations using `simulate_observations_from_model`
- Runs `BootstrapFilter` with N=200, `MeanAndVarSummaryStat`
- Writes `results/benchmark_obs.h5` (truth states + observations) and `results/particle_da.h5` (posterior mean/variance/weights at every step)
- Verifies the output structure

### `benchmark_plot.jl` (~30 s)

- Reads both h5 files
- Generates three figures into `results/`:
  - `benchmark_fields.png` — 5 rows × 2 columns of (truth, posterior mean) at T = 0, 320, 740, 960, 1280 s, with the 15 station locations overlaid
  - `benchmark_ess.png` — ESS over time with 10 % reference line
  - `benchmark_weights.png` — sorted normalised weights at T = 1280 s

### `benchmark_sweep.jl` (~2–3 hours)

- Runs the same model with 11 different parameter combinations (N ∈ {50, 200, 500, 1000, 5000, 10000}, σ_obs ∈ {0.01, 0.1, 1.0, 5.0}, n_stations ∈ {1, 5, 15}), fast → slow ordering
- Each run writes its own `sweep/sweep_<label>.h5`
- Plots and CSV are refreshed after every config, so partial results are durable if you stop early
- Produces three diagnostic plots and a results table in `sweep/`

## Notes

- Scripts deliberately use **single-rank** ParticleDA (plain `julia ...`, no `mpiexec`). To actually use MPI, launch with `mpiexec -n 4 julia --project=test ...`. See `glacier-notes/02_particleda_mpi_explanation.md`.
- The benchmark.yaml's `station_filename: "inputs/stationsW1.txt"` is a path relative to the repo root, resolved by ParticleDA's YAML loader. The scripts `cd` to the repo root before running, so this works regardless of where you launch from.
- Findings and the bigger-picture analysis live in `~/Desktop/particleDA/ParticleDA.jl/glacier-notes/`.
