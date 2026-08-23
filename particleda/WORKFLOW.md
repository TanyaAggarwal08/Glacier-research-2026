# How to run the glacier ParticleDA workflow

> One-stop cheat sheet. **What lives in each file. What to run after each kind of change. Where outputs land.**

---

## 1. File map — what lives where

```
glacier-code/particleda/
│
├── glacier_model.jl              ← THE MODEL (Julia module)
│                                   - Defines GlacierModelParameters struct
│                                   - Implements the ParticleDA interface:
│                                       sample_initial_state!
│                                       update_state_deterministic!  ← the upwind advection
│                                       update_state_stochastic!     ← per-step process noise
│                                       get_observation_mean_given_state!
│                                       sample_observation_given_state!
│                                       get_log_density_observation_given_state
│                                       write_model_metadata, write_state
│                                   - Reads parameters from YAML.
│                                   - You edit this when the *physics* changes.
│
├── glacier.yaml                  ← CANONICAL CONFIG (grid-aligned 16 sensors)
├── glacier_random.yaml           ← VARIANT CONFIG (scattered sensors from txt file)
│                                   - Filter knobs: nprt, seed, output_filename
│                                   - Model knobs: nx, ny, σ_init, σ_obs,
│                                     σ_proc, advection_epsilon, time_step,
│                                     n_integration_step, station_filename
│                                   - Sim knobs: n_time_step, seed
│                                   - Edit these to *re-tune* without touching code.
│
├── stations_random_16.txt        ← SENSOR LOCATIONS (x_m, y_m), 16 random points
│                                   - Comma-separated CSV; # = comment
│                                   - Edit / replace to change sensor layout.
│
├── run_glacier_pda.jl            ← THE RUNNER
│                                   - Calls simulate_observations_from_model
│                                     (generates synthetic truth + obs) and
│                                     run_particle_filter (runs the PF).
│                                   - Reads YAML; writes glacier_obs.h5 and
│                                     particle_da.h5 next to YAML.output_filename.
│                                   - Takes optional YAML path as ARG.
│
├── run_seed_sweep.jl             ← Multi-seed driver (5 seeds, same config)
├── run_obs_cadence.jl            ← Compare every-1c vs every-2c cadence
│
├── plot_glacier_pda.jl           ← MAIN PLOTTER
│                                   - Reads particle_da.h5 + glacier_obs.h5
│                                     from a directory; writes GIFs + PNGs there.
│                                   - Takes optional results-dir as ARG.
│
├── plot_seed_sweep.jl            ← Aggregates 5 seed runs into one plot set
├── plot_obs_cadence.jl           ← Compares every-1c vs every-2c
│
├── WORKFLOW.md                   ← THIS FILE
└── results/                      ← All outputs go here
    ├── particle_da.h5            ← Latest filter output (most recent run)
    ├── glacier_obs.h5            ← Latest truth + observations
    ├── *.png, *.gif              ← Latest plots
    ├── run01_tight_prior/        ← Archived earlier runs (snapshots)
    ├── run02_wide_prior_100obs/
    ├── run03_wide_16obs/
    ├── run04_grid_aligned_16obs/ ← Canonical reference (preserved)
    ├── run05_random_scattered_16obs/  ← Scattered-sensor demo (preserved)
    └── seed_sweep_v2/, obs_cadence/   ← multi-seed / cadence sweeps
```

---

## 2. What to run, depending on what you changed

### A. You edited `glacier.yaml` (parameters only — no code change)

```bash
# 1. Run the filter (uses default YAML next to the script):
julia --project=test glacier-code/particleda/run_glacier_pda.jl

# 2. Make the plots (defaults to results/):
julia --project=test glacier-code/particleda/plot_glacier_pda.jl
```

Outputs land in `glacier-code/particleda/results/`.

### B. You edited a non-default YAML (e.g. `glacier_random.yaml` for scattered sensors)

```bash
# 1. Pass the YAML as the first argument:
julia --project=test glacier-code/particleda/run_glacier_pda.jl glacier-code/particleda/glacier_random.yaml

# 2. Pass the matching results dir to the plotter (read it from the YAML's output_filename):
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run05_random_scattered_16obs
```

### C. You edited `glacier_model.jl` (code change — physics, observation operator, etc.)

```bash
# Same as A or B — just run the runner. Julia recompiles the module automatically.
julia --project=test glacier-code/particleda/run_glacier_pda.jl

# Then plots:
julia --project=test glacier-code/particleda/plot_glacier_pda.jl
```

You do **not** run `julia glacier_model.jl` directly. It's a module, not an entry point. The runner pulls it in via `include`.

### D. You edited a stations file (e.g. `stations_random_16.txt`)

```bash
# The YAML points at the stations file. Just rerun via the YAML it's referenced from:
julia --project=test glacier-code/particleda/run_glacier_pda.jl glacier-code/particleda/glacier_random.yaml
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run05_random_scattered_16obs
```

### E. You only edited `plot_glacier_pda.jl` (plotting code — same data)

```bash
# No need to rerun the filter! Just rerun the plotter against existing HDF5:
julia --project=test glacier-code/particleda/plot_glacier_pda.jl

# Or for a specific archived run:
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run04_grid_aligned_16obs
```

This is the fastest iteration loop — plotting takes ~30 s, the filter takes ~15 s for the canonical config but more for longer cadences.

### F. You want to compare across seeds (robustness check)

```bash
julia --project=test glacier-code/particleda/run_seed_sweep.jl       # ~75 s (5 seeds)
julia --project=test glacier-code/particleda/plot_seed_sweep.jl      # aggregates into one plot set
```

Outputs land in `results/seed_sweep_v2/`.

### G. You want to compare two observation cadences

```bash
julia --project=test glacier-code/particleda/run_obs_cadence.jl
julia --project=test glacier-code/particleda/plot_obs_cadence.jl
```

Outputs in `results/obs_cadence/`.

---

## 3. The two-step pattern, always

The workflow is always **runner first, plotter second**:

| Step | What it does | Reads | Writes |
|---|---|---|---|
| **Runner** | Simulates truth + observations, runs the particle filter | YAML | `particle_da.h5`, `glacier_obs.h5` (HDF5 binary files) |
| **Plotter** | Reads HDF5, makes GIFs and PNGs | `particle_da.h5`, `glacier_obs.h5` | `*.png`, `*.gif` |

If you change something the **filter** depends on (model code, YAML params, stations file) → rerun the runner, *then* the plotter.

If you only change something the **plotter** depends on (plot styles, axis limits, new diagnostic) → just rerun the plotter.

---

## 4. Naming convention for results

If you want to keep a run for later comparison instead of overwriting it, snapshot it before the next run:

```bash
# Move the freshly produced files into a named folder:
mkdir -p glacier-code/particleda/results/my_experiment_name
mv glacier-code/particleda/results/*.{gif,png,h5} glacier-code/particleda/results/my_experiment_name/
```

This is what the existing `run01_..` / `run02_..` / etc. folders are — preserved snapshots of past experiments.

The default `run_glacier_pda.jl` writes to `results/` (overwriting). To write to a specific folder directly without manual moving, copy the YAML and change `filter.output_filename`:

```yaml
filter:
  output_filename: "glacier-code/particleda/results/my_experiment_name/particle_da.h5"
```

Then run `julia --project=test glacier-code/particleda/run_glacier_pda.jl path/to/your.yaml` and the plotter against that folder.

---

## 5. Common gotchas

| Symptom | Fix |
|---|---|
| `LoadError: ArgumentError: Package YAML not found in current path` | You forgot `--project=test`. Always include it. |
| Plots appear at old location instead of where I expected | The runner reads `output_filename` from the YAML. Update both `output_filename` AND the second arg to the plotter. |
| IDE doesn't show new files in `results/` | Refresh the IDE folder view. The files are on disk; the IDE just caches. |
| `julia: Permission denied` when running | Use `julia --project=test path/to/script.jl`, not `julia` directly. |
| Plot script can't find HDF5 | The runner failed earlier. Look for `Run complete.` in the runner output. |
| `julia` is slow on first run | First call has Julia startup + precompilation (~5 s). Subsequent calls in same shell are faster. |

---

## 6. Quick reference — the three commands you'll use 90% of the time

```bash
# 1. Default canonical run + plots
julia --project=test glacier-code/particleda/run_glacier_pda.jl
julia --project=test glacier-code/particleda/plot_glacier_pda.jl

# 2. Random scattered sensors variant
julia --project=test glacier-code/particleda/run_glacier_pda.jl  glacier-code/particleda/glacier_random.yaml
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run05_random_scattered_16obs

# 3. Just re-plot an old archived run (no filter rerun)
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run04_grid_aligned_16obs
```

That's it. Bookmark this file.
