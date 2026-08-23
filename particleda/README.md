# Glacier Bootstrap-PF on ParticleDA

Particle-filter data assimilation for a glacier basal-friction (β) estimation problem. State is β directly (no log transform). The forward map is a surrogate `ux = 1000 / β`. Smooth (correlated) noise on the prior and process, sparse scattered sensors, hourly cadence by default.

Detailed background, audits, and design notes live in [`glacier-notes/`](../../glacier-notes/).

---

## File map

```
glacier-code/particleda/
├── glacier_model.jl              ← THE MODEL (Julia module)
├── glacier.yaml                  ← canonical config
├── glacier_no_obs.yaml           ← ablation: filter ignores observations
├── glacier_random.yaml           ← scattered sensors variant
├── stations_*.txt                ← sensor coordinates for the *_random / *_crosssection variants
│
├── run_glacier_pda.jl            ← single-run filter driver (uses YAML config)
├── run_obs_vs_no_obs.jl          ← obs/no-obs ablation experiment
├── run_particle_tracking.jl      ← per-particle tracking driver (NONLINEAR adv)
├── run_linear_advection.jl       ← per-particle tracking driver (LINEAR adv)
├── run_pseudorandom_wave.jl      ← pseudo-random wave prior/background run
│
├── plot_glacier_pda.jl           ← plotter for the canonical run
├── plot_obs_vs_no_obs.jl         ← plotter for the ablation
├── plot_particle_tracking.jl     ← plotter for run07 (nonlinear)
├── plot_linear_advection.jl      ← plotter for run08 (linear)
├── plot_pseudorandom_wave.jl     ← plotter for run11 (pseudo-random prior)
│
└── results/
    ├── run06_obs_baseline/
    ├── run06_no_observations/
    ├── run06_ablation_compare/
    ├── run07_particle_tracking/      ← outputs from the nonlinear PF
    └── run08_linear_advection_run/   ← outputs from the linear PF
```

Heavy HDF5 files (≈1.2 GB each) live on the external SSD at
`/Volumes/ZX20/USRA 2026/run07_particle_tracking/` and `run08_linear_advection_run/`.
The pseudo-random wave run writes to `run11_pseudorandom_wave/`.

---

## 1. Change parameters and re-run on your own

All the knobs that matter are in **two places** for the particle-tracking experiments. Other drivers use YAML files (see §3).

### A. Filter / physics parameters

Edit the `yaml_params` Dict near the top of either driver:

- [run_particle_tracking.jl](run_particle_tracking.jl) (lines ~41–53) — nonlinear advection
- [run_linear_advection.jl](run_linear_advection.jl) (lines ~36–48) — linear advection
- [run_pseudorandom_wave.jl](run_pseudorandom_wave.jl) — pseudo-random truth/background pair

```julia
yaml_params = Dict("glacier" => Dict(
    "nx" => 40, "ny" => 40,             # grid (don't change unless you also rebuild L)
    "x_length"   => 160_000.0,           # domain x in metres
    "y_length"   => 160_000.0,
    "station_filename"   => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta"      => 150.0,       # σ_init — initial particle spread (β units)
    "process_std_beta"   => 7.0,         # σ_proc — per-step process noise (β units)
    "obs_noise_std"      => 0.10,        # σ_obs  — observation noise (ux units)
    "advection_epsilon"  => 5e-4,        # ε — used only by the NONLINEAR branch
    "n_integration_step" => 10,          # advection sub-steps per filter step
    "time_step"          => 3600.0,      # filter cadence (seconds; 3600 = hourly)
    "min_beta"           => 10.0,        # β floor (positivity clamp)
    "noise_length_scale" => 15_000.0,    # ℓ — smooth-noise correlation length (m)
))
```

### B. Experiment-level knobs

Above the Dict, in the same file:

```julia
const NPRT      = 1000    # particle count
const T         = 100     # number of filter steps
const K_TRACK   = 15      # particles to dump full fields for (top + median + bottom)
const SEED_PF   = 42      # filter rng seed
const SEED_OBS  = 123     # truth/observation rng seed
const PROBE_CELLS_IJ = [(20, 20), (10, 10), (10, 30), (5, 15)]   # cells to track all particles at
```

### C. Output folder

To save results to a **new folder name** so you don't overwrite previous results, change two lines in the driver:

```julia
const OUT     = joinpath("glacier-code", "particleda", "results",
                         "my_new_run_name")          # ← change this
const EXT_OUT = "/Volumes/ZX20/USRA 2026/my_new_run_name"  # ← and this
```

Then change the **plotter** to read from the same place. Open [plot_particle_tracking.jl](plot_particle_tracking.jl) or [plot_linear_advection.jl](plot_linear_advection.jl) and edit the top two paths:

```julia
const TRACK = "/Volumes/ZX20/USRA 2026/my_new_run_name/tracking.h5"
const OUT   = joinpath("glacier-code", "particleda", "results",
                       "my_new_run_name")
```

For the pseudo-random wave run, use [run_pseudorandom_wave.jl](run_pseudorandom_wave.jl) and [plot_pseudorandom_wave.jl](plot_pseudorandom_wave.jl). That experiment uses `prior_mode = "pseudo_random_wave"` plus separate truth/background seeds.

### D. Switch between linear and nonlinear advection

Open [glacier_model.jl](glacier_model.jl). Around line 230 there are two advection blocks: one commented out (NONLINEAR), one active (LINEAR). Swap which one is commented to flip the model. Also flip the CFL line just above (line ~222):

- Linear: `max_speed_now = 1.0`
- Nonlinear: `max_speed_now = maximum(1 .+ ε .* β)`

### E. Change the prior β field

In [glacier_model.jl](glacier_model.jl) around lines 121-128 you'll find:

```julia
n_modes = 3                              # spatial frequency knob
ω = n_modes * 2π / p.x_length
…
β_prior = [2000.0 + 2000.0 * sin(ω * xi) * sin(ω * yj) for yj in ys, xi in xs]
```

Three things you can tune:

- **First `2000.0`** — baseline β. Sets the y-centre of the prior.
- **Second `2000.0`** — sinusoid amplitude. Sets how big the bumps are.
- **`n_modes`** — number of full cycles across the domain in each direction. `1` gives the original 4-lobe pattern (one bump up + one bump down across the row); `3` gives a 6×6-lobe pattern with about 6 visible peaks in any cross-section (~27 km half-wavelength). Higher → more ups and downs.

If you want the pseudo-random Evensen-style prior instead, set `prior_mode = "pseudo_random_wave"` and tune:

- `prior_center_beta` and `prior_signal_scale_beta` for the truth field
- `background_std_beta` for the background offset
- `prior_max_wavenumber` for the number of sine-wave modes
- `prior_truth_seed` / `prior_background_seed` for reproducibility

If you raise the baseline or amplitude so β goes above ~4000, also bump the y-axis `ylim` in the plotters (currently `(0, 4500)` — search both `plot_*.jl` files).

---

## 2. Two commands to run a tracking experiment

After every parameter change:

```bash
# 1) Run the filter (generates tracking.h5 on the SSD)
julia --project=test glacier-code/particleda/run_particle_tracking.jl

# 2) Render all GIFs / PNGs from tracking.h5
julia --project=test glacier-code/particleda/plot_particle_tracking.jl
```

For the linear-advection variant, swap both names to `run_linear_advection.jl` / `plot_linear_advection.jl`.

Outputs land in the `OUT` folder you configured. The big HDF5 lands in `EXT_OUT` on the SSD.

---

## 3. Other drivers (YAML-based, simpler workflow)

The original drivers use YAML files instead of inline dicts:

| Driver | Config | Purpose |
|---|---|---|
| `run_glacier_pda.jl [path/to.yaml]` | `glacier.yaml` (default) or arg | Single filter run, no per-particle dump |
| `run_obs_vs_no_obs.jl` | embedded YAMLs | Run obs-on vs obs-off ablation |

To change parameters for these, edit the corresponding YAML directly (`glacier.yaml`, `glacier_no_obs.yaml`, …). Then:

```bash
julia --project=test glacier-code/particleda/run_glacier_pda.jl
julia --project=test glacier-code/particleda/plot_glacier_pda.jl
```

---

## 4. Common edits and what they do

| Want to try | Change | Where | Typical range |
|---|---|---|---|
| Tighter initial prior | `init_std_beta` | driver / YAML | 50 – 400 |
| Less / more process drift | `process_std_beta` | driver / YAML | 1 – 30 |
| Sharper / blunter observations | `obs_noise_std` | driver / YAML | 0.01 – 0.30 |
| Smoother / choppier ensemble | `noise_length_scale` | driver / YAML | 5 000 – 60 000 |
| More / fewer particles | `NPRT` | driver | 200 – 5 000 |
| Longer / shorter run | `T` | driver | 20 – 300 |
| Different cadence | `time_step` (keep `dt = time_step / n_integration_step < ~457 s`) | driver / YAML | 600 – 7 200 |
| Move sensor positions | edit `stations_crosssection.txt` (x, y in metres) | stations file | — |
| Linear ↔ nonlinear advection | swap commented blocks in `glacier_model.jl` | model | — |
| Bigger / smaller prior amplitude | the second `1000.0` in the prior line of `glacier_model.jl` | model | 100 – 1500 |

---

## 5. Outputs you can look at

In `results/<your_run_name>/`:

- `crosssection_anim.gif` — β vs x along y = 76 km, 15 tracked particles
- `crosssection_anim_all_particles.gif` — same but all 1000 particles as a fuzzy band
- `sensor_trail_anim.gif` — β vs time at the on-row sensor cell, all 1000 particles
- `particle_snapshots.png` — 4 timesteps × (truth + 15 particles) heatmaps
- `spaghetti_probe_cells.png` — ensemble spread at 4 probe cells over time
- `weight_evolution.png` — log-weight of 15 tracked particles over time
- `ess_tracking.png` — ESS history

The 1.2 GB `tracking.h5` on the external SSD holds **every** particle's full state at every timestep — load it directly with HDF5 if you want to do custom analysis.

---

## 6. Troubleshooting

- **Plot script can't read `tracking.h5`?** Check the external SSD is mounted at `/Volumes/ZX20/USRA 2026/`. If not, edit the `TRACK` constant in the plotter to point wherever you saved it.
- **`@warn "CFL violation likely"`?** Your `dt = time_step / n_integration_step` is above the safe ceiling. Increase `n_integration_step` (or decrease `time_step`) until `dt < 0.2 · dx / v_max`.
- **`Juliaup configuration is locked` and Julia hangs?** A stale `juliaup self update` process. Find it with `ps aux | grep juliaup` and `kill -9 <pid>`.
- **Want to compare two runs side by side?** Save them under different `OUT` names (and `EXT_OUT` names), and open both GIFs in two Quick Look windows.

---

## 7. Where the science context lives

Numbered notes in [`glacier-notes/`](../../glacier-notes/) carry the rationale behind every parameter choice and every result. Highlights:

- `09_dt_and_cfl_handling.md` — why we substep advection
- `10_one_step_advection.md` — numerical diffusion from first-order upwind
- `17_full_audit.md` — end-to-end audit of the pipeline
- `18_smooth_noise_hourly_cadence.md` — switch to smooth noise + hourly cadence
- `19_noise_pipeline_deepdive.md` — all the noise math
- `20_particle_visualization.md` — what each visualisation actually shows
- `24_pseudorandom_wave_run.md` — full reference for the run11 pseudo-random wave experiment
