# temp_run — temperature-driven basal drag, with smooth process noise

Self-contained. One file, one command, all results.

```bash
julia glacier-code/temp_run/run_temperature_noise.jl
```

Everything lands in `results/noise_temperature_forcing/`. Takes about a minute.

## What it does

A **forward** experiment — nothing is estimated, there is no particle filter.
Each time step runs four stages:

| stage | what happens |
|---|---|
| 1. temperature | seasonal + diurnal cycles, plus an elevation lapse along x |
| 2. smooth noise | `T* = T + σ_T·N`, with `N` smooth in space *and* time |
| 3. basal drag | `β = 2000 − 40·(T* + 10)`, floored at 10 — warm ⇒ slippery |
| 4. ice velocity | full WAVI momentum solve on that β |

The clean (noise-free) case is computed alongside the noisy one, step for step,
so every figure compares like with like.

## The noise

Smooth in **both** senses, which is the point of the experiment:

- **Space** — innovations are `L·z`, with `L` the Cholesky factor of a
  squared-exponential kernel `exp(−r²/2ℓ²)`, ℓ = 15 km. This is the *same*
  generator the particle filter uses for its own process noise, so the
  perturbations look like the ones the filter is built to handle rather than
  white pixel noise. The blobs are 30–40 km across.
- **Time** — an AR(1) chain, `N_t = ρ·N_{t−1} + √(1−ρ²)·L·z`, with
  ρ = exp(−Δt/τ). The field drifts instead of flickering every step.

Because the kernel has unit diagonal, `σ_T` is directly the perturbation
standard deviation in °C. The script verifies this and prints the realised
values against the targets.

## Results at the defaults (3 days, hourly, σ_T = 2 °C, τ = 6 h)

| | clean | noisy |
|---|---|---|
| temperature | −5.05 … 14.85 °C | −5.91 … 15.00 |
| β | 1006 … 1802 Pa·s/m | 1000 … 1836 |
| ice speed | 17.54 … 31.25 m/yr | 17.26 … 31.60 |
| corr(β, speed) over time | **−0.9895** | **−0.9891** |
| corr pooled over cells & times | −0.8578 | −0.8605 |
| speed departure from clean | — | **0.33 m/yr rms (1.4%)** |

Noise check: realised sd 2.05 °C (target 2.00), lag-1 autocorrelation 0.851
(target 0.846).

**Two things worth reading off these numbers.**

**±2 °C of smooth noise moves the ice speed by only 1.4%.** The β–speed
correlation is essentially untouched. Two stages of averaging damp it: velocity
is set by membrane stresses, so each cell responds to a whole neighbourhood of β
and 30–40 km blobs partly cancel; then the interior mean averages ~1,100 cells.
A ±2 °C perturbation is also small beside the ~20 °C diurnal swing driving the
signal. Encouraging for robustness, awkward for observability — if realistic
temperature uncertainty moves velocity less than a typical observation error,
velocity observations cannot constrain it. Try `σ_T = 8` to find where the
signal clears the noise floor.

**The pooled correlation (−0.86) is weaker than the time-series one (−0.99) for
*both* runs.** That gap is not the noise. It is the spatial lapse term putting a
fixed x-tilt into β that velocity does not follow linearly. Related: the field is
nearly uniform in y, so the 16 sensors carry little more information than one
would — visible in the timeseries plot, where the sensor mean sits on top of the
interior mean.

## Outputs

| file | contents |
|---|---|
| `tempflow_noisy_evolution.gif` | clean T \| noisy T \| β \| speed, one frame per step |
| `tempflow_noisy_timeseries.png` | clean vs noisy T, β and speed against time |
| `tempflow_noisy_response.png` | β–speed scatter, clean vs noisy |
| `tempflow_noise_field.png` | the noise itself, plus its measured autocorrelation |
| `summary.txt` | every number the run printed |

## Arguments

```
julia run_temperature_noise.jl [n_days] [start_day] [per_day] [σ_T] [τ_h] [seed]
```

| arg | default | meaning |
|---|---|---|
| `n_days` | 3 | length of the run |
| `start_day` | 200 | day-of-year to start from (200 ≈ annual peak) |
| `per_day` | 24 | steps per day; 24 = hourly |
| `σ_T` | 2.0 | temperature noise sd, °C; `0` disables noise entirely |
| `τ_h` | 6.0 | noise decorrelation time, hours |
| `seed` | 42 | RNG seed |

```bash
julia run_temperature_noise.jl 3 200 24 8.0 6 42   # stronger noise
julia run_temperature_noise.jl 365 0 1 2.0 24 42   # a full year, daily steps
julia run_temperature_noise.jl 3 200 24 0.0 6 42   # noise off (clean only)
```

## Notes

**Boundary conditions.** WAVI's orientation names are array indices, not compass
directions: `"north"/"south"` are the **x** edges, `"east"/"west"` the **y**
edges. This script uses a wall at x = 0 (`u_iszero = ["north"]`, the ice divide)
and free-slip sidewalls at y = 0 and y = L (`v_iszero` includes `"east","west"`).
Without the sidewalls those edges act as calving fronts and run ~32× faster than
the interior, which buries the β signal under the colour scale.

**Speed colour scales are interior-only** in the animation. The x = 0 divide
holds ice back over roughly three columns, so a full-field scale skews the low
end. Statistics use the same `4:37` interior mask.

**The β(T) map is a caricature**, not a melt model: a straight inverse line,
`β = β_c − k(T − T_c)`. It is meant to produce a plausible β range from a
temperature signal, nothing more.

**Runs in the default Julia environment** (needs WAVI and Plots). No threads
needed.

## Relationship to the exploratory code

The originals live in `glacier-code/particleda/` (`temperature_graph.jl`,
`temperature_driven_flow.jl`, `temperature_driven_flow_noisy.jl`) and are kept
as the testing record. This is the clean re-implementation and reproduces their
numbers exactly.
