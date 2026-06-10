# Random sensor placement — run-05

> **What this run tests.** Same physics, same filter hygiene, same number of sensors (16) as the canonical run-04, but the sensors are now scattered at **arbitrary (x, y) locations** read from a text file instead of being placed on a regular stride through the grid. The point: confirm that the filter still recovers β when observations are placed realistically (the way real InSAR / GPS data is distributed), and make sure the stations-file workflow is in place for when we move to real data.
>
> **Outputs:** [results/run05_random_scattered_16obs/](../glacier-code/particleda/results/run05_random_scattered_16obs/).
> **Stations file:** [glacier-code/particleda/stations_random_16.txt](../glacier-code/particleda/stations_random_16.txt).
> **Canonical reference (grid-aligned 16 obs):** [results/run04_grid_aligned_16obs/](../glacier-code/particleda/results/run04_grid_aligned_16obs/).

---

## 1. What changed in the code

The model gained a new YAML parameter:

```yaml
model:
  glacier:
    station_filename: "glacier-code/particleda/stations_random_16.txt"
```

When this is set to a non-empty string, the model reads the file (comma-separated `x_metres, y_metres`, comments OK with `#`) and snaps each coordinate to the nearest grid cell to derive the flat sensor indices. When it's empty (or omitted), the model falls back to the legacy `sensor_stride` setting.

The relevant code is `_build_sensor_indices` in [glacier_model.jl](../glacier-code/particleda/glacier_model.jl). It mirrors the tsunami pattern at [test/models/llw2d.jl:288-295](../test/models/llw2d.jl#L288-L295).

Two helper changes alongside it:

- [run_glacier_pda.jl](../glacier-code/particleda/run_glacier_pda.jl) now accepts an optional YAML path as a command-line argument; the output paths come from the YAML's `filter.output_filename`. So you can have any number of YAML configs side-by-side, each writing into its own results subfolder.
- [plot_glacier_pda.jl](../glacier-code/particleda/plot_glacier_pda.jl) now accepts an optional results-directory argument and reads + writes everything inside that directory.

Workflow becomes:

```bash
julia --project=test glacier-code/particleda/run_glacier_pda.jl  glacier-code/particleda/glacier_random.yaml
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run05_random_scattered_16obs
```

To try a different sensor layout, edit (or copy) `stations_random_16.txt`, update `station_filename` and `output_filename` in a YAML, and rerun — no code change.

---

## 2. The stations file format

Plain text, comma-separated, comments allowed. Example: [stations_random_16.txt](../glacier-code/particleda/stations_random_16.txt).

```
# Random scattered sensor locations for the glacier surrogate.
# Generated with seed 123, 16 stations, uniform in [16km, 144km] x [16km, 144km].
# Format: x_metres, y_metres
132006.4, 133655.2
72767.2, 70530.5
111446.2, 111563.0
...
```

Rules:
- Coordinates in metres, in the same coordinate system as the model domain (origin at lower-left, x_length × y_length).
- Each line is one sensor. Order doesn't matter.
- Lines starting with `#` are ignored (so you can comment what each sensor is).
- Coordinates outside the domain get clamped to the edge cells (we never go out of bounds, but two sensors that would land in the same cell after snapping are deduplicated to one).

This is exactly the pattern tsunami's [inputs/stationsW1.txt](../inputs/stationsW1.txt) follows, so when we eventually plug in real InSAR / GPS data, we just need the (x, y) of each measurement station.

---

## 3. How this run's sensors are placed

Sixteen `(x, y)` pairs drawn uniformly in `[16 km, 144 km] × [16 km, 144 km]` with `Random.seed!(123)`. The 16 km margin keeps sensors away from the domain edge where the periodic boundary makes interpretation awkward.

You can see the layout in [run05_random_scattered_16obs/final_beta_error.png](../glacier-code/particleda/results/run05_random_scattered_16obs/final_beta_error.png) — the red dots are clustered in the upper half of the domain by chance (the random draw was upper-biased). That's actually informative: the filter has to extrapolate from sparse southern coverage.

Compare to the grid-aligned [run04_grid_aligned_16obs/final_beta_error.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/final_beta_error.png), where sensors are spread more evenly because they came from `sensor_stride = 100` through column-major flat indices.

---

## 4. Headline result — same physics, same filter health

| Metric | run-04 (grid-aligned 16 obs) | run-05 (random scattered 16 obs) |
|---|---:|---:|
| Filter steps | 200 | 200 |
| Internal dt | 400 s | 400 s |
| Total model time | 80 000 s | 80 000 s |
| Mean ESS | ~540 / 1000 (~54 % Np) | ~480 / 1000 (~48 % Np) |
| Final RMSE(β) | ~115 | ~118 |
| Wall-clock | ~15 s | ~15 s |

ESS plot ([run05/ess_evolution.png](../glacier-code/particleda/results/run05_random_scattered_16obs/ess_evolution.png)) sits around the 0.5·Np threshold with the same broad shape as run-04. RMSE curve ([run05/rmse_beta.png](../glacier-code/particleda/results/run05_random_scattered_16obs/rmse_beta.png)) drops from ~405 to ~118 over the run — textbook convergence, indistinguishable in shape from the grid-aligned baseline.

**Takeaway: sensor *layout* matters far less than sensor *count* and `obs_noise_std` at this regime.** The pressure-budget formula already predicted this — pressure depends on `n_obs` and `(σ_signal/σ_obs)²`, not on *where* the sensors sit, provided they sample roughly representative cells.

---

## 5. Where there *is* a difference

It's in the spatial residual map.

Look at the multi-point β plot ([run05/beta_multi_point.png](../glacier-code/particleda/results/run05_random_scattered_16obs/beta_multi_point.png)):

- At points (10, 10), (10, 30), (30, 30), (20, 20) — the PF mean (dashed) tracks the truth (solid) reasonably closely after step ~50 (model time ~20 000 s).
- At point (30, 10) — bottom-left of the domain — the dashed PF mean drifts away from the solid truth for the first half of the run, only catching up after ~50 000 s. That's because none of the 16 random sensors landed nearby; with all 16 clustered upward, the bottom-left corner has very little observational constraint and relies on the dynamics propagating information in from observed regions.

This is real and important: **with sparse, unevenly placed sensors, accuracy is no longer spatially uniform.** The filter still converges globally (RMSE drops), but pointwise error varies with distance to the nearest sensor.

For the grid-aligned run-04, every cell is at most ~10 cells from a sensor; for run-05, the corner cells can be 25+ cells from the nearest random sensor. The filter compensates through the dynamics (advection couples cells over time) and the prior (the spatial smoothness in the initial draw), but uncertainty is genuinely larger in undersampled regions.

If we plotted a **spatial map of posterior Var(β)** (from `state_var` in the HDF5) for the two runs, you'd see this directly — run-05 would have larger variance in the southern half. That's the right way to assess "spatial confidence" once we have real data.

---

## 6. Practical workflow notes

### 6.1 To try a different layout

```bash
# 1. Edit or replace the stations file
vim glacier-code/particleda/stations_random_16.txt
# (or generate a new one via Random/whatever)

# 2. Optionally copy and edit the YAML if you want a new output dir
cp glacier-code/particleda/glacier_random.yaml glacier-code/particleda/glacier_my_layout.yaml
sed -i '' 's|run05_random|my_layout|g' glacier-code/particleda/glacier_my_layout.yaml

# 3. Run
mkdir -p glacier-code/particleda/results/my_layout
julia --project=test glacier-code/particleda/run_glacier_pda.jl  glacier-code/particleda/glacier_my_layout.yaml
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/my_layout
```

### 6.2 Suggestions for layouts worth trying

- **Edge-biased (mimic InSAR coverage along glacier centreline):** 16 sensors along a line at `y = 80 000`, evenly spaced from `x = 10 000` to `x = 150 000`. Models satellite swath coverage.
- **Clustered (mimic GPS station network):** 8 sensors in a 4×2 dense cluster around `(80 000, 80 000)`, 8 sensors scattered elsewhere. Models real-world deployment patterns.
- **Sparse perimeter:** 16 sensors only near the edges of the domain, none in the interior. Worst-case for an inverse problem; useful for finding the limits.

Each of these would take ~15 seconds + plot time.

---

## 7. Sanity-check: ESS / RMSE for run-05 confirm the run is healthy

- **ESS** ([ess_evolution.png](../glacier-code/particleda/results/run05_random_scattered_16obs/ess_evolution.png)): bouncing around 480, with most steps above the 0.5·Np reference. Slightly noisier than run-04 because the spatially-uneven obs gives slightly less informative weights at some steps.
- **RMSE(β)** ([rmse_beta.png](../glacier-code/particleda/results/run05_random_scattered_16obs/rmse_beta.png)): smooth drop from ~405 to ~118 over 80 000 s. Same convergence shape as run-04.
- **Max weight** ([weight_extremes.png](../glacier-code/particleda/results/run05_random_scattered_16obs/weight_extremes.png)): an early spike (wide-prior moment), then collapses below ~0.05 by step 5 and stays there.
- **GIFs** ([true_beta.gif](../glacier-code/particleda/results/run05_random_scattered_16obs/true_beta.gif), [est_beta.gif](../glacier-code/particleda/results/run05_random_scattered_16obs/est_beta.gif), [error_beta.gif](../glacier-code/particleda/results/run05_random_scattered_16obs/error_beta.gif)): scattered red dots show the new layout; the PF mean visibly catches up to the truth over time, with residuals smaller near sensors than far from them.

Nothing unhealthy here. The stations-file workflow is in place, the model handles arbitrary layouts, and the filter is robust to where you place a fixed budget of sensors.

---

## 8. Glossary additions

| Term | Meaning |
|---|---|
| **Stations file** | A plain-text CSV of `(x, y)` sensor coordinates in metres. Each row is one sensor. Comments allowed with `#`. Lets you swap sensor layouts without code changes. |
| **Grid-aligned sensors** | Sensors placed on a regular pattern (e.g. every k-th column-major flat index). Our `sensor_stride` produces this. |
| **Scattered sensors** | Sensors placed at arbitrary (x, y) locations from the stations file. What real-world deployments look like. |
| **Coordinate snapping** | Converting a user-supplied (x, y) in metres into the nearest grid-cell index. Done with `round(x / dx) + 1`. |
| **Deduplicated sensors** | If two user-supplied (x, y) snap to the same grid cell, we keep only one. The file might say "16 stations" but the filter sees fewer if any collide. |
