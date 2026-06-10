# Visualising individual particles

ParticleDA's built-in summary stat only records ensemble mean and variance to HDF5 — useful for stats, useless for "what is particle 47 actually doing". For this run, we ran a custom bootstrap filter that exposes per-particle state, then produced four targeted visualisations.

## 1. The custom tracking driver

[run_particle_tracking.jl](../glacier-code/particleda/run_particle_tracking.jl) reimplements the same bootstrap PF that `ParticleDA.run_particle_filter` would, but keeps every particle's full β field in memory and selects 8 to dump.

Settings (smaller than canonical so the RAM and runtime are reasonable):

| | value |
|---|---:|
| `nprt` | 200 |
| `T` | 100 hourly steps (≈ 4 days) |
| `K_track` | 8 (top-3 + median-2 + bottom-3 by final-step weight) |
| Probe cells | (20,20), (10,10), (10,30), (5,15) |

The PF loop:

```
for t = 1..T:
    predict each particle:  advect 10 substeps + smooth process noise
    score: log p(y_t | β^p) for each p
    normalise weights w_p
    store w_p, ESS, and the FULL ensemble snapshot (before resampling)
    systematic resample
```

Important: we store the snapshot **before** resampling so each particle index keeps a stable identity for the next step's predict. After resampling, indices get scrambled — useful for the filter, useless for "what did particle 47 do across the run".

### What gets saved

[results/run07_particle_tracking/tracking.h5](../glacier-code/particleda/results/run07_particle_tracking/tracking.h5):

```
/truth/beta                (ny, nx, T+1)              the truth path
/ensemble_mean/beta        (ny, nx, T+1)              mean across all 200 particles
/particles_full/beta       (ny, nx, K=8, T+1)         the 8 tracked particles
/particles_full/indices    (8,)                       which particle ids
/probe_cells/beta          (200, T+1, 4)              ALL particles at 4 cells
/probe_cells/ij_pairs      (2, 4)                     which cells
/weights/raw               (200, T)                   normalised weights
/weights/ess               (T,)                       ESS time series
/sensor_indices            (16,)
```

Total ~10 MB. Disk is cheap.

### Picking the tracked subset

After the run, we sort particles by **final-step weight**:

- **Top 3** ("winners") — particles the filter favours at the end
- **Median 2** — for a baseline of "what an average particle looks like"
- **Bottom 3** ("losers") — particles the filter is rejecting

In this run, the top-3 final weights are 0.0148, 0.0113, 0.0112; bottom-3 are 0.0011, 0.0010, 0.0008. Uniform would be 1/200 = 0.005.

---

## 2. The four visualisations

All under [results/run07_particle_tracking/](../glacier-code/particleda/results/run07_particle_tracking/).

### 2.1 Cross-section animation — `crosssection_anim.gif`

A slice through `y = 20` (the middle row), showing β(x) at every hour as a 1-D curve.

- **Thick black** — truth
- **Blue dashed** — ensemble mean (across all 200 particles)
- **Thin coloured** — the 8 tracked particles

Watch the smooth-noise property in action: each particle is its own coherent wave that drifts left-to-right with the advection. They don't all collapse to one curve, but they do cluster around the truth where the sensors live.

Also useful as a still: `crosssection_final.png` shows the final timestep alone.

### 2.2 Spaghetti per probe cell — `spaghetti_probe_cells.png`

2×2 grid, one panel per probe cell:

- (20, 20) — interior, **not** a sensor
- (10, 10) — interior, **not** a sensor
- (10, 30) — interior, **not** a sensor
- (5, 15) — interior, **not** a sensor

Each panel: all 200 particles plotted as thin grey α=0.15 lines, plus truth (black) and ensemble mean (blue dashed) on top. The grey "fan" is the ensemble spread at that cell over time.

What you should look for:
- **Fan width** = spread of the prior at that cell. Wider = filter is less confident.
- **Mean tracking truth** = is the filter learning that cell?
- **Drift in fan position** = how advection moves the prior around.

(If we'd added a sensor cell to the probes, you'd see the fan narrowing dramatically right after each observation.)

### 2.3 Per-particle snapshot collage — `particle_snapshots.png`

4 rows × (1 + K) = 4 × 9 grid of heatmaps. Rows are snapshot times (t = 1, 34, 68, 100 hours). Columns are: truth, then the 8 tracked particles. Same colour scale (β ∈ [300, 1700]) so they're directly comparable.

What you see:
- At t=1, all particles look roughly similar (just smooth noise around the prior sinusoid).
- By later times, top-weight particles **look more like the truth** at the sensor locations, while bottom-weight particles diverge.
- Each particle remains a coherent smooth field (smooth-noise property — no random pixel-scale flicker).

The per-particle title shows the particle id and its current weight.

### 2.4 Weight evolution — `weight_evolution.png`

The 8 tracked particles' weights on a log y-axis, with `1/N = 0.005` marked as a horizontal dashed line.

- Top-K particles spend most of their time **above** uniform, especially late in the run.
- Bottom-K particles spend most of their time **below** uniform — they were already losing but survived a few resamples before the final score put them at the bottom.
- Resampling spikes show as discontinuities: after every resample, surviving particles' weights re-cluster (because we record weights *before* resampling each step, the resample event is implicit between adjacent t and t+1 points).

There's also a bonus `ess_tracking.png` showing the ESS time series for this nprt=200 run — bounces between ~50 and ~160.

---

## 3. How to re-run

```bash
# 1. Generate fresh tracking.h5 (custom PF run)
julia --project=test glacier-code/particleda/run_particle_tracking.jl

# 2. Render all 4 visualisations from it
julia --project=test glacier-code/particleda/plot_particle_tracking.jl
```

To track different particles or cells, edit the constants near the top of `run_particle_tracking.jl`:

```julia
const NPRT           = 200
const T              = 100
const K_TRACK        = 8
const PROBE_CELLS_IJ = [(20, 20), (10, 10), (10, 30), (5, 15)]
```

To pick by some other criterion than final-step weight (e.g. highest *peak* weight, or particles with smallest tracking error at a specific cell), edit the "Pick 'important' particles" block near the end.

---

## 4. What this lets you see that the standard plots can't

| Question | Standard plots | These plots |
|---|---|---|
| "Is the mean tracking truth?" | rmse_compare.png ✓ | spaghetti (mean line) ✓ |
| "Are the particles diverse?" | (only var, not full distribution) | spaghetti ✓ |
| "Do winning particles look like the truth?" | — | snapshot collage ✓ |
| "How smooth is each particle in space?" | — | cross-section anim ✓ |
| "Which particle is the filter promoting?" | (max weight only) | weight evolution ✓ |
| "Does the prior fan move with advection?" | — | spaghetti ✓ |

The point is to take the filter from a black box that emits one mean curve, to a transparent ensemble where you can watch individual hypotheses evolve and get accepted or rejected.

---

## 5. Glossary additions

| Term | Meaning |
|---|---|
| **Tracked particle** | One of the K particles whose full β field we dump at every timestep for visualisation. Identity is stable only between resampling events — after a resample, the particle at index 47 may have come from a different ancestor than before. |
| **Spaghetti plot** | All particles drawn as thin α-blended lines so the ensemble looks like a fuzzy band. The width of the band shows ensemble spread. |
| **Snapshot collage** | A grid of heatmaps showing several particles' state fields side-by-side at a fixed timestep. Lets you see whether different particles are exploring qualitatively different hypotheses about β. |
| **"Winners" and "losers"** | Particles ranked by their normalised weight just before resampling. Winners get duplicated; losers get dropped. |
