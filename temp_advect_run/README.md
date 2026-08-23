# temp_advect_run — realistic β **and** a filter that works

Self-contained. One file, one command, all results.

```bash
julia -t 10 glacier-code/temp_advect_run/run_temperature_advect_pf.jl
```

Results → `results/temperature_advect/`. Measured **23.8 min**; budget 20–90 min.

> Use plain `julia`, **not** `julia --project=.` — the repo project environment
> has a broken `NetCDF_jll` artifact and WAVI will not load.

## The result

| | RMSE(β) | reduction |
|---|---|---|
| background | 299.2 | |
| **final (step 24)** | **125.4** | **58.1%** |

speed RMSE 2.75 → **0.96 m/yr** · ESS mean 309.1, min 227.8 · only 8 resamples ·
16,024 WAVI solves

**And it had not finished.** RMSE was still falling at −0.79 per step over the
last six steps, with the best value at the final step. Double-bump plateaued;
this run ran out of clock, not out of information. A longer run would very
likely pass it.

## Where it sits

| run | dynamics | RMSE | reduction |
|---|---|---|---|
| `double_bump_run` | advection, `v = 1 + ε·β` | 298.8 → 112.6 | 62% (20 steps) |
| **`temp_advect_run`** | **temperature: drag + transport** | **299.2 → 125.4** | **58.1%** (24 steps) |
| `temp_field_run` | temperature increment only | 299.2 → 264.0 | 11.8% |
| `temp_pf_run` | temperature, 2 parameters not a field | 439.3 → 67.4 | 84.7% |

This is the run the whole series was aiming at: **a believable β field, recovered
about as well as the invented one.** The realism cost ~4 percentage points, not
50.

## What changed, and why it mattered

`temp_field_run` had temperature enter as an **added number**:

```
β_next = β_now − k·ΔT + noise
```

which cancels out of `particle − truth` exactly, leaving the error a pure random
walk that 16 fixed sensors could never localise. Here temperature also enters as
**transport**:

```
β_next(x) = β_now(x − v·dt) − k·ΔT + noise
```

and the error is *carried*: `error_next(x) ≈ error_now(x − v·dt)`. Each sensor
stops being a point and becomes a track.

**The velocity is not a new free parameter.** A slippery bed carries its own
pattern faster, so `v = V_SCALE/β`, and temperature sets β through the known map
— warm ⇒ low β ⇒ fast. `V_SCALE = 2800` is pinned by matching double-bump's
2.0 m/s at its mean β of 2000; mean β here is ~1400, so 2.0 × 1400 = 2800. Same
transport rate, so the comparison is controlled.

It is also **state-dependent** — the velocity is read from the field being
advected, so a particle with wrong β transports its own error at the wrong rate
and every particle deforms differently. A shared velocity field cannot do that.

**Honest label:** 2 m/s is ~10⁸× faster than real ice. This is a numerical device
for moving the bed pattern, exactly like double-bump's advection (whose own
comment called it "a crude stand-in for a bed whose properties evolve"). It is
kept identical so the comparison is fair, not because it is glaciology.

## The diagnostics

### 1. The error field sweeps the domain — and the correlation proves it

Pattern correlation of the error field with its own shape at step 0:

| step | 0 | 4 | **9** | 16 | **21** | 24 |
|---|---|---|---|---|---|---|
| correlation | 1.000 | 0.322 | **−0.410** | −0.027 | **+0.514** | 0.378 |
| `temp_field_run` | 1.000 | 0.982 | 0.951 | 0.929 | 0.775 | 0.850 |

It goes **negative**, then comes back positive. That is not decay — it is the
error pattern being carried around the periodic domain and arriving back.

The numbers check out exactly. Transport is 7.7 km/step and the background wave
has a dominant wavelength of ~160 km, so:

- half a wavelength (anti-phase, correlation most negative) = 80/7.7 ≈ **10 steps** → observed minimum at step 9 ✓
- a full wavelength (back in phase) = 160/7.7 ≈ **21 steps** → observed maximum at step 21 ✓

The diagnostic traces a clean cosine at precisely the designed period. The
transport is doing exactly what it was built to do.

### 2. Distant cells were the big winners

| cells | count | RMSE start → end | reduction | in `temp_field_run` |
|---|---|---|---|---|
| near a sensor (≤10 km) | 336 | 352.4 → 133.8 | **62.0%** | 17.5% |
| **far from sensors** | 1264 | 283.4 → 123.0 | **56.6%** | **9.5%** |

Far-from-sensor cells improved **6× better than before** (9.5% → 56.6%). That is
the entire point: a cell 50 km from any sensor is no longer invisible, because
the flow carries it into view.

**A surprise worth noting:** for much of the run the *near* cells are the worse
of the two (step 10: near 220.7, far 172.3), and they finish worse too
(133.8 vs 123.0). The likely reason is that near-sensor cells are continuously
**re-supplied with fresh, never-yet-observed error advected in from upstream**,
while a far cell that already passed a sensor got corrected and then drifts away
clean. Sensor proximity stops being an advantage once transport is doing the
work.

### 3. Transport actually happened

| | |
|---|---|
| mean velocity | 2.15 m/s (target 2.0) |
| per step | 7.7 km ≈ 1.9 cells |
| over the run | **186 km**, against a 160 km domain — a full sweep |
| Courant number | 0.19 (stable well below 1) |
| velocity clamp bound | **0.022%** of cell-substeps — a safety net, never a constraint |

This is reported rather than assumed: a run in which β travelled 2 km would have
tested nothing.

### 4. The filter is healthier too

ESS mean 309.1 (was 280.1) and **8 resamples instead of 13**. With informative
observations the weights behave better, so less diversity is spent on
resampling and less jitter is injected. Better information makes the filter
cheaper as well as more accurate.

## What this settles

Across the series the controlled comparisons now say:

- **Realism was never the problem.** Adding topography to strengthen β's spatial
  structure 8× made things *worse* (4.4%). Field realism is not what the filter
  needs.
- **Noise tuning was never the problem.** Cutting jitter 20 → 5 also made things
  worse (0.4%).
- **Transport was the whole problem.** Adding it, with everything else held
  fixed, moved 11.8% → 58.1%.

What made `double_bump_run` work was never the double bump. It was the
advection.

## What would be worth trying next

1. **Run longer.** RMSE was still falling at −0.79/step at step 24. 48 steps
   (two diurnal cycles) is the cheapest remaining gain and would show whether
   this passes double-bump's 112.6.
2. **Transport in y as well as x.** Advection is +x only, so `y` rows with no
   sensor still never pass beneath one. A 2D velocity would close the last gap
   — and the near/far inversion above suggests coverage is now the binding
   constraint again.
3. **Localisation.** Still untested, still cheap, and now more likely to help
   than when the error field was frozen.

## Outputs

| file | contents |
|---|---|
| `diagnostics.png` | **the key figure** — error pattern correlation (with the `temp_field_run` line for reference) and observability |
| `rmse_beta.png` | RMSE(β) vs time, split near / far from sensors |
| `ess_tracking.png` | ESS per step vs the N/2 threshold |
| `forcing_timeseries.png` | temperature forcing and the interior-mean β response |
| `beta_evolution.gif` | truth β \| mean β \| error, per step |
| `velocity_evolution.gif` | truth \| mean \| absolute error \| relative error, per step |
| `tracking.h5` | everything numeric, incl. all four diagnostic series |
| `summary.txt` | every number the run printed |

## Settings

| | value |
|---|---|
| domain / grid | 160 km square, 40×40, 4 km cells |
| surface | `z_s = 1060·√(1 − x/L)`, lapse rate 0.0059 °C/m |
| forcing | diurnal ±10 °C, seasonal ±15 °C, AR(1) wobble sd 1.0 °C, τ 6 h |
| β map (**known**) | `β = 2000 − 40·(T + 10)`, floor 10 |
| **transport** | `v = 2800/β` m/s, clamped [0.5, 5.0], 10 substeps of 360 s, +x periodic |
| background | truth + 300·(smooth wave, seed 29) — one shared wrong field |
| init spread / process / jitter | 200 / 10 / 20, correlation length 15 km |
| N, T, K, σ_obs | 500, 24 hourly, 10, 0.5 |
| sliding | `weertman_m = 1` (τ_b = β·u) |
| boundaries | `u_iszero=["north"]`, `v_iszero=["south","east","west"]` |

```
julia -t <threads> run_temperature_advect_pf.jl [N] [T] [K] [σ_obs] [seed_pf] [seed_obs]
julia -t 4  run_temperature_advect_pf.jl 20 4 3     # smoke test, ~10 s
julia -t 10 run_temperature_advect_pf.jl 500 48 10  # two diurnal cycles
```

**Boundary conditions.** WAVI's orientation names are array indices, not compass
directions: `"north"/"south"` are the **x** edges, `"east"/"west"` the **y**
edges. Zero the component that crosses each edge — `u` for x-edges, `v` for
y-edges. Without the sidewalls those edges act as calving fronts and run ~32×
faster than the interior.
