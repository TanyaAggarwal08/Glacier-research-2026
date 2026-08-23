# temp_field_run — improving the β field, with temperature instead of advection

Self-contained. One file, one command, all results.

```bash
julia -t 10 glacier-code/temp_field_run/run_temperature_field_pf.jl
```

Results → `results/temperature_field/`. Measured **35.4 min**; budget 30–90 min
(per-solve cost on this machine has ranged 0.10–2.7 s with load).

> Use plain `julia`, **not** `julia --project=.` — the repo project environment
> has a broken `NetCDF_jll` artifact (`libhdf5_hl.310.dylib` missing) and WAVI
> will not load. The default environment works.

## What this is

The double-bump experiment with exactly **one** substitution:

| | `double_bump_run` | `temp_field_run` |
|---|---|---|
| state | β field, 1600 cells | **same** |
| background | truth + 300·(smooth wave), seed 29 | **same** |
| ensemble | background + 200·(smooth wave) | **same** |
| process / jitter | 10 / 20 | **same** |
| sensors, σ_obs, N, K | 16, 0.5, 500, 10 | **same** |
| **dynamics** | nonlinear advection | **βₜ₊₁ = βₜ − k·(Tₜ₊₁ − Tₜ)** |

Nothing else differs. The β(T) map is **given** (`C = 2000`, `k = 40`) — unlike
its sibling `temp_pf_run`, there is nothing to estimate but the field itself.

## Result

| | RMSE(β) | reduction |
|---|---|---|
| start (background) | 299.2 | |
| **end (step 24)** | **264.0** | **11.8%** |
| best (step 11) | 257.5 | 13.9% |

speed RMSE 2.75 → 2.13 m/yr · ESS mean 280.1, min 223.5 · 13 resamples ·
17,524 WAVI solves

This reproduces the flat-lapse temperature baseline from the (since-lost)
`realistic_beta_run` **exactly** — 299.2 → 264.0, 11.8% — which is a useful
check that the rebuild is faithful.

## The diagnostics, which are the point

### 1. Sensor coverage is a real limit — and now measured

| cells | count | RMSE start → end | reduction |
|---|---|---|---|
| **near** a sensor (≤ 10 km) | 336 | 352.4 → 290.9 | **17.5%** |
| **far** from sensors | 1264 | 283.4 → 256.3 | **9.5%** |

Cells an observation can genuinely speak about improve **1.8× faster** than the
rest. So the filter is not broken — it works where it can see, and 79% of the
domain is somewhere it cannot.

**But coverage is not the whole story.** Extrapolate generously: if every cell
behaved like a near-sensor cell, the run would land at ~17.5%. Against
advection's **62%**, that leaves most of the gap unexplained. Adding sensors
will help, and by a measurable amount — it will not by itself close it.

### 2. The error field barely deforms

Pattern correlation of the error field with its own shape at step 0:

| step | 0 | 6 | 12 | 18 | 24 |
|---|---|---|---|---|---|
| correlation | 1.000 | 0.960 | 0.934 | 0.837 | **0.850** |

After a full diurnal cycle the error is still **85% the same field it started
as**. This is structural, not a tuning artefact. The temperature increment
`−k·(Tₜ₊₁ − Tₜ)` depends only on the shared temperature record, never on the
state, so it is added identically to the truth and to every particle and
cancels out of the difference:

```
(β_p − β_true)ₜ₊₁ = (β_p − β_true)ₜ + (η_p − η_true)
```

The error field is a pure random walk. What little decorrelation appears above
comes from process noise and jitter — not from the dynamics.

Advection was different because its velocity was **state-dependent**
(`1 + ε·β`): a particle with the wrong β pushed its own error into new places,
so 16 fixed sensors kept being shown new aspects of it. Temperature cannot do
that. This is why the same filter, the same background and the same sensors
give 62% there and 12% here.

### 3. Observability rises and falls with the bed

With `weertman_m = 1`, speed ≈ driving stress / β, so `∂log(speed)/∂β = −1/β`:
a given β error is **loudest when the bed is warm and slippery**. Over the
cycle `1/β̄` swings 5.81e-4 → 9.81e-4, a factor of **1.76**.

| | net ΔRMSE |
|---|---|
| warming limb (steps 1–14, T: −4 → +14.5 °C) | **−40.2** |
| cooling limb (steps 14–24, T: +14.5 → −3 °C) | **+4.9** |

Essentially all the improvement happens while the bed is warming, and the
cooling half gives a little back.

**Read that cautiously.** The warming limb is also the *early* part of the run,
when there is the most removable error, and any filter improves fastest early.
The two effects are confounded here and this run cannot separate them — the
step-by-step correlation between ΔRMSE and observability is only 0.26, which is
weak. Starting the run at a different diurnal phase would separate them.

## Where this leaves the three runs

| run | dynamics | unknown | RMSE | reduction |
|---|---|---|---|---|
| `double_bump_run` | advection (state-dependent) | β field, 1600 | 298.8 → 112.6 | **62%** |
| `temp_field_run` (this) | temperature (state-independent) | β field, 1600 | 299.2 → 264.0 | **11.8%** |
| `temp_pf_run` | temperature | C, k + residual | 439.3 → 67.4 | **84.7%** |

`temp_pf_run` looks best and is best, but not because temperature helped the
field: 16 sensors identify **2 parameters** easily. Its residual field degraded
throughout (2.0 → 63.5) and made up 94% of its final error — the same disease
measured here.

**The honest summary:** a realistic-looking β is not automatically an easier β
to recover. What made double-bump work was the *dynamics*, not the field.

## FOLLOW-UP: this was fixed — see `temp_advect_run`

The diagnosis above was acted on. Letting temperature drive **transport** as well
as drag (`v = 2800/β`, so warm ⇒ slippery ⇒ fast) — with the background, the
spread, the noise, the sensors and the filter all unchanged — moved the result:

| | RMSE(β) | reduction | error pattern corr |
|---|---|---|---|
| this run (no transport) | 299.2 → 264.0 | 11.8% | 0.850 |
| `temp_advect_run` | 299.2 → **125.4** | **58.1%** | **0.378** (dips to −0.410) |

Far-from-sensor cells went from 9.5% to **56.6%**. Diagnostic 2 below was the one
that pointed at the fix, and it is what confirmed it worked.

This run remains the correct **control**: it isolates what a state-independent
temperature increment can do on its own, which is very little.

## What would actually be worth trying

Ranked by what the diagnostics above now support, rather than by guesswork:

1. **More sensors.** The near/far split makes this concrete for the first time —
   near-sensor cells really do improve 1.8× faster. Doubling to 32 or 64 is the
   direct, honest test, and the near/far numbers predict roughly where it lands.
2. **Localisation.** 336 of 1600 cells are informed but every observation
   currently reweights the *whole global field*. Restricting each observation's
   influence to its neighbourhood is the standard fix for exactly this regime
   and needs no extra data.
3. **Anything state-dependent in the dynamics.** Diagnostic 2 says this is the
   larger of the two effects. Even a weak β-dependent term would start deforming
   the error field and let fixed sensors see new directions.

Already tested and **not** worth repeating (from the lost `realistic_beta_run`):
adding topography to strengthen β's spatial signal 8× made recovery *worse*
(4.4%), and cutting jitter 20 → 5 also made it worse (0.4%). Noise tuning and
field realism are both dead ends here.

## Outputs

| file | contents |
|---|---|
| `rmse_beta.png` | RMSE(β) vs time, split into near / far from sensors |
| `diagnostics.png` | **the key figure** — error pattern correlation, and observability 1/β̄ |
| `ess_tracking.png` | ESS per step vs the N/2 threshold |
| `forcing_timeseries.png` | temperature forcing and the interior-mean β response |
| `beta_evolution.gif` | truth β \| mean β \| error, per step |
| `velocity_evolution.gif` | truth \| mean \| absolute error \| **relative error**, per step |
| `tracking.h5` | everything numeric, incl. all three diagnostic series |
| `summary.txt` | every number the run printed |

Velocities are **not** stored during assimilation — β is the complete state, so
speed at every frame is recovered afterwards by re-solving WAVI on the saved β
(2·(T+1) = 50 solves, ~10 s).

The velocity animation carries a fourth **relative-error** panel: β falls while
speed rises over the warm half of the cycle, so the same fractional error looks
small in absolute β and large in absolute speed. The relative panel is the
honest common measure.

## Settings

| | value |
|---|---|
| domain / grid | 160 km square, 40×40, 4 km cells |
| surface | `z_s = 1060·√(1 − x/L)`, lapse rate 0.0059 °C/m |
| forcing | diurnal ±10 °C, seasonal ±15 °C, AR(1) wobble sd 1.0 °C, τ 6 h |
| β map (**known**) | `β = 2000 − 40·(T + 10)`, floor 10 |
| background | truth + 300·(smooth wave, seed 29) — one shared wrong field |
| init spread / process / jitter | 200 / 10 / 20, correlation length 15 km |
| N, T, K, σ_obs | 500, 24 hourly, 10, 0.5 — one diurnal cycle |
| sliding | `weertman_m = 1` (τ_b = β·u) |
| boundaries | `u_iszero=["north"]`, `v_iszero=["south","east","west"]` |

```
julia -t <threads> run_temperature_field_pf.jl [N] [T] [K] [σ_obs] [seed_pf] [seed_obs]
julia -t 4  run_temperature_field_pf.jl 20 4 3     # smoke test, ~20 s
```

**Boundary conditions.** WAVI's orientation names are array indices, not compass
directions: `"north"/"south"` are the **x** edges, `"east"/"west"` the **y**
edges. Zero the component that crosses each edge — `u` for x-edges, `v` for
y-edges. Without the sidewalls those edges act as calving fronts and run ~32×
faster than the interior.
