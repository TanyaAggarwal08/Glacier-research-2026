# temp_pf_run — learning the β(T) relationship from velocity observations

Self-contained. One file, one command, all results.

```bash
julia -t 10 glacier-code/temp_pf_run/run_temperature_pf.jl
```

Everything lands in `results/temperature_pf/`. **Budget 20-90 minutes** — the work
is fixed at ~15,000 WAVI solves, but per-solve cost on this machine has been
measured from 0.077 s (21 min) to 2.70 s (11 hours) depending on load. Check
`uptime` before starting a long one.

## The question

Basal drag is assumed to follow surface temperature through a linear map

```
β(x,y,t) = C − k·(T(x,y,t) − T_ref) + δ(x,y,t)
           └──── the map ────┘        └ residual ┘
```

but **the map is not known** — `C` and `k` are what we want to find out. The only
data are noisy observations of ice surface **speed** at 16 sensors. Can a
particle filter recover the relationship?

## The answer: yes

| | prior | final | truth | error |
|---|---|---|---|---|
| **C** (β at T_ref) | 2296.8 | **1979.5 ± 69.3** | 2000.0 | **1.0%** |
| **k** (sensitivity) | 19.85 | **40.24 ± 3.67** | 40.00 | **0.6%** |

RMSE(β) 439.3 → **67.4** (85% reduction), split as map **22.1** + residual **63.5** ·
speed RMSE 3.70 → **0.56** m/yr · ESS mean 319.7, min 233.4 · 7 resamples ·
15,024 WAVI solves

The prior is deliberately wrong: `k` starts at **half** the true sensitivity, so
the filter has to roughly double it from 16 noisy speed sensors alone.

### A tuning lesson worth keeping

The first version of this run used `DELTA_INIT_SD = 150` and `JITTER_DELTA = 20`
and produced a result that *looked* like partial failure: RMSE(β) fell to 77 by
step 4 and then **climbed back to 175**, with the velocity error drifting up from
0.63 to 4.31 m/yr.

The map/residual split showed it was not a filter failure at all — the map error
was 4.3 (essentially perfect) and the entire problem was the residual, which grew
to 175. The cause: δ has 1600 degrees of freedom constrained by 16 sensors, so
the filter cannot distinguish between δ fields that fit equally well and simply
commits to whichever prior draw helps. With a prior **3× wider than the truth
ever reaches** (filter δ sd 154 vs truth 51), that commitment injected structure
that was not there.

Matching the prior to the scale the truth actually reaches — `DELTA_PROC_SD·√T`
≈ 50 — and shrinking the per-resample jitter fixed it:

| | wide prior | matched prior |
|---|---|---|
| RMSE(β) final | 174.5 | **67.4** |
| from the residual | 175.4 | **63.5** |
| δ sd, filter vs truth | 154 vs 51 (**3.0×**) | 36 vs 51 (**0.70×**) |
| k error | 1.4% | **0.6%** |
| speed RMSE final | 1.54 | **0.56** m/yr |

The ratio is now 0.70, i.e. slightly *under*-dispersed rather than over — a mild
opposite bias, and much the safer side to sit on.

## How it differs from the double-bump experiment

| | double-bump run | this run |
|---|---|---|
| unknown | the β field | **the β(T) map**, plus a residual field |
| state | 1600 | **1602** (C, k, δ) |
| forecast step | advect β in x | **temperature advances; no advection at all** |
| what drives time evolution | advection velocity | **the diurnal temperature cycle** |
| observation | log-speed at 16 sensors | same |
| method | tempered PF | same |

Each particle carries its own `C`, `k` and smooth residual `δ`. At every step the
temperature moves one hour (known), each particle re-derives its β through *its
own* map, WAVI turns that into velocity, and particles whose predicted speeds
match the observations gain weight.

Error lives on β, as in the earlier work: initial error and process noise both
enter through `δ`, so the filter corrects the map, the initial error and the
accumulated process noise together.

## Why it works, and what to watch

**The diurnal swing is the signal.** As the day warms and cools, the truth's β
moves and the ice measurably speeds up and slows down. A particle with the wrong
`k` predicts the wrong *amount* of speed-up, and the cycle re-tests it every hour.
That is why the default run is 24 hourly steps — one full cycle, so every
particle is tested across the whole temperature range rather than a narrow slice.

**C and k are identified by different things**, and `parameter_recovery.png`
shows it clearly:

- **C** is fixed by the instantaneous mean level of β. It drops onto the truth
  within ~2 hours and stays there.
- **k** is fixed only by the *time variation*. It climbs steadily from 20,
  crosses 40 around hour 10, and its uncertainty tightens from ±11.5 to ±2.8.

**Identifiability caveat.** `C` and `δ` are only partly separable — a uniform
shift in `δ` is indistinguishable from a shift in `C` at any single instant, since
both raise β everywhere equally. What breaks the tie is that `δ` is spatially
structured while `C` is exactly uniform, and that `k` is pinned by time variation
which the near-static `δ` cannot mimic. `k` is therefore the more meaningful test
of the two.

**RMSE(β) settles near 67 while the parameters are recovered to ~1%.** The split
printed in the summary and drawn in `rmse_beta.png` shows why: map 22.1, residual
63.5. The map is identifiable from 16 sensors; the fine spatial structure of δ
is not, and never will be — 1600 unknowns against 16 observations. The headline
result is the parameter recovery, and the residual sets the floor on field RMSE.

**Watch for the spike near the temperature extremum** (around step 13 in the
default run: β RMSE 140, speed RMSE 2.6, against ~40 and ~0.4 either side). The
decomposition shows it is the MAP curve that spikes, not the residual, and the
reason is structural: β = C − k(T − T_ref), so a small remaining error in `k` is
multiplied by (T − T_ref), which is largest exactly when the temperature is
furthest from the reference. The filter then corrects it and the error falls
back. It is a real effect of the parameterisation, not instability.

## Outputs

| file | contents |
|---|---|
| `parameter_recovery.png` | **the key figure** — C and k posteriors ± sd against truth |
| `parameter_scatter.png` | particle cloud in (C, k) space, prior vs posterior |
| `rmse_beta.png` | RMSE(β) against time |
| `ess_tracking.png` | ESS per step vs the N/2 threshold |
| `forcing_timeseries.png` | the temperature forcing and the β response |
| `beta_evolution.gif` | truth β \| mean β \| error, per step |
| `velocity_evolution.gif` | truth speed \| mean speed \| error, per step |
| `tracking.h5` | everything numeric, including per-particle final parameters |
| `summary.txt` | every number the run printed |

## Setup details

**Temperature is known.** One field, generated once and shared by the truth and
every particle: seasonal + diurnal + elevation lapse, plus a smooth AR(1) wobble
(sd 1 °C, τ 6 h) so it looks like a real record rather than a clean sinusoid.
Because everyone sees the same T it is realistic *forcing*, not a source of
estimation error. All the uncertainty is in β.

**Parameters get their own jitter on resample** (±25 on C, ±1.5 on k). The
smooth-field noise only touches `δ`; without separate parameter jitter the (C, k)
cloud collapses to duplicated values after the first resample and can never adapt
again.

**Boundary conditions.** WAVI's orientation names are array indices, not compass
directions: `"north"/"south"` are the **x** edges, `"east"/"west"` the **y** edges.
For any edge, zero the component that crosses it — `u` for x-edges, `v` for
y-edges. This run uses a wall at x = 0 (`u_iszero = ["north"]`, the ice divide)
and free-slip sidewalls at y = 0 and y = L. Without the sidewalls those edges act
as calving fronts and run ~32× faster than the interior.

**Linear sliding** (`weertman_m = 1`), so `τ_b = β·u` and the field passed to WAVI
is identically the drag coefficient being estimated.

## Arguments

```
julia -t <threads> run_temperature_pf.jl [N] [T] [K] [σ_obs] [seed_pf] [seed_obs]
```

| arg | default | meaning |
|---|---|---|
| `N` | 500 | particles |
| `T` | 24 | hourly steps (24 = one diurnal cycle) |
| `K` | 10 | tempering stages |
| `σ_obs` | 0.5 | observation sd, log-speed units |
| `seed_pf` | 42 | filter seed |
| `seed_obs` | 123 | truth + observation-noise seed |

```bash
julia -t 4  run_temperature_pf.jl 20 4 3      # smoke test, ~30 s
julia -t 10 run_temperature_pf.jl             # standard run, 20-90 min (load-dependent)
julia -t 10 run_temperature_pf.jl 500 48 10   # two diurnal cycles
```

**Threads matter** — the N likelihood evaluations per stage are independent WAVI
solves, spawned in parallel. Use `-t 8..10`.

## Where this could go next

- **Harder priors.** `k` starting at half truth is recovered easily; try a factor
  of 4 out, or a wrong sign, to find where identifiability breaks.
- **Fewer sensors, or noisier ones.** 16 sensors and σ_obs = 0.5 is generous.
- **A nonlinear map.** The linear form is itself an assumption; a quadratic or
  threshold map would test whether velocity data can distinguish functional forms.
- **Melt-water hydrology**, deliberately excluded here to keep the first version
  interpretable.
