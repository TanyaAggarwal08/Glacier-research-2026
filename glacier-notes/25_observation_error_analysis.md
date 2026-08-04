# Observation error analysis — is σ_log = 0.20 actually calibrated?

Purpose: run14 switched the observation operator from `ux = 1000/β` to
`log(β)` at the sensor cells, with `obs_noise_std_log = 0.20`. That value was
justified by a plausibility argument ("0.20 ≈ 20 % relative error"), not a
measurement. This note records the measurement: what the observation-space
ensemble spread actually is, how σ_log compares to it, what the full ESS trace
looks like, and exactly which variables differ between run12 and run14.

**Nothing here was retuned.** Everything below is read out of the existing
`run14_.../tracking.h5`, so the numbers describe the run as it was published.

Diagnostic script:
`glacier-code/particleda/rmse_experiments/diagnose_14_log_obs_calibration.jl`

```bash
julia --project=test glacier-code/particleda/rmse_experiments/diagnose_14_log_obs_calibration.jl
```

## 1. The operator being tested

`glacier_model.jl` now carries an `obs_space` switch with three values:

| `obs_space` | H(β) at sensor k | σ used |
|---|---|---|
| `velocity` (default) | `ux = 1000/β` | `obs_noise_std` |
| `log_velocity` | `log(1000) − log(β)` | `obs_noise_std_log` |
| `log_beta` (run14) | `log(max(β, min_beta))` | `obs_noise_std_log` |

The two log modes differ only by a sign and a constant. The likelihood sees
only the squared residual, so they induce **the same** Gaussian on relative
β-misfit — `log(β_particle) − log(β_truth) + ε`. That is the whole point of
log space: one σ buys the same *fractional* accuracy at every β, whereas in
velocity space a single σ means 10 % relative error at β = 1000 but 30 % at
β = 3000.

The delta method (`σ_log ≈ σ_v/ux = σ_v·β/1000`) is what produced 0.20: it is
the equivalent of the incumbent `σ_v = 0.10` **at the prior centre β = 2000**
(where `ux = 0.5`). Note that equivalence holds *only* at that centre.

## 2. Observation-space spread, s_obs

`s_obs` = std across the 1000 particles of the predicted observation `log(β)`,
per sensor. Measured on the **forecast** ensemble: run14 stores
`all_particles[:, :, t+1] = particles` after propagation but *before*
resampling, so slice `t+1` is exactly the ensemble the likelihood scored at
step `t`. That is the correct ensemble for this comparison.

| t | mean s_obs | median | min | max |
|---|---:|---:|---:|---:|
| 0 | **0.1124** | 0.1069 | 0.0832 | 0.2417 |
| 50 | **0.0320** | 0.0315 | 0.0207 | 0.0488 |
| 100 | **0.0424** | 0.0416 | 0.0305 | 0.0605 |

Per-sensor values are tight around the mean, with one outlier: sensor 22 at
t = 0 sits at 0.2417, roughly 2× every other sensor.

Sanity check on the machinery: `init_std_beta = 200` at β ≈ 2000 is a 10 %
relative spread, and log-space spread ≈ relative spread, so the measured
0.1124 at t = 0 is right where it should be.

## 3. σ_log versus s_obs

Target band: observation noise should sit at roughly 0.1–0.3× the
observation-space spread.

| t | mean s_obs | ratio σ_log/s_obs | ratio range across sensors | verdict |
|---|---:|---:|---|---|
| 0 | 0.1124 | **1.78** | 0.83 – 2.40 | too loose |
| 50 | 0.0320 | **6.26** | 4.10 – 9.67 | too loose |
| 100 | 0.0424 | **4.72** | 3.31 – 6.55 | too loose |

**0.20 does not fall in the 0.1–0.3 band at any timestep.** It misses by ~6×
at t = 0 and by ~20–60× at mid-run and final. At t = 0 the assumed noise is
already 1.8× the *entire prior ensemble spread* — the "noise ≳ spread"
regime, where per sensor per step the observations are nearly uninformative.

Figure: [s_obs_vs_sigma_run14.png](../glacier-code/particleda/results/rmse_analysis/s_obs_vs_sigma_run14.png)

**Caveat on the t = 50 and t = 100 rows.** `s_obs` there is not an
independent measurement. The ensemble is narrow at mid-run partly *because*
the filter has been resampling on 87 % of steps and collapsing it. So
small `s_obs` → large ratio is partly circular. The t = 0 ratio of **1.78** is
the only one measured on a genuinely prior ensemble, and it is the one to
trust for tuning.

## 4. ESS trace — the mean hides an early collapse

Mean ESS of 321 is not the story.

| Metric | Value |
|---|---:|
| mean ESS | 321.1 |
| median ESS | 329.6 |
| **minimum ESS** | **5.6** |
| **timestep of minimum** | **t = 5** |
| steps with ESS < N/2 | **87 / 100 (87 %)** |
| steps with ESS < N/10 | 7 / 100 |

First ten steps: **8.4**, 148.8, 637.9, 153.2, **5.6**, 248.0, 178.8, 217.6,
224.8, 249.5

The filter collapses to ESS ≈ 8 on the *very first* assimilation, bounces,
collapses again to 5.6 at t = 5, and only after ~t = 10 settles into
oscillating between roughly 100 and 670 around the N/2 trigger. Resampling
therefore fires on nearly every step.

Figure: [ess_trace_run14_logbeta.png](../glacier-code/particleda/results/rmse_analysis/ess_trace_run14_logbeta.png)

## 5. run12 vs run14 — what is and isn't confounded

The code side is genuinely clean. Verified numerically from the two HDF5
files, not just by reading the drivers:

| Check | Result |
|---|---|
| truth fields | **bit-identical** (max \|Δβ\| = 0.000e+00) |
| initial ensembles | **bit-identical** (max \|Δβ\| = 0.000e+00) |
| sensor layout | identical (same 30 stations) |
| truth prior / background prior | identical |
| N particles | 1000 vs 1000 |
| T steps | 100 vs 100 |
| `init_std_beta`, `process_std_beta` | 200, 10 — identical |
| `noise_length_scale`, advection | 15 km, Lax–Wendroff — identical |
| `SEED_PF`, `SEED_OBS` | 42, 123 — identical |

The truth stays bit-identical because both operators draw exactly `n_obs`
normals per step, so the `rng_truth` stream never desynchronises. The only
differences in the driver are `obs_space => "log_beta"`,
`obs_noise_std_log => 0.20`, the output directory, and some extra printed
HDF5 attributes.

Headline comparison:

| Metric | run12 (velocity) | run14 (log β) |
|---|---:|---:|
| RMSE(β) final | 196.5 | **169.7** |
| RMSE(β) mean | 230.7 | **215.5** |
| mean ESS | 265.3 | **321.1** |
| min ESS | **1.0** | 5.6 |

**Two things still confound the interpretation, and neither shows up in a diff:**

1. **The observations aren't the same data.** run12 assimilates
   `y = ux + N(0, 0.1²)`; run14 assimilates `y = log β + N(0, 0.2²)`. The
   0.20 was picked to be delta-method-equivalent to σ_v = 0.10 *at β = 2000
   only* — away from the prior centre the two runs carry genuinely different
   information. So "log wins" is entangled with "log happened to get a
   better-matched noise level."
2. **One seed, no error bars.** The 6.6 % mean-RMSE improvement comes from a
   single (SEED_PF = 42, SEED_OBS = 123) realisation of a filter that is
   resampling on 87 % of steps and hitting single-digit ESS. run12's own
   minimum ESS is 1.0 — it degenerated to a single particle at some point.
   A 6.6 % gap should not be treated as a real effect without repeat seeds.

## 6. The tension that has to be resolved before retuning

The ratio test says *too loose, observations nearly uninformative*. The ESS
trace says *collapsing to 5 particles*. These sound contradictory. Both are
true, and the resolution determines which way σ_log should move.

**The ratio is a per-sensor, per-step quantity. The weight is not.** The
log-weight sums squared residuals over all 30 sensors, and `log_weights`
*accumulates across steps* (the Liu & Chen recipe in
`22_lw_rpf_reference.md`) until a resample resets it. Discrimination
therefore compounds two ways the per-sensor ratio cannot see:

- 30 sensors summed into one weight, and
- weights carried across several steps between resamples.

A σ that looks flat against one sensor at one step can still produce a
near-degenerate weight vector once both effects are included.

**Consequence:** lowering σ_log — the direction the naive ratio test points —
would likely make the ESS collapse *worse*, not better. The 0.1–0.3 band is
calibrated for a single effective observation. With 30 spatially-correlated
sensors (ℓ = 15 km on a 160 km domain, so the effective count is well below
30) plus cumulative weights, the correct target band is wider.

**Recommended next diagnostic:** measure the spread of the *total* log-weight
across particles, not per-sensor `s_obs`. That is the quantity ESS actually
responds to, and it is what should drive the σ_log decision.

## 7. Status

σ_log = 0.20 is **unverified, not vindicated**. The run14 RMSE improvement is
real but small, single-seed, and confounded with a change in effective
observation informativeness. No parameter has been changed as a result of
this analysis.

## See also

- `24_pseudorandom_wave_run.md` — the run11/run12 pseudo-random wave setup
  these runs build on
- `13_observation_noise.md` — why σ_obs = 0.10 in velocity space, and the
  `ux = 1000/β` noise-amplification argument this note's log operator is
  meant to fix
- `22_lw_rpf_reference.md` — ESS-gated resampling, cumulative log-weights,
  and the RPF jitter that drives the s_obs circularity in §3
- `15_observations_ablation.md` — ESS as the cleanest "are observations doing
  anything?" indicator
- `16_beta_space_state.md` — why the state is β rather than log β, which is a
  separate question from the observation operator
