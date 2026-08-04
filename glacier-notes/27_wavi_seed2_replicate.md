# WAVI observation operator — second seed (replicate of Exp 18)

Purpose: re-run the Exp 18 base setup (WAVI as the observation operator,
upwind + nonlinear advection + pseudo-random-wave prior + tempering) with a
**different random seed**, changing nothing else, to check whether the β
recovery reported in `26_wavi_observation_operator.md` is a property of the
filter or a lucky single-seed draw. This is the "repeat seeds for error bars"
next-step from that note — the first independent replicate.

## 1. What changed vs Exp 18 (seed 1)

**Only the seeds.** Both `SEED_PF` (particle initial ensemble + all
resampling/jitter draws) and `SEED_OBS` (truth stochastic trajectory +
observation noise) were changed, so this is a genuinely independent
realisation: a different truth path, different observations, and a different
particle ensemble. Everything else — N, T, K, σ_obs, ε, advection scheme,
prior, WAVI operator, sensor layout — is identical.

| | Exp 18 (seed 1) | seed 2 (this run) |
|---|---:|---:|
| SEED_PF | 42 | **7** |
| SEED_OBS | 123 | **456** |
| N / T / K | 500 / 20 / 10 | 500 / 20 / 10 |
| σ_obs / ε | 0.5 / 5e-4 | 0.5 / 5e-4 |
| advection | nonlinear upwind | nonlinear upwind |

The `prior_truth_seed`/`prior_background_seed` in `Params` are unchanged, so
the truth's *initial* (t=0) field is the same base field; the two runs diverge
through the stochastic evolution, observations and ensemble draw.

To make this possible the run scripts now take the seeds and an output
run-tag via optional ARGS (positions 7–9 on `experiment_18_wavi_obs.jl`; the
tag as ARG 1 on the two plot scripts), all defaulting to the original Exp 18
values so seed-1 behaviour is untouched.

## 2. Results — the recovery reproduces

| metric | seed 1 | **seed 2** |
|---|---:|---:|
| RMSE(β) initial → final | 299.2 → 143.9 | **301.1 → 157.3** |
| RMSE(β) reduction | ×0.48 | **×0.52** |
| mean ESS | 292 (58% N) | **294.9 (59% N)** |
| min ESS | 230 @t=1 (46% N) | **227.0 @t=5 (45% N)** |
| velocity RMSE at sensors (rel) | 17.7% → 7.6% | **21.9% → 8.6%** |
| velocity RMSE at sensors (abs) | 1089 → 425 m/yr | **1300 → 456 m/yr** |
| wall time | 96 min | **104.6 min** |
| WAVI solves / tempering overhead | 13,540 / ×1.35 | **14,040 / ×1.40** |
| resamples | 8 | **8** |

**The conclusion holds under a fresh seed.** β RMSE again roughly halves
(×0.52 here vs ×0.48 for seed 1), the ESS again oscillates around N/2 with no
first-step (or any-step) collapse — the widest-ensemble step is min ESS ≈ 227
(45% N), this time at t=5 rather than t=1 — and the sensor velocity relative
RMSE again falls to single digits (8.6% vs 7.6%). β RMSE and velocity RMSE move
together in both runs, the same consistency check as seed 1: the *state* is
genuinely improving, not just fitting observation noise. Cost and tempering
overhead are essentially the same (8 resamples both runs, ×1.35–1.40).

Taken together the two seeds are strong evidence the particles genuinely help:
the improvement is reproducible across an independent realisation, not an
artefact of one draw.

## 3. Figures

Full diagram set in `results/experiment_18_wavi_obs_seed2/` (same set and
layout as the seed-1 folder):

- `rmse_beta.png` — monotone-ish decline 301 → 157, with the same mid-run
  (t≈8–15) plateau the seed-1 run showed before a late drop.
- `ess_tracking.png` — sawtooth around N/2, min 227 @t=5, all stages ≥ N/2 by
  the end of each step's tempering ladder.
- `beta_field_truth_vs_mean.png` — 3-panel: initial guess β (t=0 prior mean) |
  truth β (final) | mean β (final). The lumpy random-wave guess collapses onto
  the smooth truth structure, tightest near the sensors.
- `velocity_field_truth_vs_mean.png` — 3-panel WAVI speed: the initial guess's
  misplaced fast lobe (top) is corrected; final mean ≈ truth.
- `velocity_rmse_abs.png` / `velocity_rmse_rel.png` — sensor velocity RMSE vs
  time; relative drops 21.9% → 8.6%.
- `crosssection_final.png`, `sensor_trail.png`, `obs_fit.png` — as in seed 1.

The 3-panel field-evolution figures report β RMSE(initial-guess → final mean)
of 370.4 → 157.3 and speed RMSE 1904.5 → 425.9 m/yr. These "initial guess"
numbers are larger than the §2 initial RMSE (301.1) because they compare the
t=0 guess against the *final* (t=T) truth, so they also carry the truth's own
20-step advection — the same framing caveat noted for seed 1.

## 4. Files and reproduction

```bash
# default env (has WAVI); 10 threads. ARGS: N T K σ ε mode SEED_PF SEED_OBS run_tag
julia -t 10 glacier-code/particleda/experiment_18_wavi_obs.jl \
      500 20 10 0.5 5e-4 full 7 456 experiment_18_wavi_obs_seed2
julia glacier-code/particleda/plot_exp18_velocity_rmse.jl   experiment_18_wavi_obs_seed2
julia glacier-code/particleda/plot_exp18_field_evolution.jl experiment_18_wavi_obs_seed2
```

- Plots → `glacier-code/particleda/results/experiment_18_wavi_obs_seed2/`.
- `tracking.h5` + `checkpoint.h5` (~135 MB each) → external SSD at
  `/Volumes/ZX20/USRA 2026/experiment_18_wavi_obs_seed2/`.

## 5. Next steps

- With two seeds agreeing, a small seed sweep (3–5 seeds) would give proper
  error bars on RMSE(β) and ESS — cheap now (~1.7 h/run threaded).
- Still verification scale (N = 500, T = 20). The main-run fidelity push
  (N = 1000 and/or T = 50) is unchanged from the Exp 18 plan.

## See also

- `26_wavi_observation_operator.md` — Exp 18 seed-1 base run this replicates,
  and the full description of the setup, WAVI operator, and compute cost.
- `23_ice_flow_integration.md` — the WAVI wrapper (β → velocity).
- `25_observation_error_analysis.md` — σ calibration and the t=1 ESS collapse
  that tempering fixes (holds again here).
