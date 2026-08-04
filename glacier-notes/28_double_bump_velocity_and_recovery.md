# Double-bump test — does velocity respond to β, and can the filter recover it?

Purpose: a two-part test, motivated by the honest caveat in
`26_wavi_observation_operator.md` that the truth-vs-mean WAVI speed heatmaps
"look nearly identical" — i.e. it was unclear how strongly the surface velocity
actually responds to β. We swap the pseudo-random-wave prior for a clean,
large-amplitude **double-bump** truth (legacy sinusoid) and ask:

1. **Static:** does a structured β field produce a matching velocity signature?
2. **Filtered:** can the particle filter recover that structured β from the
   WAVI observations?

Both are opt-in tests — the pseudo_random_wave setup is the untouched default.

## 0. Reversible setup (nothing removed)

`double_bump` was re-added to `ice_experiment_dynamics.jl` alongside
`pseudo_random_wave` (default). Unlike the legacy twin (glacier_model.jl
returned `β_prior, β_prior`, so truth == prior), here the truth is the double
bump and the background keeps the seed2-style offset (`truth + background_std ·
wave`), so there is a genuine initial guess.

- Dynamics: `Params(prior_mode="double_bump", prior_amplitude_beta=…,
  prior_n_modes=…)`. Default stays `pseudo_random_wave`.
- Run script: `experiment_18_wavi_obs.jl` ARGS 10–12 = `prior_mode amplitude
  n_modes` (default `pseudo_random_wave 2000 3` → seed runs unchanged).

Truth field used throughout: **center 2000, amplitude 1500, n_modes 1** →
β ∈ [500, 3500], one 2×2 checkerboard of opposite bumps (two high, two low).

## 1. Static test — velocity clearly responds to β

`test_double_bump_velocity.jl` (no PF; 3 WAVI solves) evaluates WAVI on the
truth β, the initial-guess β, and a **uniform-β baseline**, then plots the
speed and the speed **anomaly** (speed − uniform), which isolates the β effect
from the geometry-driven background flow.

Results (`results/experiment_18_double_bump_test/`):

| quantity | value |
|---|---:|
| corr(β−center, truth speed anomaly) | **−0.820** |
| truth speed anomaly range | −4090 … +7766 m/yr |

- **Strong anti-correlation** (−0.82): higher β → more basal drag → slower ice,
  exactly as the Weertman law predicts. The bumps appear as clean opposite
  lobes in `velocity_anomaly.png` — red (fast) over low-β lobes, blue (slow)
  over high-β lobes.
- The anomaly is *essential* to see this: in the **raw** speed field
  (`velocity_field.png`) the geometry-driven pattern dominates and the bumps
  are hard to spot — the same effect noted in `26`. Subtracting the uniform-β
  baseline exposes the β signature.
- The velocity lobes are broader/smoother than the β bumps — the membrane-
  stress low-pass (ice responds to β non-locally over a stress-coupling
  length). Signal present and strong, just spatially spread.

## 2. Filtered test (Scenario B) — the filter recovers the bumps

Full PF with the double-bump truth and an **anchored** guess (truth + 300
offset, so the ensemble brackets the truth). Everything else = Exp 18.

Run: `experiment_18_wavi_obs.jl 500 20 10 0.5 5e-4 full 42 123
experiment_18_double_bump double_bump 1500 1`

| metric | value | vs pseudo-random seed 1 |
|---|---:|---:|
| RMSE(β) initial → final | **298.8 → 146.3** (×0.49) | 299.2 → 143.9 |
| mean ESS | 282.9 (57% N) | 292 |
| min ESS | **228 @t=2** (46% N) — no collapse | 230 @t=1 |
| velocity RMSE at sensors (rel) | **17.2% → 8.4%** | 17.7% → 7.6% |
| wall / solves / overhead / resamples | 93.5 min / 14,040 / ×1.40 / 8 | 96 min / 13,540 / ×1.35 / 8 |

Figures (`results/experiment_18_double_bump/`):

- **`crosssection_final.png`** — the clearest evidence: at y=48 km the ensemble
  mean sits almost on top of the truth across the whole double-bump profile
  (peak ~2700 → trough ~1050 → rise), with a tight 5–95% particle band.
- **`beta_field_truth_vs_mean.png`** — the noisy initial guess collapses onto
  the truth's bump structure; the high-β and low-β lobes are recovered.
- **`velocity_field_truth_vs_mean.png`** — the initial-guess speed is visibly
  wrong; the recovered mean matches truth. Here the correction is visible even
  in the **raw** speed (not only the anomaly), because the β signal is large.

**Reading of the comparison.** The absolute β RMSE (~146) is essentially the
same as the pseudo-random runs — but that is a *stronger* result, because the
double-bump signal is ~2× larger in amplitude (1500 vs 300). Recovering a
bigger, structured field to the same absolute error is a better *relative*
recovery, and the visual evidence (cross-section, raw velocity) is far more
convincing. Big + large-scale signal ⇒ the filter nails it, and ESS stays
healthy under tempering with no first-step collapse — consistent with the
prediction that the anchored guess (~1.5σ from truth) is well within tempering's
reach.

## 3. Not run — Scenario A (far guess), the predicted failure mode

Deliberately left for later: a *far, unanchored* guess (center + ~1000·wave,
clamped to β=10, ≈5σ from truth). Predicted outcome: first-step degeneracy and
failed large-scale recovery, because no particle brackets the truth and the
process/jitter noise (10/step, 20/resample) cannot travel ~1000 in β over 20
steps — a clean demonstration of the particle filter's core limitation (it
cannot recover structure the prior ensemble does not already bracket). A
large-amplitude far guess is available in `test_double_bump_velocity.jl` behind
a clearly-marked, removable TEST TWEAK block (currently removed).

## Files

- `glacier-code/particleda/ice_experiment_dynamics.jl` — `double_bump` prior
  option (reversible; default pseudo_random_wave).
- `glacier-code/particleda/test_double_bump_velocity.jl` — static β→velocity
  test (β / raw speed / anomaly), 3 WAVI solves.
- `glacier-code/particleda/experiment_18_wavi_obs.jl` — ARGS 10–12 select the
  prior. Scenario B run used `double_bump 1500 1`.
- Outputs: `results/experiment_18_double_bump_test/` (static),
  `results/experiment_18_double_bump/` (filtered). Trajectory + checkpoint on
  the SSD at `experiment_18_double_bump/`.

## See also

- `26_wavi_observation_operator.md` — Exp 18 base setup; the "velocity looks
  geometry-dominated" caveat that motivated this test.
- `27_wavi_seed2_replicate.md` — the pseudo-random-wave replicate compared
  against here.
