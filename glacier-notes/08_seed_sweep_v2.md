# Seed sweep v2 — same experiment, bug-fixed dynamics

> **What this note answers.** After we fixed the `time_step` / `n_integration_step` bug in [07_observation_cadence.md](07_observation_cadence.md), we should re-do the 5-seed robustness check ([06_seed_sweep_robustness.md](06_seed_sweep_robustness.md)) on the new code. This note compares v1 (buggy) vs v2 (fixed) results so we know whether the bug ever mattered, and which numbers to trust going forward.
>
> **Files:** [run_seed_sweep.jl](../glacier-code/particleda/run_seed_sweep.jl), [plot_seed_sweep.jl](../glacier-code/particleda/plot_seed_sweep.jl), per-seed outputs in [results/seed_sweep_v2/](../glacier-code/particleda/results/seed_sweep_v2/). Old (v1) outputs preserved in [results/seed_sweep_run03/](../glacier-code/particleda/results/seed_sweep_run03/).

---

## 1. Why redo it

The earlier seed sweep ran on code that had a quiet bug: `update_state_deterministic!` ignored the YAML's `time_step` and computed its own internal dt from a stability formula. So when the YAML said `time_step: 1.0`, the model actually used `dt ≈ 457 s`.

That sweep "worked" — ESS healthy, RMSE convergent — but only because 457 s happens to be a reasonable physical scale for our toy. With the fix, the YAML's `time_step` actually controls dt. If we re-run the sweep with `time_step = 400 s` (just below the CFL ceiling) we should get behaviour close to the buggy v1 — same physical scale, just now reachable from the config file instead of by accident.

We need to confirm that.

---

## 2. What changed in the config

Everything except `time_step` is identical to v1.

| Parameter | v1 sweep | v2 sweep | Notes |
|---|---:|---:|---|
| `nprt` | 1000 | 1000 | particle count |
| `init_std_theta` | 0.30 | 0.30 | initial spread (wide) |
| `process_std_theta` | 0.007 | 0.007 | per-step process noise |
| `obs_noise_std` | 0.10 | 0.10 | observation noise std |
| `sensor_stride` | 100 | 100 | gives 16 sensors |
| `n_integration_step` | 1 | 1 | one internal sub-step per filter step |
| `time_step` | 1.0 (ignored — actual dt was 457) | **400.0** (actual dt) | the only meaningful change |
| `n_time_step` | 200 | 200 | number of filter steps |
| Seeds tested | 42, 7, 13, 99, 2024 | 42, 7, 13, 99, 2024 | identical |

So in physical-time terms, v1 covered roughly `200 × 457 ≈ 91,400 s` of model time per run; v2 covers `200 × 400 = 80,000 s`. About 12 % less total time, but the per-step dynamics are very close.

---

## 3. Side-by-side numbers

### Per-seed table (v2)

| seed | mean ESS | min ESS | frac > 0.5·Np | RMSE @ t=1 | RMSE final |
|---:|---:|---:|---:|---:|---:|
| 42 | 536.7 | 1.9 | 0.630 | 348.9 | 123.5 |
| 7 | 523.1 | 1.0 | 0.645 | 387.9 | 122.3 |
| 13 | 555.8 | 2.4 | 0.675 | 337.6 | 115.7 |
| 99 | 592.1 | 2.4 | 0.805 | 331.2 | 98.7 |
| 2024 | 589.2 | 1.4 | 0.815 | 356.8 | 114.7 |
| **mean ± std** | **559 ± 32** | — | **0.71 ± 0.09** | **352 ± 22** | **115 ± 9** |

### v1 vs v2 head-to-head

| Metric | v1 (buggy) | v2 (fixed) | Δ as % of v1 |
|---|---:|---:|---:|
| Mean ESS | 562 ± 26 | 559 ± 32 | −0.5 % |
| Frac > 0.5·Np | 0.70 ± 0.07 | 0.71 ± 0.09 | +1.4 % |
| RMSE final | 114 ± 9 | 115 ± 9 | +0.9 % |
| RMSE at t=1 | 333 ± 9.5 | 352 ± 22 | +5.7 % |

All differences are inside the seed-noise floor we measured earlier (≈ 5–15 % per metric). **The bug fix changes nothing observable at this physical scale.** That's the result we hoped for.

The one slightly larger gap is the initial RMSE (333 → 352, ~6 %). That makes sense: the truth and initial particles are both drawn from the prior, and the random draws happen to give a slightly less-aligned starting state in v2 (because the RNG path is now different — fixing the bug changed at least one call to `rand`, shifting the random stream downstream). It's bookkeeping noise, not a real effect.

---

## 4. What the envelope plots show

- **[ess_envelope.png](../glacier-code/particleda/results/seed_sweep_v2/ess_envelope.png)** — Median ESS rises from ~400 to a steady-state band around 580 after step ~30. The ribbon (min/max across seeds) is narrow above step 50, indicating tight agreement. The dashed line at 500 (0.5·Np) sits just below the median, so most seeds clear it most of the time. Visually indistinguishable from v1.

- **[rmse_envelope.png](../glacier-code/particleda/results/seed_sweep_v2/rmse_envelope.png)** — Median RMSE drops from ~350 to ~115 with a narrow ribbon that tightens over time. The convergence shape is the textbook "filter learning" curve we saw in v1.

- **[ess_all_seeds.png](../glacier-code/particleda/results/seed_sweep_v2/ess_all_seeds.png)** and **[rmse_all_seeds.png](../glacier-code/particleda/results/seed_sweep_v2/rmse_all_seeds.png)** — All five seed lines overlap heavily. No seed is dramatically different from the others.

- **[maxweight_all_seeds.png](../glacier-code/particleda/results/seed_sweep_v2/maxweight_all_seeds.png)** — All five spike to ~1.0 in the first 1–2 steps (the inevitable wide-prior moment when one particle wins big), then collapse to under 0.05 within 5 steps and stay there.

---

## 5. What this means

1. **v2 is the new canonical baseline.** Any future experiment should be compared to v2 numbers, not v1. The main [glacier.yaml](../glacier-code/particleda/glacier.yaml) is now updated to `time_step: 400.0` so running [run_glacier_pda.jl](../glacier-code/particleda/run_glacier_pda.jl) gives bug-fixed behaviour by default.

2. **Run-03's reported behaviour is intact.** The seed-sweep means (562 → 559) and final RMSEs (114 → 115) move by under 1 %. So when the earlier notes claim "the filter converges, ESS sits at 56 % of Np, frac > 0.5·Np is around 0.70", those statements remain true on the fixed code.

3. **The seed-noise floor is still ≈ 5–15 %.** Same as v1. Any future single-seed comparison (σ_obs sweep, surrogate→WAVI swap, different cadence) needs to beat this margin to count as a real effect.

4. **One specific thing to note for downstream comparisons.** v2's RMSE-at-t=1 is 352 vs v1's 333 (a 6 % shift, near the noise floor). This is a *starting-point* artefact from the RNG stream shifting after the bug fix, not a dynamics change. When comparing convergence curves across v1 and v2 runs, normalise on the *final* RMSE rather than the *initial* — the initial drift is bookkeeping.

---

## 6. Quick answer to "what `n_integration_step` are we using?"

For all v2 / canonical runs from here on:

- **`time_step = 400 s`** — that's one filter step.
- **`n_integration_step = 1`** — one internal physics sub-step per filter step.
- **internal dt = 400 / 1 = 400 s** — well below the CFL ceiling of ~457 s.

So the filter is updating **every 400 model-seconds**, not every 1 second or every 2 seconds. The "tsunami-style every-2-seconds" idea doesn't translate to our toy — at 2 s the dynamics barely move ([07_observation_cadence.md](07_observation_cadence.md) §2 explains why). 400 s is what gives the physics enough room to actually advect β.

If we ever want **half** as many observations over the same model time, we set `time_step = 800, n_integration_step = 2` (still internal dt = 400). That's the run-05 setup; we already have data for it in [results/obs_cadence/](../glacier-code/particleda/results/obs_cadence/).

---

## 7. Glossary (carried over from [07_observation_cadence.md](07_observation_cadence.md))

| Term | Meaning |
|---|---|
| **dt** | Size of one internal physics step in time. |
| **`time_step`** | (YAML) Model-time covered by one filter step. |
| **`n_integration_step`** | (YAML) Number of internal sub-steps per filter step. dt = `time_step / n_integration_step`. |
| **CFL ceiling** | Largest dt the numerical scheme stays stable for. Here ≈ 0.2 × dx / max_speed ≈ 457 s. |
| **ESS** | Effective Sample Size — how many *independent* particles the weighted ensemble is "worth". 1 = degenerate, Np = healthy. |
| **frac > 0.5·Np** | Fraction of filter steps where ESS clears the 50 %-of-Np rule of thumb. Higher = healthier. |
| **RMSE final** | Root-mean-square error of estimated β vs true β, averaged over the last 10 steps. Smaller = better tracking. |
| **Seed-noise floor** | The natural spread in any metric just from drawing different random truths and particles. We measured it at ~5–15 % per metric across 5 seeds. |
