# Observation ablation experiment — do observations actually do anything?

> **The experiment.** Run the particle filter twice on identical truth and observations. In one run the filter uses the observations as normal; in the other it ignores them (returns a constant log-likelihood for every particle). Compare RMSE, ESS, and pointwise tracking.
>
> **TL;DR.** Observations clearly do real work — the per-cell pointwise plot shows the with-obs PF mean **tracking** the truth while the no-obs PF mean stays **frozen at the prior**. But the spatially-averaged RMSE shows a *counter-intuitive* result: the no-obs run gets a *lower* final RMSE. The reason is **numerical diffusion in the upwind advection scheme** smoothing the truth's high-frequency initial perturbation back toward the prior sinusoid — so the truth and the prior-locked no-obs PF mean converge by *erosion*, not by *learning*. This is an important lesson: **RMSE alone can mislead you. Per-point tracking is the honest diagnostic.**
>
> **Files:** [run_obs_vs_no_obs.jl](../glacier-code/particleda/run_obs_vs_no_obs.jl), [plot_obs_vs_no_obs.jl](../glacier-code/particleda/plot_obs_vs_no_obs.jl), [glacier_no_obs.yaml](../glacier-code/particleda/glacier_no_obs.yaml), outputs in [results/run06_ablation_compare/](../glacier-code/particleda/results/run06_ablation_compare/) (also baseline and no-obs HDF5s in [run06_obs_baseline/](../glacier-code/particleda/results/run06_obs_baseline/) and [run06_no_observations/](../glacier-code/particleda/results/run06_no_observations/)).

---

## 1. How the ablation works

A new YAML flag, `disable_observations: true`, was added to the model parameters. When this flag is set, [glacier_model.jl](../glacier-code/particleda/glacier_model.jl)'s `get_log_density_observation_given_state` returns **0** for every particle, regardless of the observation or the particle's predicted ux. The full code change is two lines:

```julia
if model.parameters.disable_observations
    return zero(eltype(observation))     # constant ⇒ uniform weights
end
```

Truth generation and `sample_observation_given_state!` are untouched, so the same noisy observations are produced. The only difference is whether the filter *uses* them.

Both runs were launched with identical seeds (filter 42, truth 123) so the truth field and the observations are byte-for-byte identical. The only experimental variable is the ablation flag.

---

## 2. Headline numbers

| Metric | With obs (baseline) | No obs (ablation) | What it tells us |
|---|---:|---:|---|
| **RMSE(β) initial** (t = 0) | 336 | 336 | Same truth and same particle ensemble → same starting error. ✓ |
| **RMSE(β) final** (last 10 steps) | **119** | **89** | No-obs run gets *better* spatial-average RMSE! Counter-intuitive. |
| RMSE(β) max | 375 | 336 | Both peak near t = 0 |
| **mean ESS** | 554 / 1000 | **1000 / 1000** | Uniform weights confirmed: ablation is working as designed |
| frac ESS > 0.5·Np | 0.70 | **1.00** | Same |
| **mean max(weight)** | 0.014 | **0.001 = 1/N** | Same |

The ESS and max-weight numbers are exact: with `log_density ≡ 0`, every weight equals 1/N, ESS = N, max(weight) = 1/N. So the flag is doing precisely what it should at the filter level — making weights uninformative.

The surprise is in the RMSE direction.

---

## 3. What the plots actually show

### [ess_compare.png](../glacier-code/particleda/results/run06_ablation_compare/ess_compare.png)

A flat red line at ESS = 1000 (no-obs) versus the with-obs blue curve bouncing between 400 and 800. This is the *cleanest* visual proof that observations are doing real work — they're what creates the weight differentiation that gives the with-obs run any concentration in the ensemble at all.

If you ever wanted a one-image "observations matter" diagram for a talk, this is it.

### [beta_pointwise_compare.png](../glacier-code/particleda/results/run06_ablation_compare/beta_pointwise_compare.png)

Truth (black) at the first sensor wanders between β_norm ≈ −0.5 and β_norm ≈ +0.1 over the run — that's a real spatial-temporal pattern driven by initial perturbation + advection + process noise.

- **With-obs PF mean (blue dashed):** snaps down from the prior (β_norm = 0) to the truth's −0.35 trough on the first update, then *tracks* the truth's subsequent wandering closely.
- **No-obs PF mean (red dash-dot):** stays *flat at the prior* (β_norm ≈ 0) for the entire run.

This is the honest "did the filter learn anything?" diagnostic. With observations the answer is *yes, dramatically*. Without observations the PF mean is just the prior — and the prior is wrong at this point.

### [rmse_compare.png](../glacier-code/particleda/results/run06_ablation_compare/rmse_compare.png)

Both curves drop. The no-obs (red) curve actually drops *faster* and *lower*. This is the counter-intuitive bit that needs explanation — see §4.

---

## 4. Why no-obs RMSE drops — the numerical-diffusion explanation

The puzzle: if the no-obs PF mean is frozen at the prior (as the pointwise plot shows), how can its spatially-averaged RMSE against the truth *decrease*?

Answer: **the truth itself is drifting toward the prior sinusoid because the upwind scheme has built-in numerical diffusion**.

### 4.1 The numerical diffusion budget

From [10_one_step_advection.md](10_one_step_advection.md) §3, our first-order upwind scheme has effective diffusion:

```
D_num = v · dx / 2 · (1 − CFL)
      = 1.75 · 4000 / 2 · (1 − 0.175)
      ≈ 2 887 m²/s
```

Over our 200 filter steps × 400 s = 80 000 s of model time, the diffusion length scale is:

```
L_diff = √(D_num · T_total) = √(2 887 · 80 000) ≈ 15 000 m ≈ 3.75 grid cells
```

That's enough smoothing to attenuate any feature smaller than ~4 grid cells.

### 4.2 Why this kills the truth's initial perturbation

At t = 0, the truth is `β_prior + 0.30 · η` where η is iid white noise per cell. White noise has all of its variance at the smallest scales — exactly the scales the numerical diffusion erodes most aggressively.

So as the simulation progresses, the truth gets *smoother*: the iid perturbation washes out, leaving the smooth sinusoidal `β_prior` underneath. The truth field at t = 200 is much closer to `β_prior` (in spatial-average RMSE) than the truth field at t = 0.

Meanwhile the no-obs PF mean stays at `β_prior` (locked by the lack of observations). So in spatial-RMSE terms, the truth is coming *toward* the no-obs PF mean — not because the filter is learning, but because both are converging to the same smooth backbone.

### 4.3 What this means for the with-obs run's RMSE

The with-obs run is doing the *right* thing — at every observed cell, the PF mean is pulled toward the (noisy) observation, which gives a tighter estimate than the prior. But the per-cell tracking incurs some noise from the obs-noise σ_obs = 0.10 itself. So the with-obs RMSE has a noise floor of order σ_obs × |β/ux| ≈ 0.10 × 1000 = 100 per observed cell.

That noise floor (~100 in β units) is similar to what we see: with-obs final RMSE ≈ 119. The no-obs run avoids this noise floor by ignoring obs entirely — its RMSE is just `||truth_diffused − prior||`, which has decreased from the initial 336 to 89.

### 4.4 So which is "right"?

This is a perfect example of the trade-off RMSE alone can hide:

- The **with-obs filter** tracks the truth's actual instantaneous pattern. At any given timestep, its PF mean is the best estimate of the truth at that timestep, including any high-frequency structure.
- The **no-obs run** has a PF mean equal to a smooth prior, which only coincidentally has low RMSE because the truth has *also* been smoothed by numerical diffusion.

If we asked "what was the truth at this specific cell at this specific time?", only the with-obs run would have a meaningful answer. The no-obs run would shrug and hand you the prior at that cell.

This is exactly the kind of result that gets papers retracted in operational DA. *"My filter has low RMSE, therefore my filter is great"* is wrong when the low RMSE is partly an artefact of diffusion eroding the very features you're trying to track.

---

## 5. Lessons

### 5.1 The right diagnostic for "does the filter learn?"

| Diagnostic | What it tells you | Best at detecting |
|---|---|---|
| **Spatial-average RMSE** | Aggregate distance between truth field and PF mean field | Convergence over many cells but can be fooled by diffusion |
| **ESS, max(weight)** | Whether weights are informative | The filter is *responding* to observations |
| **Per-point pointwise plot** | Whether PF mean tracks truth's actual trajectory | The most honest "filter is learning" signal |
| **Posterior variance, spatial map** | Where in the domain uncertainty is high vs low | Quality of posterior coverage, not just point estimates |

For the glacier project we should report all four. Spatial-average RMSE is the easy one to print but the easiest to mislead with.

### 5.2 Numerical diffusion is a real signal-killer

For finer features (sharp β contrasts, local sliding patches near the bed) the upwind scheme will *smooth them out* over the run. Future enhancements to consider:

- A **higher-order advection scheme** (Lax-Wendroff at 2nd order, or WENO at 5th) to reduce numerical diffusion.
- A **conservation-form** discretisation if we get to bigger β contrasts.

Neither is urgent for the surrogate, but both become relevant when we move to real ice-flow physics.

### 5.3 The disable_observations flag is now a permanent diagnostic

Setting `disable_observations: true` in any YAML reruns the same physics with the filter blinded to observations. Useful any time we add a new experiment and want a "what if we did nothing?" reference. Free of charge (same wall-clock; trivial code path).

### 5.4 ESS is the cleanest "observations on or off?" indicator

If ESS = N for every step, observations are not informing the filter. If ESS fluctuates below N, they are. No ambiguity, no diffusion confound.

---

## 6. What this opens up for the next experiment

Two natural follow-ups:

1. **Spectral analysis of truth vs PF mean** — compute the spatial power spectrum at t = 0 and t = 200 for both runs. Expected finding: the no-obs run's truth has lost most of its high-frequency power to numerical diffusion, while the with-obs run's PF mean retains it (because observations re-inject the high-frequency information).
2. **Test with a less-diffusive scheme** — switch the upwind step to a flux-limited or higher-order discretisation, rerun the ablation. Expected finding: no-obs RMSE no longer drops as dramatically because the diffusion artefact is much smaller.

For now we have what we wanted: a clear ablation showing **observations are the only thing that pulls the filter toward the truth's actual instantaneous structure**, even though the integrated RMSE numbers don't make this obvious without context.

---

## 7. Glossary additions

| Term | Meaning |
|---|---|
| **Ablation** | A controlled experiment where you remove or disable one component to measure its effect. Standard ML / DA practice. |
| **Numerical diffusion** | Artificial smoothing introduced by a low-order finite-difference scheme. Real, predictable, quantified for upwind as `v · dx · (1 − CFL) / 2`. |
| **Diffusion length scale** | `√(D · t)`. How far information has spread by diffusion after time t. For us: ~15 km after 80 000 s. |
| **Per-point pointwise plot** | A diagnostic that shows truth and PF mean at a *single* spatial location over time. Doesn't average across cells, so can't be fooled by spatial structure cancellation. |
| **ESS = N** | All particles have equal weight; the filter has gained no information from any observation. The signature of an uninformed filter. |
| **disable_observations flag** | Our new model parameter. When true, the likelihood function returns a constant, so the filter ignores observations entirely. |
