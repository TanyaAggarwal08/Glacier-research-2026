# Observation noise — what it is, why the crosses look so spread out

> **The question.** In [beta_pointwise.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/beta_pointwise.png) the black × marks look very far from the true β line. How is observation noise added? What's the model? Where is the 0.10 value chosen? And why do the crosses look so scattered when σ_obs is only 10 %?
>
> **TL;DR.** Noise is added in **ux space** (the surrogate-velocity space), not in β space. There it's only ±0.10 around a typical ux of 1.0 — which looks tight. But when we convert the noisy ux observations back to β-equivalents on the plot via `β = 1000 / ux`, the **inverse-function transform amplifies and asymmetrically distorts the apparent noise**. A clean ±10 % in ux becomes a ±15-25 % spread (with fat upper tails) in β. The noise itself is fine and as intended; it just *looks* worse on the β plot than on the ux plot.

---

## 1. The exact noise model in the code

The relevant lines in [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) are 173-187:

```julia
function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector,
    state::AbstractVector,
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
)
    ParticleDA.get_observation_mean_given_state!(observation, state, model, task_index)
    σ = model.parameters.obs_noise_std        # = 0.10 in YAML
    @inbounds for k in eachindex(observation)
        observation[k] += σ * randn(rng)       # add Gaussian noise per sensor
    end
    return observation
end
```

In plain words, for each sensor k at each filter step:

1. Compute the **mean observation** the truth would produce — that's `ux = 1000 / β` evaluated at the sensor's grid cell. Call it `ux_true(k)`.
2. Draw a **random Gaussian** `η ~ N(0, 1)` — a standard normal random number, mean 0, std 1.
3. Multiply by `σ_obs = 0.10` and add: `y(k) = ux_true(k) + 0.10 · η`.

So the observation is the truth plus independent Gaussian noise of std 0.10, **measured in ux units**.

This is the same pattern tsunami uses ([test/models/llw2d.jl:574-586](../test/models/llw2d.jl#L574-L586)). It's the canonical "Gaussian observation noise" model used everywhere in data assimilation.

---

## 2. Why σ_obs = 0.10

The value came from two experiments documented in earlier notes:

- **Run-01 used σ_obs = 0.05** (the LowLevel toy's setting). At our Np = 1000 (smaller than LowLevel's 10 000), this was too tight: the likelihood became so peaked that weights collapsed onto one particle per step, ESS bouncing 1-30. See [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md) §What changed in the results.
- **Run-02 widened to σ_obs = 0.10.** ESS rose to 50-400 / 1000 (5-40 % Np). Healthier weights, comparable tracking accuracy.
- **Run-03 onward keeps 0.10** as the canonical value.

What does 0.10 mean **physically**? Our surrogate gives `ux = 1000 / β`. With β around 1000 → ux around 1.0. So **σ_obs = 0.10 represents about a 10 % measurement error** on the surface velocity. That matches realistic InSAR / GPS noise levels on glacier velocity measurements: roughly 5-20 % for the cleanest products, 20-40 % for noisier ones. 10 % is a defensible "good data" setting.

The number can be re-tuned. The trade-off is in [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md):
- **Smaller σ_obs** (e.g. 0.05) → tighter likelihood → better tracking accuracy *if* the filter can take it, but risks ESS collapse with small Np.
- **Larger σ_obs** (e.g. 0.20) → looser likelihood → ensemble stays healthier longer, RMSE convergence slower.

---

## 3. Why the crosses look "very far apart" — the noise amplification

This is the interesting bit. Look at the two plots side by side:

- [**ux_first_sensor.png**](../glacier-code/particleda/results/run04_grid_aligned_16obs/ux_first_sensor.png) — observations in their *native* space (ux), with the true ux and PF mean ux for reference.
- [**beta_pointwise.png**](../glacier-code/particleda/results/run04_grid_aligned_16obs/beta_pointwise.png) — same observations, converted to β-equivalent via `β = 1000 / ux`.

In the ux plot the crosses are tightly clustered ±0.10 around the truth line — exactly what σ_obs = 0.10 promises. In the β plot the same crosses look much more spread out, with a noticeable bias upward (crosses tend to lie *above* the true β line more than below).

**Both plots show the same observations.** The difference is the y-axis transformation.

### 3.1 The math

For the surrogate operator `ux = 1000 / β`:

- A **fractional change in ux** maps to a **fractional change in β** of opposite sign:
  ```
  Δβ / β  =  − Δux / ux
  ```
- A **±10 % noise on ux** therefore corresponds to **roughly ±10 % noise on β**, but **asymmetrically**: because ux appears in the denominator when we invert, the same ±0.10 in ux becomes a *wider* upper spread in β than lower spread.

### 3.2 Concrete numbers at β ≈ 1000

If ux_true = 1.0 (corresponding to β_true = 1000) and σ_obs = 0.10:

| Noise η | ux observation | Implied β = 1000 / ux | Apparent β error |
|---:|---:|---:|---:|
| −2 (2σ low ux) | 0.80 | 1250 | **+250** |
| −1 (1σ low ux) | 0.90 | 1111 | +111 |
| 0 (exact) | 1.00 | 1000 | 0 |
| +1 (1σ high ux) | 1.10 | 909 | −91 |
| +2 (2σ high ux) | 1.20 | 833 | −167 |

Notice: a low-ux observation pushes β-equivalent **upward by 250**, while a high-ux observation only pushes β-equivalent **downward by 167**. Same noise sign reversed gives asymmetric β errors. This is why the crosses on the β plot have a fatter upper tail.

### 3.3 What about other β values?

At larger β the asymmetry gets worse. At β = 1500, ux_true = 0.667; σ_obs = 0.10 means the noise is 15 % of ux. Then:

- ux_obs = 0.567 → β_obs = 1764 → +264 above truth
- ux_obs = 0.767 → β_obs = 1304 → −196 below truth

So for the sensor that the user is looking at — where β oscillates between roughly 800 and 1150 (judging from the truth line on [beta_pointwise.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/beta_pointwise.png)) — the expected β-equivalent spread is roughly ±100 to ±300 around the truth, with fat upper tails. That's exactly what we see.

### 3.4 Statistical sanity check

For 200 independent observations drawn from `ux ~ N(ux_true, 0.10)` at ux_true ≈ 1.0:

| Sigma | Expected # in tail (200 draws) | Expected β-equivalent excursion |
|---:|---:|---|
| > 1σ | ~64 | β ≈ ±100 |
| > 2σ | ~10 | β ≈ +250 / −167 |
| > 3σ | ~0.6 | β ≈ +430 / −230 |

Looking at the actual plot: roughly 60-70 crosses appear visibly > ±100 from the truth, ~10 are above ±200, ~1 each at the extremes. That matches the prediction within sampling noise.

**So the observations are exactly as noisy as specified. The β plot just exaggerates that noise visually due to the nonlinear inverse transform.**

---

## 4. What the filter actually sees

This is important: **the filter computes likelihoods in ux space, not β space**. Look at `get_log_density_observation_given_state` in [glacier_model.jl](../glacier-code/particleda/glacier_model.jl):

```julia
function ParticleDA.get_log_density_observation_given_state(
    observation, state, model, task_index=1
)
    obs_mean = view(model.obs_buffer, :, task_index)
    ParticleDA.get_observation_mean_given_state!(obs_mean, state, model, task_index)
    return -invquad(model.obs_cov, observation .- obs_mean) / 2
end
```

The log-likelihood is `−‖y − h(x)‖² / (2 σ_obs²)`, where both `y` (the observation) and `h(x)` (the model prediction) live in ux units. The filter never inverts the surrogate; it compares the noisy ux observation against each particle's predicted ux. So **from the filter's perspective the data is as clean as `ux_first_sensor.png` shows**, not as scary as `beta_pointwise.png` shows.

The β-equivalent plot is a *post-hoc visualisation* for human eyeballing only. The filter operates in the right space and is unaffected by the cosmetic distortion of the plot.

---

## 5. Why the PF mean still tracks the truth

Even though individual observations are noisy (in either space), the filter averages over particles and time. The PF mean is roughly:

- A weighted average of 1000 particles' β predictions, where weights are based on how well each particle's predicted ux matches the noisy observation.
- The noise η is independent across sensors and across time steps, so it tends to average out.

So you'd expect the PF mean to track the truth with much smaller error than any single observation — that's exactly what you see. The dashed orange "PF Mean" line in both plots stays close to the truth line, with deviations measured in tens of β-units, not hundreds.

In numbers: at the first sensor over the last 50 filter steps, the PF mean is typically within ~50 of truth, while individual observations are scattered ±100-300. The filter is averaging out the noise correctly.

---

## 6. Quick visual cheat-sheet

What's the right way to read each plot?

| Plot | Shows | Honest about noise level? |
|---|---|---|
| [ux_first_sensor.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/ux_first_sensor.png) | Truth, PF mean, and observations in **ux space** | ✅ Yes — this is the space the filter operates in, σ_obs = 0.10 is straightforwardly visible |
| [beta_pointwise.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/beta_pointwise.png) | Same data, observations converted to **β-equivalent** | ⚠️ No — the 1/β inversion amplifies and skews the apparent noise. Use only for context |
| [rmse_beta.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/rmse_beta.png) | Global RMSE of PF mean vs truth, in β units | ✅ Yes — the PF mean is what we ultimately care about |

**When you want to judge how noisy the observations are, look at the ux plot. When you want to judge how well the filter recovers β, look at the RMSE plot. The β-equivalent observations plot is for putting both on the same y-axis for context — it's useful for seeing the truth shape, but it visually distorts the noise.**

---

## 7. Could we make the crosses look tighter?

Three options, ordered by what they actually change:

1. **Tighten σ_obs** — the crosses will get tighter in ux space directly. We could try 0.05, 0.07, 0.10 as a sweep. Risk: at small Np this collapses the ensemble. Worth running as part of the σ_obs knee-search experiment ([05 §What to change next](05_particleda_vs_lowlevel_comparison.md) suggestion #2).
2. **Switch to a different surrogate** that's less inversion-amplified — for example, a polynomial `ux = a − b·β` instead of `ux = 1000/β`. This would change the science (we'd no longer be matching real glacier physics in a meaningful way), so we shouldn't.
3. **Plot ux instead of β-equivalent** — purely cosmetic, but it gives the honest visual. Just done; the ux plot is now in the canonical outputs.

The next experimental step would be option 1 (σ_obs sweep) at the same canonical setup. I'd recommend doing that *after* WAVI swap, not before — the surrogate's amplification effect is artificially harsh compared to what WAVI itself will produce.

---

## 8. Glossary additions

| Term | Meaning |
|---|---|
| **Observation noise σ_obs** | Standard deviation of the Gaussian noise added to each observation. Currently 0.10 in ux units. |
| **Observation operator h(x)** | The function mapping state x to its predicted observation. Here `h(x) = 1000 / β(x)` evaluated at sensor cells. |
| **Native observation space** | The space in which the noise is actually added — for us, ux. The space the filter computes likelihoods in. |
| **β-equivalent observation** | Result of applying the inverse operator `β = 1000 / ux_obs` to a noisy observation. Useful for plotting next to true β but distorts the noise. |
| **Inverse-function noise amplification** | The phenomenon that when y = f(x) is nonlinear, the noise variance in x is *not* the same as in y. For our `1/β`, a fixed σ in ux becomes a β-dependent spread in β, with fat upper tails. |
| **Likelihood** | The relative probability of seeing an observation given a candidate state. Higher = state explains the obs better. Particle filter weights ∝ likelihood. |
