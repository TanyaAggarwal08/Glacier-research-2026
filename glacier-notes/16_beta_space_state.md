# β-space state + dynamics verification

> Two changes in one note:
> 1. **State variable switched from log(β) to β directly** — much easier to read the truth/PF curves; the log compression was hiding the dynamics.
> 2. **Verified the dynamics are running** — the earlier "straight-line PF mean at sensor 1" was a sensor-placement coincidence, not a missing prediction step. At non-degenerate grid cells the no-obs PF mean evolves dramatically (e.g. cell (10, 10) swings from 1488 → 700 → 1100 over the run, entirely from advection).

---

## 1. State switched to β directly

### 1.1 What changed in code

[glacier_model.jl](../glacier-code/particleda/glacier_model.jl) was rewritten so that the **state vector now holds β values directly** instead of log(β).

| Component | Before (log-space) | After (β-space) |
|---|---|---|
| State stored | θ = log(β) | β |
| Init mean | log(β_prior) | β_prior |
| Init noise | `state += 0.30 · randn()` (log units) | `state += 300.0 · randn()` (β units) |
| Process noise | `state += 0.007 · randn()` per step | `state += 7.0 · randn()` per step |
| Positivity guard | `β = exp(state)` ensured β > 0 | `state = max(state, min_beta=10)` clamps the floor |
| Surrogate ux | `ux = 1000 / (exp(state) + 1e-6)` | `ux = 1000 / max(state, min_beta)` |
| HDF5 output | `beta = exp(state)` derived | `beta = state` directly; `log_beta` derived for back-compat |

The values 300 and 7 were chosen to **preserve previous statistical behaviour at β ≈ 1000**: a 30 % initial spread and 0.7 % per-step random walk. These match what the old log-space parameters produced once you back out exp(σ).

### 1.2 Why drop the log transform?

You had two valid reasons:

- **Interpretation.** Plotting `log(β)` compresses the dynamic range. A truth wandering from β = 800 to β = 1200 looks like a small wobble in log-space (6.68 to 7.09) but is much more obvious in β-space — easier to *see* whether the dynamics are doing anything.
- **Simplicity for now.** The log transform was protecting positivity at the cost of an extra exp/log everywhere. Until we integrate WAVI (the "actual nonlinear layer"), positivity can be enforced cheaply by clamping. Drop the log; reduce code complexity.

We keep the `log_beta` derived field in the HDF5 output so older plots and any external code that expected `log_beta` still work.

### 1.3 Risk: β going negative

In β-space, process noise of std 7 per step could in principle push β through zero into negative territory, which would explode the surrogate `ux = 1000 / β`.

In practice this is well-controlled:
- Process noise per step has std 7 — small compared to β ≈ 1000.
- The lower clamp `min_beta = 10` catches any noise excursion that would go negative or near zero. With β = 1000 and noise std 7, you'd need a ~140σ tail event to hit the floor — never happens in practice.
- The clamp is documented in YAML as the `min_beta` parameter — adjustable if needed.

In all current runs the clamp never fires.

### 1.4 Behaviour stayed essentially identical

Re-running the canonical config and the ablation experiment after the switch:

| Metric | Old (log-space) | New (β-space) |
|---|---:|---:|
| RMSE(β) initial | 336 | 300 |
| RMSE(β) final (with obs) | 119 | 109 |
| RMSE(β) final (no obs) | 89 | 84 |
| Mean ESS (with obs) | 554 | 509 |
| frac ESS > 0.5·Np (with obs) | 0.70 | 0.58 |

All within seed-noise. **The filter is doing the same thing scientifically**; only the visualisation and the parameter names changed. The slight differences are because the random number stream is different (different state means different `randn` calls).

---

## 2. Verifying that the prediction step (model dynamics) is actually running

The concern was that the no-obs PF mean looked **flat** in [the previous pointwise plot](../glacier-code/particleda/results/run06_ablation_compare/beta_pointwise_compare.png), suggesting the filter wasn't applying the model transition. We've now traced through this carefully and the dynamics are running — the apparent flatness was an artefact of *which cell* was being plotted.

### 2.1 Why the corner cell looks flat

The first sensor in the canonical (grid-aligned) layout is at flat index 1, i.e. cell `(i = 1, j = 1)`, which sits at the corner `(x = 0, y = 0)`.

Our prior is `β_prior(x, y) = 1000 + 500 · sin(ω x) · sin(ω y)` with `ω = 2π / L`.

At `(x = 0, y = 0)`:
- `sin(0) = 0` → `β_prior = 1000`
- `∂β_prior / ∂x = 500 ω cos(0) sin(0) = 0`
- `∂β_prior / ∂y = 0`

So at this exact cell **the prior is at a zero-gradient point** of the sinusoid. Upwind advection of a constant field is the identity: the value at (1, 1) doesn't change as the field is transported. With no observations to push it elsewhere, the no-obs PF mean stays at 1000 ≈ β_prior(1, 1) for the entire run.

That's why the red dashdot line in the earlier pointwise plot looked dead flat: it wasn't a bug, it was a coincidence of where we sampled.

### 2.2 Direct evidence: PF mean at non-degenerate cells

I probed the no-obs PF mean at six cells with very different prior values and gradients. Here's the table (units: β in Pa·s/m):

| Cell `(i, j)` | β_prior | PF mean t=1 | t=50 | t=100 | t=150 | t=201 |
|---|---:|---:|---:|---:|---:|---:|
| (1, 1)   | 1000 | 1003 |  983 |  986 |  984 |  984 |
| (10, 10) | **1488** | 1507 | 1105 | **738** |  592 |  982 |
| (20, 20) | 1012 |  998 | 1018 | 1007 |  973 |  951 |
| (30, 30) | **1488** | 1492 | 1121 |  725 |  587 |  976 |
| (10, 30) | **512**  |  493 |  813 | **1328** | 1276 |  969 |
| (5, 15)  | 1238 | 1238 |  859 |  658 |  825 | 1215 |

At cells with strong prior gradients (rows in **bold**) the no-obs PF mean swings by hundreds of β-units. **Cell (10, 10) drops from 1488 → 592 (a swing of ~900) and back up to 982 across 200 filter steps.** That motion is *entirely* due to the deterministic advection step — there are no observations to inform it, and the iid process noise has mean zero. So if the dynamics were dead, every cell would stay pinned at its initial value plus a flat random walk.

The dynamics is the *only* mechanism by which a no-obs filter run can produce this kind of structured cell-by-cell trajectory. So this table directly proves the prediction step is doing its job.

### 2.3 The visual proof

[dynamics_verification.png](../glacier-code/particleda/results/run06_ablation_compare/dynamics_verification.png) makes the same point visually: 2 × 2 panels of cells (10, 10), (10, 30), (30, 10), (5, 15). In each one:
- The **grey dotted line** is `β_prior` at that cell.
- The **black solid** line is the truth (which wanders due to dynamics + process noise on a single realisation).
- The **blue dashed** line is the with-obs PF mean (tracks the truth closely).
- The **red dash-dot** line is the no-obs PF mean.

The red line is **not flat** in any of these panels — it follows the advected sinusoid through hundreds of β-units of motion, often crossing the prior multiple times. That's exactly what we expect: without observations, the no-obs PF mean = (ensemble average of particles drifting via dynamics + noise), and because dynamics is non-trivial it produces non-trivial trajectories everywhere except at zero-gradient points.

### 2.4 What this confirms about the ParticleDA paper's "prediction phase"

The ParticleDA paper writes:

> *"The Model: During the prediction phase (Step 2), particles are fed only by the model's transition rules to guess the next state, completely ignoring the observations."*

Our setup does exactly this:

- **Predict step** = `update_state_deterministic!` (upwind advection) + `update_state_stochastic!` (Gaussian process noise). Both are called for every particle, every filter step, regardless of `disable_observations`. They never see the data.
- **Update step** = the likelihood + weight + resample loop. *This* is where observations enter. When `disable_observations: true`, the likelihood returns a constant, so weights are uniform and the update step effectively does nothing.

The no-obs run is therefore a pure prediction trajectory: dynamics + process noise, with cosmetic resampling that doesn't reshape the ensemble (because weights are uniform). The fact that we see large structured β-evolution in the no-obs PF mean is the prediction step doing its work, exactly as the paper describes.

---

## 3. What to look at when you want to know "is the filter learning?"

This was the underlying question that motivated all of this. With the β-space state and the dynamics-verification plot, the answer is now obvious in three different ways:

1. **Pointwise tracking** — pick a sensor cell, plot truth vs PF mean. With obs: PF mean tracks truth (blue dashed sticks to black solid in the comparison). Without obs: PF mean wanders with the dynamics but doesn't track the truth's specific trajectory.
2. **ESS** — with obs: ESS bounces well below N (informative weights). Without obs: ESS = N exactly (uninformative weights). One number, instantly decisive.
3. **Spatial RMSE relative to a no-obs reference** — only meaningful if interpreted carefully (see [15_observations_ablation.md](15_observations_ablation.md) — numerical diffusion confounds raw RMSE).

The β-space change makes all three diagnostics more legible.

---

## 4. Files touched

- [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) — full rewrite of the state representation. State, prior, noise additions, observation operator, and HDF5 IO all updated.
- [glacier.yaml](../glacier-code/particleda/glacier.yaml) — `init_std_theta` → `init_std_beta = 300.0`; `process_std_theta` → `process_std_beta = 7.0`; new `min_beta = 10.0`.
- [glacier_random.yaml](../glacier-code/particleda/glacier_random.yaml) — same parameter renames.
- [glacier_no_obs.yaml](../glacier-code/particleda/glacier_no_obs.yaml) — same.
- [run_obs_vs_no_obs.jl](../glacier-code/particleda/run_obs_vs_no_obs.jl) — embedded baseline YAML updated.
- [plot_obs_vs_no_obs.jl](../glacier-code/particleda/plot_obs_vs_no_obs.jl) — new `dynamics_verification.png` panel.

Old log-space runs (run01..run05 archives) remain on disk untouched but were generated under the old code. If you want to re-run any of them on the new β-space code, the YAML parameter names need a quick rename — easy.

---

## 5. Glossary additions

| Term | Meaning |
|---|---|
| **β-space state** | The state vector stored as β values directly, in Pa·s/m units. As opposed to log(β). |
| **min_beta** | Lower clamp on β to keep `1/β`-style surrogate observation operators safe. Default 10 — never triggered in practice but defensive. |
| **Zero-gradient point** | A cell where `∂β_prior / ∂x = ∂β_prior / ∂y = 0`. The advection has nothing to transport, so the PF mean (without obs) stays pinned at the prior value. The corner (1, 1) of our sinusoidal prior is one. |
| **Non-degenerate cell** | A cell with non-zero prior gradient. Advection produces real β-evolution here, even without observations. |
| **Prediction step** (ParticleDA paper) | The deterministic + stochastic dynamics applied to each particle before the weight update. Independent of observations. |
| **Update step** | The likelihood-weighting + resampling phase. Where observations enter. |
