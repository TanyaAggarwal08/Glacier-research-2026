# Smooth noise + hourly cadence + sensor/unsensored RMSE split

Three changes in one pass, motivated by the recommendations in [§8.3 of the full audit](17_full_audit.md):

1. **Smooth (spatially correlated) noise** replaces iid noise on both the initial perturbation and the per-step process noise.
2. **Hourly filter cadence** with internal substepping: filter updates every 3600 s, but the advection PDE keeps integrating in between with `n_integration_step = 10` (internal dt = 360 s, comfortably under the CFL ceiling).
3. **Sensor-cell vs unsensored-cell RMSE** plotted and reported separately — the diagnostic the audit recommended.

---

## 1. What changed

### 1.1 Smooth noise (squared-exponential covariance)

[glacier_model.jl](../glacier-code/particleda/glacier_model.jl) now builds a Cholesky factor `L` of a 2D squared-exponential kernel at init:

```
K[i, j] = exp( −‖xᵢ − xⱼ‖² / (2 ℓ²) )
L Lᵀ = K + 1e-8·I   (jitter for numerical PD)
```

Where `ℓ = noise_length_scale` (default 30 km — about 3/16 of the 160 km domain).

To draw a smooth Gaussian field with marginal std σ:

```
z ∼ N(0, I_n)
δβ = σ · L · z       # nearby cells move together, far ones decouple
```

We use this everywhere the model previously drew iid `σ·randn()`:
- `sample_initial_state!` — initial spread is now spatially coherent (entire regions push up or down together).
- `update_state_stochastic!` — process noise is also smooth, so per-step random walks don't shred the field.

The cost is one one-time Cholesky (1600×1600, ~0.2 s) plus a per-call `L · z` matvec (2.5 M flops). Negligible compared to the advection loop.

Setting `noise_length_scale: 0` falls back to legacy iid behaviour.

### 1.2 Hourly cadence

YAML now:

```yaml
time_step: 3600.0          # 1 h between filter updates
n_integration_step: 10     # internal advection dt = 360 s
```

CFL safe ceiling at our parameters is `0.2 · dx / max_v = 0.2 · 4000 / 1.75 ≈ 457 s`. Our dt of 360 s sits below that. The one-shot CFL warning never fires.

Per filter step, the advection now moves a wave by `v · time_step ≈ 1.5 · 3600 = 5400 m ≈ 1.35 cells`, versus 0.15 cells before. The truth and particles experience substantially more dynamic action between observations — closer to a real-world hourly InSAR cadence.

Total run length: `200 steps × 1 h = 200 h ≈ 8.3 days`.

### 1.3 Sensor / unsensored RMSE split

[plot_obs_vs_no_obs.jl](../glacier-code/particleda/plot_obs_vs_no_obs.jl) now adds a `rmse_split` helper that computes two RMSE series per case: one over the ~16 sensor cells, one over the ~1584 unsensored cells. Output goes to `rmse_split_sensor_vs_unsensored.png`.

---

## 2. The result

End-of-run numbers from the rerun ablation (200 hourly steps, smooth noise):

| Metric | with obs | no obs |
|---|---:|---:|
| RMSE(β) global, final-10 mean   | **264** | 335 |
| **RMSE at sensor cells**, final | **16**  | 168 |
| **RMSE at unsensored cells**, final | **265** | 336 |
| mean ESS | 466 | 1000 |
| frac ESS > 0.5·Np | 0.45 | 1.00 |

Two things to notice:

- **Global RMSE is finally lower with obs than without.** Under iid noise this was inverted (no-obs 84 vs with-obs 109 — see audit §7). The smooth noise lets observations propagate spatial information: when a particle has the right β at a sensor cell, it also tends to have the right β within ~ℓ of that cell, so favouring it via likelihood reweighting improves a *neighborhood*, not just a point.
- **Sensor-cell RMSE is dramatically better with obs** (16 vs 168, ≈10×). This is the assimilation signal the global average was hiding.

The "no-obs ESS = 1000 exactly" line confirms the ablation flag still works — uniform weights, no information from observations.

---

## 3. Why smooth noise fixes the audit's central puzzle

The audit (§7) traced the counter-intuitive iid result to a combination of:

1. iid noise → particles have independent error at every cell, no neighbour structure
2. Sparse coverage → only 1 % of cells are sensored
3. Likelihood reweighting is global (every particle gets one scalar weight), so a particle that fits a sensor cell at (i, j) is selected *as a whole* — including its bad values at the 1584 unsensored cells

Under iid noise, "the whole particle" has no correlation between sensor-cell quality and unsensored-cell quality. Resampling effectively transports random unsensored fields, which is why unsensored-cell RMSE got *worse* with observations than without.

Under smooth noise with ℓ = 30 km, a particle whose sensor cell is right is *likely also right* within roughly 30 km of that sensor. Now reweighting genuinely improves unsensored cells too — modestly (21 % better, 265 vs 336) but in the right direction.

This is exactly the mechanism described in §8.3 recommendation #1 of the audit. It worked.

---

## 4. Files touched

- [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) — added `noise_length_scale` param, `noise_factor` (Cholesky `L`) and `noise_buffer` in the struct, `_build_noise_factor`, `_apply_noise!`. `sample_initial_state!` and `update_state_stochastic!` now route through `_apply_noise!`.
- [glacier.yaml](../glacier-code/particleda/glacier.yaml) — `time_step: 3600`, `n_integration_step: 10`, `noise_length_scale: 30000`.
- [glacier_random.yaml](../glacier-code/particleda/glacier_random.yaml) — same parameter changes.
- [glacier_no_obs.yaml](../glacier-code/particleda/glacier_no_obs.yaml) — same.
- [run_obs_vs_no_obs.jl](../glacier-code/particleda/run_obs_vs_no_obs.jl) — embedded baseline YAML updated.
- [plot_obs_vs_no_obs.jl](../glacier-code/particleda/plot_obs_vs_no_obs.jl) — `rmse_split` helper + new `rmse_split_sensor_vs_unsensored.png` panel + sensor / unsensored rows in the printed summary.

---

## 5. Glossary additions

| Term | Meaning |
|---|---|
| **Squared-exponential kernel** | `K(x, x′) = exp(−‖x − x′‖² / (2 ℓ²))`. Smooth Gaussian random fields drawn from this kernel have continuous, infinitely differentiable sample paths and a characteristic length scale `ℓ`. Equivalent to Matérn covariance with ν → ∞. |
| **Cholesky factor** | The lower-triangular matrix `L` such that `L Lᵀ = K`. Standard way to sample from a multivariate normal: if `z ∼ N(0, I)` then `L z ∼ N(0, K)`. |
| **Marginal std vs joint correlation** | The marginal std (`σ`) is how much *one cell* moves. Correlation length (`ℓ`) is how strongly its movement is tied to its neighbours. We tune them independently. |
| **CFL safe dt** | `0.2 · dx / max_v ≈ 457 s` in our setup. The upwind scheme is stable for any `dt` below this; we pick `dt = 360 s` for headroom. |
| **Sensor-cell RMSE** | RMSE computed only over cells that contain a sensor. Directly measures "is the filter using the data". |
| **Unsensored-cell RMSE** | RMSE computed over the remaining ~1584 cells. Measures how well information is propagated by the prior+dynamics. |
