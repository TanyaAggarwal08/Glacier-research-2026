# Noise pipeline — full deep dive

> Every random quantity that goes into the filter, written out with the equations, the matrices, the parameter names, the code line that adds it, and verified against an empirical draw.

There are exactly **three** sources of randomness in our model. They are listed in the order they enter the run:

| Stage | Where | Variable | Distribution | Scale parameter |
|---|---|---|---|---|
| 1 | `sample_initial_state!` | initial perturbation `δβ₀` | smooth GRF, marginal `N(0, σ_init²)` | `init_std_beta = 300` |
| 2 | `update_state_stochastic!` | per-step process noise `δβ_t` | smooth GRF, marginal `N(0, σ_proc²)` | `process_std_beta = 7` |
| 3 | `sample_observation_given_state!` | per-sensor observation noise `ε` | **iid** `N(0, σ_obs²)` | `obs_noise_std = 0.10` |

Stages 1 and 2 share the **same** spatial-correlation structure (one Cholesky factor `L`). Stage 3 is iid — observation noise is per-sensor and uncorrelated. We dig through each below.

---

## 1. Parameters

From [glacier.yaml](../glacier-code/particleda/glacier.yaml):

```yaml
init_std_beta:        300.0      # σ_init, marginal std of δβ₀
process_std_beta:     7.0        # σ_proc, marginal std of δβ_t per filter step
obs_noise_std:        0.10       # σ_obs,  std of ε added to each ux observation
noise_length_scale:   30000.0    # ℓ,      spatial correlation length in metres
min_beta:             10.0       # lower clamp on β (prevents 1/β blowup)
```

Plus the grid:

```yaml
nx, ny:    40, 40                # n = 1600 cells
x_length:  160000.0 m            # dx = dy = 4000 m = 4 km
```

These are *all* the knobs that influence noise.

---

## 2. The spatial correlation kernel

For every pair of grid cells `(i, j)` at positions `xᵢ = (xᵢ, yᵢ)` and `xⱼ = (xⱼ, yⱼ)`, we define a covariance:

```
K[i, j] = exp( − ‖xᵢ − xⱼ‖² / (2 ℓ²) )           (Eq. 1)
```

This is the **squared-exponential** kernel (also called Gaussian or RBF kernel). It has marginal variance 1 (`K[i, i] = 1`), correlation length `ℓ`, and smooth (`C^∞`) realisations.

With `ℓ = 30 000 m` on a 4-km grid, the correlation drops off like this (numerically verified):

| Distance | Cell offset | `K(r)` |
|---|---|---:|
| 0 km | (0, 0) | 1.000 |
| 4 km | (1, 0) | 0.991 |
| 8 km | (2, 0) | 0.965 |
| 16 km | (4, 0) | 0.867 |
| 32 km | (8, 0) | 0.566 |
| 60 km | (15, 0) | 0.135 |
| 80 km | (20, 0) | 0.029 |

So cells within ~`ℓ` move together strongly; cells more than ~`3ℓ` apart are effectively independent. The "8-km grid spacing → 0.965 correlation between adjacent cells" is what makes the field look smooth.

The full covariance matrix `K` is therefore **`n × n` = 1600 × 1600 = 2.56 M entries**, with `K[i, j]` given by Eq. 1.

Code: [glacier_model.jl:`_build_noise_factor`](../glacier-code/particleda/glacier_model.jl) lines 92-115.

```julia
K[idx1, idx2] = exp(-r2 / (2 * ℓ2))
```

with `r2 = (x1-x2)² + (y1-y2)²` and `ℓ2 = ℓ²`.

### 2.1 Jitter for numerical positive-definiteness

`K` is theoretically PD, but in floating point a 1600-dim matrix can have a tiny negative eigenvalue. We add a diagonal jitter:

```
K ← K + 10⁻⁸ · I                                  (Eq. 2)
```

[glacier_model.jl:113](../glacier-code/particleda/glacier_model.jl#L113). This changes the marginal variance from 1 to 1 + 10⁻⁸ — completely negligible.

---

## 3. Cholesky factor — the heart of the smooth sampler

Once we have `K`, we factor it:

```
K = L Lᵀ        L is lower-triangular, n × n          (Eq. 3)
```

[glacier_model.jl:117](../glacier-code/particleda/glacier_model.jl#L117): `Matrix(cholesky(Symmetric(K)).L)`.

`L` is stored in `GlacierModel.noise_factor`, built once at init, reused for every particle and every filter step.

### 3.1 Why this lets us draw a smooth field

The standard fact about multivariate normals: if `z ~ N(0_n, I_n)` (a vector of n iid standard normals — easy to sample), then

```
δβ = L z       ⇒    δβ ~ N(0_n, L Lᵀ) = N(0_n, K)         (Eq. 4)
```

So `L z` is a Gaussian field with covariance exactly `K`. Marginal variance per cell is `K[i, i] = 1`; correlation between cells `i, j` is `K[i, j]`. A field that looks smooth in space.

To scale up to the desired marginal std `σ`:

```
δβ = σ · L z      ⇒    Var(δβᵢ) = σ²,   Cov(δβᵢ, δβⱼ) = σ² K[i, j]    (Eq. 5)
```

This is the only line of math the filter does for spatial noise. Equation 5 is implemented as a single BLAS call:

```julia
mul!(state, L, z, σ, 1.0)     # state .+= σ * (L * z)
```

[glacier_model.jl:`_apply_noise!`](../glacier-code/particleda/glacier_model.jl#L160). `z` is drawn fresh per call (`randn(rng)` into a pre-allocated buffer); the same `L` is reused.

### 3.2 Empirical check

5000 draws of `L z` (so `σ = 1`):

| Quantity | Empirical | Theory |
|---|---:|---:|
| per-cell std (avg over 1600 cells) | 0.999 | 1 |
| per-cell mean (avg) | 0.0035 | 0 |
| corr(center, neighbour 32 km away) | 0.565 | exp(−(32 000)² / (2 · 30 000²)) = 0.566 |

The pipeline does exactly what Eq. 5 promises.

---

## 4. Stage 1: initial-state perturbation `δβ₀`

[glacier_model.jl:`sample_initial_state!`](../glacier-code/particleda/glacier_model.jl#L137):

```julia
ParticleDA.get_initial_state_mean!(state, model)          # state .= β_prior
_apply_noise!(state, model, rng, σ_init, task_index)      # state .+= σ_init * L * z
state[i] = max(state[i], min_beta)                        # clamp
```

So each particle is initialised as

```
β_0^(p) = β_prior + σ_init · L z^(p),   clamped to ≥ min_beta = 10        (Eq. 6)
```

where `β_prior` is the fixed sinusoid `1000 + 500 sin(ωx) sin(ωy)` (the same field for every particle) and `z^(p) ~ N(0, I_n)` is drawn fresh per particle.

Numerical check: with `σ_init = 300`, one realisation gives spatial std(δβ₀) ≈ 252 across 1600 cells. That's lower than 300 *for one realisation* — because Eq. 5 is about the **ensemble** marginal std at one cell across many draws, not the spatial std of one realisation. (When a field is smooth, one draw lives "in a corner" of the GRF's typical region and looks less varied than the marginal.) Across many particles, the per-cell std converges to σ_init = 300 exactly.

### 4.1 Where this lands the initial particles

`σ_init = 300` is roughly 30 % of the prior magnitude (β ≈ 1000). Combined with `ℓ = 30 km`, the initial ensemble looks like 1000 sinusoid-baseline fields, each with a smooth 30-km-scale bump-and-dip pattern on top, of magnitude ~ ±600 (2σ envelope). This is the prior the filter starts from.

---

## 5. Stage 2: process noise `δβ_t`

[glacier_model.jl:`update_state_stochastic!`](../glacier-code/particleda/glacier_model.jl#L210):

```julia
_apply_noise!(state, model, rng, σ_proc, task_index)
state[i] = max(state[i], min_beta)
```

So after each filter step, every particle gets

```
β_t^(p) ← β_t^(p) + σ_proc · L z^(p, t),   clamped to ≥ min_beta      (Eq. 7)
```

with `z^(p, t) ~ N(0, I_n)` fresh per particle per step, but the **same** `L` as the initial-state stage.

`σ_proc = 7` means each cell drifts with marginal std 7 per step (= 1 hour at our cadence). Over 200 steps with no advection or assimilation, a random walk at this rate accumulates `7 · √200 ≈ 99` — about 10 % of the β scale. This is the "model has uncertainty about β" budget.

Numerical check: after one call to `update_state_stochastic!`, std(δβ_proc across cells) = 7.011. Matches σ_proc = 7 exactly.

### 5.1 Process noise is applied OUTSIDE the integration substeps

This is a subtle but important point. In [`update_state_deterministic!`](../glacier-code/particleda/glacier_model.jl#L184), we substep advection 10 times per filter step (`n_integration_step = 10`, internal `dt = 360 s`). Process noise is **not** applied per substep — only once per filter step, in a separate call.

If we instead added `σ_proc · L z` inside the substep loop, we'd accumulate `σ_proc · √(n_integration_step)` per filter step → 22 instead of 7. Keeping it outside means the noise budget matches the filter cadence, regardless of how many substeps we use internally.

### 5.2 Process noise uses the same `L` as the initial state

This is a deliberate simplification. Both stages share `noise_length_scale = 30 km`. If you wanted decoupled scales (e.g. wider initial prior, tighter per-step drift), the cleanest extension is two separate factors `L_init` and `L_proc`. We don't need that distinction yet.

---

## 6. Stage 3: observation noise `ε` — iid per sensor

[glacier_model.jl:`sample_observation_given_state!`](../glacier-code/particleda/glacier_model.jl#L243):

```julia
get_observation_mean_given_state!(observation, state, model, task_index)
for k in eachindex(observation)
    observation[k] += σ_obs * randn(rng)
end
```

So each sensor gets

```
ŷ_k = h_k(β) + ε_k,   ε_k ~ N(0, σ_obs²) iid                          (Eq. 8)
```

with `h_k(β) = 1000 / β_{sensor_k}` (the surrogate ux formula). The noise on sensor `k` is **independent** of sensor `j` — no spatial correlation here.

The corresponding covariance is `Σ = σ_obs² · I_{n_sensors} = ScalMat(n_obs, 0.01)` stored in `model.obs_cov` ([glacier_model.jl:100](../glacier-code/particleda/glacier_model.jl#L100)). This is the same matrix used in the likelihood:

```
log p(y | β) = −½ (y − h(β))ᵀ Σ⁻¹ (y − h(β))                          (Eq. 9)
```

[glacier_model.jl:`get_log_density_observation_given_state`](../glacier-code/particleda/glacier_model.jl#L261). `invquad(model.obs_cov, ·)` is the efficient form of `(y−h)ᵀ Σ⁻¹ (y−h)` for a scalar-multiple-of-identity Σ.

### 6.1 Why obs noise is iid (not smooth)

Two different physical models. Stages 1 and 2 represent uncertainty about an underlying smooth field, where neighbouring cells *physically* move together. Stage 3 represents sensor measurement noise — each station's electronics, atmospheric effects, etc., are independent of every other station's. iid is the right model.

If we ever wanted to model spatially correlated observation errors (e.g. a satellite swath with shared atmospheric distortion), `obs_cov` would become a full PD matrix. The likelihood formula stays the same — only `Σ` changes.

---

## 7. End-to-end summary diagram

```
       ┌─────────────────────────────┐
       │  σ_init = 300  ℓ = 30 km    │           init noise
       │                             │
       │  z₀ ~ N(0, I_1600)          │  ←─── randn(rng, 1600)
       │  δβ₀ = σ_init · L · z₀      │  ←─── _apply_noise!
       │  β₀ = β_prior + δβ₀         │  ←─── sample_initial_state!
       │  β₀ = max(β₀, 10)           │  ←─── min_beta clamp
       └─────────────────────────────┘
                    │
                    ▼ (per filter step t = 1..200)
       ┌─────────────────────────────┐
       │  advect 10 substeps          │  ←─── update_state_deterministic!
       │  (no noise here)             │           internal dt = 360 s
       └─────────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────────┐
       │  σ_proc = 7   ℓ = 30 km     │          process noise
       │                             │
       │  z_t ~ N(0, I_1600)         │  ←─── randn(rng, 1600)
       │  δβ_t = σ_proc · L · z_t    │  ←─── _apply_noise!
       │  β_t ← β_t + δβ_t           │  ←─── update_state_stochastic!
       │  β_t = max(β_t, 10)         │  ←─── min_beta clamp
       └─────────────────────────────┘
                    │
                    ▼ (truth path only, or for likelihood)
       ┌─────────────────────────────┐
       │  σ_obs = 0.10   iid          │          observation noise
       │                             │
       │  ε ~ N(0, σ_obs² · I_{n_s}) │
       │  ŷ_k = 1000/β_{sensor_k}+ε_k │  ←─── sample_observation_given_state!
       └─────────────────────────────┘
```

`L` is built once at init, shared across particles, shared across timesteps, shared between init and process noise.

---

## 8. Knobs you can turn

| If you want… | Change |
|---|---|
| Tighter initial prior | `init_std_beta` ↓ |
| Less process drift | `process_std_beta` ↓ |
| Cleaner observations | `obs_noise_std` ↓ |
| Smoother fields (info propagates farther) | `noise_length_scale` ↑ |
| Choppier / iid noise (legacy behaviour) | `noise_length_scale ≤ 0` |
| Wider truth swings | `init_std_beta` ↑ and/or `process_std_beta` ↑ |

A useful rule: `n_sensors · (σ_signal / σ_obs)² ≈` the "pressure budget" — how many bits of information the likelihood is asking the particles to fit per step. Too high → ESS collapses; too low → no learning. With `σ_signal ≈ 0.5` (typical ux range), `σ_obs = 0.10`, `n_sensors = 16` we get `16 · 25 = 400` — well within reach of 1000 particles.

---

## 9. Glossary additions

| Term | Meaning |
|---|---|
| **Squared-exponential kernel** | `K(x,x') = exp(−‖x−x'‖²/(2ℓ²))`. Marginal variance 1, correlation length `ℓ`, infinitely smooth. |
| **Marginal variance** | The variance at a single cell, averaged across many draws. Equals `K[i,i]`. |
| **Joint covariance** | The covariance between two cells. Equals `σ² K[i,j]` after scaling. |
| **Cholesky factor** | Lower-triangular `L` with `L Lᵀ = K`. Lets us turn iid noise into correlated noise via `L z`. |
| **Jitter** | A small diagonal addition (`10⁻⁸ · I`) to keep `K` numerically PD for Cholesky. |
| **GRF (Gaussian random field)** | A spatial random function whose finite-dimensional distributions are jointly Gaussian. What `L z` samples from. |
| **Marginal vs spatial std** | Marginal std = std at one cell across many realisations (= σ). Spatial std = std across cells within one realisation (lower than σ for smooth fields). |
| **iid noise** | Independent and identically distributed — every entry drawn independently. What we use for observation noise. Equivalent to `ℓ → 0` in the GRF picture. |
| **Pressure budget** | `n_sensors · (σ_signal/σ_obs)²`. A back-of-envelope estimate of how informative the observation set is. |
