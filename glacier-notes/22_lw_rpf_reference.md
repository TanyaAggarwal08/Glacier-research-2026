# LW + RPF reference — what is added, where, and with what parameters

Scope of this note: a single-place reference for (1) the jitter added to
particles after resampling, (2) the Lax–Wendroff advection scheme, and (3)
the resampling routine. Code lives in two files only:

- `glacier-code/particleda/glacier_model.jl` — model: noise builder, the
  `_apply_noise!` helper, and the three advection branches.
- `glacier-code/particleda/run_wendroff_amplitude.jl` — PF driver: the loop
  that calls `_apply_noise!` post-resample, plus the `systematic_resample`
  implementation.

---

## 1. The error term added to particles after resampling (RPF jitter)

### What it is

Spatially-correlated Gaussian noise with marginal standard deviation
`SIGMA_JITTER` (β units) and spatial correlation length
`noise_length_scale` (m). It is the same kind of noise used everywhere
else in the model — same Cholesky factor `L`, just a different σ.

Mathematically, for a single particle state vector `β ∈ ℝⁿ` (n = nx·ny):

```
z  ~ 𝒩(0, Iₙ)              # nx·ny iid standard normals
δ  = σ · L · z              # smooth field with marginal std σ, length-scale ℓ
β ← β + δ
```

where `L` is the Cholesky factor of the squared-exponential covariance

```
K(x₁, x₂) = exp(−|x₁ − x₂|² / (2 ℓ²))   + 1e-8 · I  (PD jitter)
```

built once at `Glacier.init` and cached on `model.noise_factor`.

So the post-resample perturbation is **not white noise** — neighbouring
cells move together, which keeps the perturbed particle a physically
plausible smooth β-field rather than a salt-and-pepper mess.

### Why it is added (RPF, Musso–Oudjane–Le Gland 2001)

Systematic resampling produces many exact duplicates of the high-weight
particles. Without a jitter step the ensemble collapses to a handful of
identical trajectories ("sample impoverishment"). The post-resample jitter
re-spreads each duplicate by σ in β-space so the ensemble keeps diversity
without losing the information the weights already concentrated.

### Where it is added (exact lines)

Driver: `glacier-code/particleda/run_wendroff_amplitude.jl`

- **Constant `SIGMA_JITTER = 30.0`** declared on **line 111**.
- **Jitter loop** runs inside the resample block on **lines 166–171**:
  ```julia
  if SIGMA_JITTER > 0
      for p in 1:NPRT
          Glacier._apply_noise!(view(particles, :, p), model, rng_pf,
                                SIGMA_JITTER, 1)
      end
  end
  ```
  This sits immediately after `particles = particles[:, idx]` (the
  systematic resample on lines 162–163) and immediately before
  `log_weights .= 0.0` (line 172).

Model helper: `glacier-code/particleda/glacier_model.jl`

- **`_apply_noise!` definition** on **lines 150–165** — this is the function
  the driver calls. It draws `z ~ 𝒩(0, I)` (lines 158–160) and computes
  `state .+= σ · L · z` via `mul!(state, model.noise_factor, z, σ, 1.0)`
  on **line 162**.
- **`_build_noise_factor`** on **lines 92–116** builds `L` from the
  squared-exponential covariance with length scale `noise_length_scale`
  (the kernel is on **line 108**).

The same `_apply_noise!` is also called by `sample_initial_state!`
(line 196, σ = `init_std_beta`) and by `update_state_stochastic!`
(line 288, σ = `process_std_beta`), so the only thing distinguishing the
RPF burst from prior/process noise is the σ value — the spatial structure
is identical.

---

## 2. Lax–Wendroff advection

### What it does

Replaces the first-order upwind stencil (which has a built-in numerical
diffusion `D_num = ½·v·Δx·(1−CFL)`) with a second-order centred
difference plus a `½·α²` correction. The correction is exactly the
amount of *anti*-diffusion needed to cancel the leading-order numerical
diffusion of pure centred differences, so the modified equation has
`D_num ≡ 0` at second order. Trade-off: dispersion (oscillations near
sharp edges), but for the smooth sinusoidal β-prior the amplitude is
preserved (4.9 % loss vs. 99.99 % for default upwind — see Exp 02).

### The stencil

For each cell `(i, j)`, with periodic BCs (`mod1`), `α = v · dt / dx`:

```
β_new[j, i] = β[j, i]
              − 0.5·α     · (β[j, i+1] − β[j, i−1])      # centred diff
              + 0.5·α·α   · (β[j, i+1] − 2·β[j, i] + β[j, i−1])   # ½·α² correction
```

Note the velocity is `v = 1 + ε·β`, so with `advection_epsilon = 0.0` the
scheme is strictly linear (constant `v = 1`, no β-feedback).

### Where it lives (exact lines)

File: `glacier-code/particleda/glacier_model.jl`

- The whole `if p.advection_type == "lax_wendroff"` branch sits inside
  `ParticleDA.update_state_deterministic!` on **lines 246–263**.
- Per-cell update: **lines 252–256** (`α = v · dt / dx`, then the stencil).
- The outer `for _ in 1:p.n_integration_step` (line 247) runs the stencil
  `n_integration_step` times per call so each filter step advances by
  `dt = time_step / n_integration_step` internally — `n_integration_step`
  small substeps per one big filter step.
- The CFL safety warning (lines 227–235) fires once per process if
  `dt > 0.2 · dx / v_max`.

Background note: `glacier-notes/21_numerical_diffusion.md` has the full
modified-equation derivation.

---

## 3. Resampling

### Method

Systematic resampling (Douc–Cappé 2005, the standard low-variance
choice). Single uniform `u₀ ~ U(0, 1/N)`; then the N selection points
are `u₀ + (i−1)/N`. One pass through the cumulative weight vector picks
the indices.

### ESS-gated firing (Liu & Chen 1998)

Resampling **only** fires when `ESS < 0.5·N`. The cumulative log-weights
are carried forward across steps until that threshold is crossed; on
resample they reset to zero (uniform). This keeps the filter from
discarding information when ESS is healthy.

### Where it lives (exact lines)

File: `glacier-code/particleda/run_wendroff_amplitude.jl`

- **`systematic_resample(w, rng)`** definition on **lines 117–131**.
- **`const ESS_THRESHOLD = 0.5 * NPRT`** on **line 103**.
- **Cumulative log-weights** array initialised on **line 102** and updated
  inside the time loop on **lines 144–147**.
- **ESS evaluation** on **line 154** (`1.0 / sum(w .^ 2)`).
- **The gate** — `if ess_series[t] < ESS_THRESHOLD` on **line 161** — wraps
  the resample (line 162), the index gather (line 163), the RPF jitter
  (lines 166–171), and the log-weight reset (line 172).

---

## 4. Parameters in use (run10_wendroff_amplitude)

All values are the constants set at the top of
`glacier-code/particleda/run_wendroff_amplitude.jl` and the dict passed
to `Glacier.init`. See the file for the source-of-truth values.

| Group | Name | Value | Unit | Role |
|---|---|---:|---|---|
| Filter | `NPRT` | 1000 | — | particles |
| Filter | `T` | 100 | steps | total filter steps (≈ 100 h) |
| Filter | `K_TRACK` | 15 | — | tracked-particle subset for plots |
| Filter | `ESS_THRESHOLD` | `0.5·N` = 500 | — | resample only when ESS < this |
| Filter | `SIGMA_JITTER` | 30.0 | β units | RPF post-resample marginal std |
| Filter | `SEED_PF` / `SEED_OBS` | 42 / 123 | — | PF init RNG / truth+obs RNG |
| Grid | `nx`, `ny` | 40, 40 | — | state dim n = 1600 |
| Grid | `x_length`, `y_length` | 160 000, 160 000 | m | physical domain |
| Prior | base + amp | 2000 + 2000·sin·sin | Pa·s/m | mean field (n_modes = 3) |
| Prior | `init_std_beta` (σ_init) | 150.0 | β units | initial spread |
| Process | `process_std_beta` (σ_proc) | 7.0 | β units | per-step model noise |
| Process | `noise_length_scale` (ℓ) | 15 000 | m | smoothness of all noise (init, proc, RPF) |
| Process | `min_beta` | 10.0 | Pa·s/m | floor on β to keep it positive |
| Obs | `obs_noise_std` (σ_obs) | 0.10 | ux units (ux = 1000/β) | symmetric Gaussian on ux |
| Obs | station file | `stations_crosssection.txt` | — | 10 sensors; 1 on the y = 80 km row |
| Adv | `advection_type` | `"lax_wendroff"` | — | which stencil to run |
| Adv | `advection_epsilon` (ε) | 0.0 | — | β-feedback off → strictly linear |
| Adv | `n_integration_step` | 10 | — | substeps per filter step |
| Adv | `time_step` | 3600.0 | s | hourly filter cadence; internal dt = 360 s |

CFL with these numbers: `dx = 4000 m`, `dt_inner = 360 s`, `v = 1 m/s` ⇒
`α = v·dt/dx = 0.09`, comfortably below the LW stability ceiling of 1.

---

## See also
- `21_numerical_diffusion.md` — why LW beats first-order upwind, with the
  modified-equation derivation and amplitude predictions used in Exp 02.
- `rmse_experiment_log.md` — every sweep (σ_obs, σ_init, ℓ, …) recorded
  with the exact params used.
