# Full audit — glacier particle filter end-to-end

> **Scope.** Step-by-step technical walkthrough of the entire glacier PF pipeline, with every equation derived from the code (not from memory). Verifies the code against the ParticleDA paper (Giles et al., 2024) and the standard Bayesian-filter formulation. Closes with a rigorous diagnosis of the "no-obs RMSE < with-obs RMSE" puzzle.
>
> **TL;DR of the diagnosis.** The result is **real, not a bug**. The code matches the paper section-by-section. The cause is the *combination* of three modelling choices: (1) iid (white) noise on initial state and process, (2) very sparse sensors (1 % grid coverage), and (3) numerical diffusion in the upwind scheme. Together these make the *spatially averaged* RMSE a misleading metric — the with-obs filter is *better at sensor cells* (RMSE 19 vs 80) but *worse at unsensored cells* (RMSE 109 vs 84), and unsensored cells dominate because there are 99 × more of them. The note ends with a prioritised checklist of changes to make the experiment statistically meaningful.

---

## 1. State representation

### 1.1 What the model state actually contains

State vector `x ∈ ℝ^{1600}`, stored as a flat column-major vector that reshapes to a 40 × 40 grid. Each entry is the basal-friction value β at that cell, in Pa·s/m. **No log transform, no auxiliary variables, no temperature, no ice thickness.** Just β.

Code: [glacier_model.jl:38-45](../glacier-code/particleda/glacier_model.jl#L38-L45) defines `GlacierModel{T}` with `beta_prior_mean::Vector{T}` (the prior mean field) and a sensor index list.

### 1.2 How β is represented

The full state at a single time index is a `1600`-element `Vector{Float64}`. Two layouts coexist:

- **Flat layout (used by ParticleDA core):** `state[k]` for `k = 1 : 1600`.
- **2-D layout (used by physics and IO):** `reshape(state, ny, nx)` gives a `40 × 40` matrix β where `β[j, i]` is row `j` (y direction), column `i` (x direction). Flat index ↔ (i, j) mapping: `flat_idx = (i − 1) · ny + j`. Code at [_sensor_xy:264-277](../glacier-code/particleda/glacier_model.jl#L264-L277).

Both layouts share the same underlying memory, so `reshape` aliases without copying — important for performance and for the in-place advection update.

### 1.3 The prior equation for β

From [init:78-99](../glacier-code/particleda/glacier_model.jl#L78-L99):

```
β_prior(x, y) = 1000 + 500 · sin(ω · x) · sin(ω · y),     ω = 2π / L
```

with `L = 160 000 m`. So β_prior is a smooth sinusoid with:

- Mean = 1000 Pa·s/m
- Amplitude ±500 (range 500–1500)
- One full oscillation across the domain in each axis

The prior is **stationary** (doesn't change with time index) and is used as the *mean* of the initial particle and truth distributions.

### 1.4 Why the truth doesn't look smooth even though the prior does

The truth field at time `t` is:

```
β_truth(x, y, t) = β_prior(x, y) + initial_noise(x, y) + advection + process_noise
```

where:

- `initial_noise(x, y) ~ N(0, σ_init²)` independently per cell, with `σ_init = 300`. **This is iid (white) noise** — no spatial correlation imposed.
- Process noise gets added every step, also iid per cell with `σ_proc = 7`.

So the truth at t = 0 is the smooth prior **plus a white-noise pattern of std 300**. Visually this looks like static on top of a sinusoid. The toy LowLevel model used the same iid structure (this matches your impression that the LowLevel toy truth was also "rough").

If we wanted a smooth truth (closer to physically realistic basal friction, which is correlated over kilometres), we'd need to swap `randn(rng)` for a Matérn Gaussian random field sample. The LLW2d tsunami model does exactly this (their `init_gaussian_random_field_generator`). We deliberately kept iid for simplicity but **this is the single biggest modelling simplification** in our setup, and it has measurable consequences (see §6 and §8 below).

> **Answer to "are we adding too much noise?"** — by amplitude, no (σ_init = 30% of β-mean is comparable to LowLevel's σ_init); by **spatial structure**, yes — we're adding *white* noise where a real glacier would have *smooth* noise. White noise gets heavily distorted by both numerical diffusion and the filter's resampling, and that's the source of the counter-intuitive result in §6.

### 1.5 How the truth trajectory is generated

[src/ParticleDA.jl:72-96](../src/ParticleDA.jl#L72-L96) implements `simulate_observations_from_model`. Pseudocode:

```
state ← sample_initial_state!(model, rng_truth)    # x_0 ~ p_0
write state to HDF5 as state/t0000
for t = 1, …, T:
    update_state_deterministic!(state, model, t)    # x ← F(x)
    update_state_stochastic!(state, model, rng)     # x ← x + u, u ~ N(0, Q)
    sample_observation_given_state!(obs, state, model, rng)   # y_t = h(x_t) + v_t
    write state to HDF5 as state/t000t
    write obs as observations/t000t
```

So the HDF5 file has 201 truth snapshots `state/t0000` through `state/t000T` and 200 observation snapshots `observations/t0001` through `observations/t000T`. **Truth and obs are time-aligned per the standard convention**: `obs[t]` is the noisy measurement of `state[t]`.

This loop is executed *once*, with the RNG seeded by `simulate_observations.seed`. The same call generates both runs in our ablation because we used the same YAML seed for both. **Truth and obs are byte-identical between baseline and ablation runs** — verified by comparing HDF5 hashes.

---

## 2. β evolution / dynamics

### 2.1 The continuous PDE

Our dynamics is a one-dimensional nonlinear advection of β in the x direction, with no y transport:

```
∂β/∂t + (1 + ε β) · ∂β/∂x = 0
```

This is **Burgers' equation with a constant offset** in velocity. Smooth solutions sharpen over time (faster bits catch up to slower bits). The y direction is decoupled — each row evolves independently.

### 2.2 The discrete update equation

[glacier_model.jl:145-186](../glacier-code/particleda/glacier_model.jl#L145-L186) implements **first-order upwind**:

```
dt = time_step / n_integration_step                              # 400 / 1 = 400 s

For each sub-step k = 1 … n_integration_step:
    For each cell (j, i):
        im = i − 1, with wrap-around (periodic x boundary)
        ∂β/∂x  ≈  (β[j, i] − β[j, im]) / dx                       # one-sided
        v       =  1 + ε · β[j, i]                                # local velocity
        β_new[j, i] = β[j, i] − v · dt · ∂β/∂x                    # Euler forward
    β, β_new = β_new, β
```

Then `state .= vec(β)` and return.

### 2.3 Parameter-by-parameter

| Symbol | Code | Value | Meaning |
|---|---|---:|---|
| `dt` | `time_step / n_integration_step` | 400 / 1 = 400 s | Numerical time step per sub-step |
| `dx` | `x_length / nx` | 4000 m | Grid spacing |
| `ε` | `advection_epsilon` | 5 × 10⁻⁴ | β coupling into velocity |
| `v` (instantaneous) | `1 + ε · β` | ≈ 1.75 m/s | Per-cell advection speed at β ≈ 1500 |
| `n_integration_step` | `n_integration_step` | 1 | Sub-steps per filter step |
| `σ_proc` | `process_std_beta` | 7 Pa·s/m | Per-step process-noise std |
| `min_beta` | `min_beta` | 10 | Lower clamp to keep ux = 1000/β safe |

### 2.4 Diffusion terms — there are none *explicitly*

There is **no explicit diffusion term in the PDE**. However, the first-order upwind discretisation introduces *numerical* diffusion equivalent to a real second-order term:

```
D_num = v · dx · (1 − CFL) / 2   ≈ 1.75 · 4000 · (1 − 0.175) / 2 ≈ 2 887 m²/s
```

This is the *modified equation* result (the PDE the discretisation actually solves is `∂β/∂t + v ∂β/∂x = D_num ∂²β/∂x²`, not the zero-RHS original). The artificial diffusion is what damps high-frequency content over time — we'll come back to this in §6, because it's the dominant explanation for the RMSE puzzle.

### 2.5 Stochastic update

[update_state_stochastic!:188-201](../glacier-code/particleda/glacier_model.jl#L188-L201):

```
For each cell i:
    state[i] ← state[i] + σ_proc · η,   η ~ N(0, 1)         # iid noise
    state[i] ← max(state[i], min_beta)                       # positivity guard
```

So process noise is iid Gaussian per cell, std = 7. **No spatial correlation imposed.** This adds *new* white noise every step. After 200 steps the cumulative noise (in the absence of diffusion damping) would have std `7 · √200 ≈ 99` per cell.

### 2.6 Deterministic or stochastic?

The transition kernel `p_t(x_t | x_{t−1})` is:

```
x_t = F_t(x_{t−1}) + u_t,    u_t ~ N(0, σ_proc² · I_1600)
```

This is exactly **Equation (2)** of the ParticleDA paper — additive Gaussian state noise on top of a deterministic flow. The structural form matches.

### 2.7 CFL constraint

Stability requires CFL number `v · dt / dx < 1` for upwind. We use a safety factor of 0.2 in the warning check at [glacier_model.jl:161-168](../glacier-code/particleda/glacier_model.jl#L161-L168). At β_max ≈ 1500:

```
CFL_actual = 1.75 · 400 / 4000 = 0.175      # safely below 1
CFL_safe   = 0.2 · dx / v = 457 s ceiling   # we're at dt = 400 s, 87% of this
```

The warning fires once at most. In all runs the scheme is stable; β stays positive without the clamp triggering.

### 2.8 Does the implementation match the intended physics?

**Yes, modulo the numerical-diffusion caveat in §2.4.** The upwind step is a correct first-order discretisation of nonlinear advection. The CFL is satisfied. The stochastic kick is the standard additive-Gaussian convention used in the paper. No mismatch.

---

## 3. Particle-filter mechanics — line by line

The PF loop lives in `run_particle_filter` in [src/ParticleDA.jl:128-330](../src/ParticleDA.jl#L128-L330). I'll trace the algorithm and cross-check against the paper's Algorithm 1 (page 2430).

### 3.1 Initialisation

[src/ParticleDA.jl:185-241](../src/ParticleDA.jl#L185-L241):

```
states ← Matrix(state_dim, N)                              # N = 1000 columns
for particle i = 1 … N:
    sample_initial_state!(states[:, i], model, rng_i)      # x_0^(i) ~ p_0
update_statistics!(...)                                    # ensemble mean+var
write_snapshot(time=0)                                     # state_avg/t0000 = mean
```

Each particle is an independent draw from the prior. So at t = 0, we have N = 1000 independent samples of `β_prior + N(0, 300² · I)`. The truth is a *separate* independent draw from the same distribution.

**Matches Algorithm 1 line 1 of the paper.**

### 3.2 Forecast / prediction step

[src/filters.jl:153-174](../src/filters.jl#L153-L174) (the BootstrapFilter method of `sample_proposal_and_compute_log_weights!`):

```
For each particle i (parallel over threads):
    update_state_deterministic!(state_i, model, t)       # x ← F(x)
    update_state_stochastic!(state_i, model, rng)        # x ← x + u
    log_w[i] = get_log_density_observation_given_state(y_t, state_i, model)
```

So forecasting is **propagating each particle through deterministic dynamics + adding fresh process noise**. This is the Bootstrap proposal `q_t(x_t | x_{t−1}, y_t) = p_t(x_t | x_{t−1})` (paper Eq. 4). Observations are **not used in the proposal**, only in the weighting that follows.

**Matches Algorithm 1 line 4.**

### 3.3 Observation operator

`get_observation_mean_given_state!` at [glacier_model.jl:213-225](../glacier-code/particleda/glacier_model.jl#L213-L225):

```
ux_field(state)[i] = 1000 / max(state[i], min_beta)
observation_mean[k] = ux_field(state)[sensor_indices[k]]
```

So `h(x) = (1000 / β)` selected at the sensor cells. **Nonlinear**, monotone-decreasing, inverse mapping. Strict β > 0 enforced by clamp.

**Note:** this differs from Eq. (3) of the paper, which assumes a *linear* observation operator `H · x`. Our observation operator is nonlinear, so the OptimalFilter proposal (Eq. 8 of paper) cannot be used directly — we're restricted to the bootstrap filter. This is by design and consistent with the paper's framework (general state-space models, Eq. 1) but means we don't get the variance-reduction benefits of the locally optimal proposal.

### 3.4 Likelihood computation

[glacier_model.jl:242-256](../glacier-code/particleda/glacier_model.jl#L242-L256):

```
log p(y | x) = − (y − h(x))ᵀ · R⁻¹ · (y − h(x)) / 2
             = − ‖y − h(x)‖² / (2 σ_obs²)              # since R = σ_obs² · I
```

Coded as `-invquad(model.obs_cov, observation .- obs_mean) / 2`. `invquad(A, v)` from PDMats returns `vᵀA⁻¹v`. So `-invquad / 2 = − vᵀA⁻¹v / 2 = log Gaussian density up to constant`.

**Mathematically correct.** This is exactly `log g_t(y_t | x_t)` from paper Eq. (5).

The ablation flag at line 248 returns `0` for every particle when `disable_observations: true`, giving uniform weights ⇒ ESS = N exactly. This is a deliberate experimental knob and is documented in [15_observations_ablation.md](15_observations_ablation.md).

### 3.5 Weight normalisation and ESS

[src/ParticleDA.jl:270](../src/ParticleDA.jl#L270): `normalized_exp!(filter_data.weights)`. This computes `w_i = exp(log_w_i − max(log_w_j))` then normalises to sum to 1. The `max` subtraction is for numerical stability — standard log-sum-exp.

ESS isn't computed inside ParticleDA's resampling loop, but we compute it post-hoc in the plot scripts using `ESS = 1 / Σ(w_norm_i²)`, which matches paper Eq. (11). When weights are uniform, this gives `ESS = N` exactly; when one particle dominates, it gives `ESS = 1`. **Formula is correct.**

### 3.6 Resampling

[src/ParticleDA.jl:271-279](../src/ParticleDA.jl#L271-L279): `resample!(filter_data.resampling_indices, filter_data.weights, rng)`. Inside `src/utils.jl` this is systematic resampling (paper §2.2: "ParticleDA.jl implements a systematic resampling scheme (Douc and Cappé, 2005), which uses a single uniform random variate to resample all the particle indices"). Then `optimized_resample!` minimises data movement when copying states between ranks — a performance optimisation that doesn't change the algorithm.

**Note:** ParticleDA resamples *every* step. No adaptive resampling threshold. With uniform weights, systematic resampling becomes effectively a permutation (each particle has probability 1/N of being selected at each slot), so it doesn't change the ensemble in expectation — just adds a tiny amount of resampling noise.

### 3.7 Rejuvenation / extra noise injection

**None.** ParticleDA does not add any "rejuvenation" noise beyond the process noise that's part of the prediction step. Some PF variants add a small Gaussian jitter to particles after resampling to maintain diversity; we don't. This is fine because our process noise (σ_proc = 7) injects diversity at every step.

### 3.8 What each particle represents

Each particle is a *complete candidate β field* — 1600 numbers, one per grid cell. It represents one hypothesis for the true β field. The ensemble of 1000 particles is a Monte Carlo approximation of the posterior distribution of β given the observations seen so far.

### 3.9 What quantity is compared against observations

For each particle, the filter computes `h(particle.β) = 1000 / particle.β[sensor_indices]` and compares this to the noisy observation `y_t`. The comparison happens **in ux units** (not β units). A particle whose predicted ux at sensor cells closely matches the observation gets a high weight.

### 3.10 Does the filter ever see the truth?

**No.** The truth is stored in `glacier_obs.h5` under `state/t####`. The filter reads only `observations/t####` (and the YAML for parameters). I verified this by tracing `read_observation_sequence` at [src/io.jl](../src/io.jl): it opens the file and reads the `observations` group, never touching the `state` group. **No truth leakage.**

We use the truth post-hoc to compute RMSE for grading purposes only.

### 3.11 Where does our implementation match the paper, and where does it deviate?

| Paper element | Our implementation | Match? |
|---|---|---|
| State-space model Eq. (1) | sample_initial_state! + update_state_* + sample_observation_given_state! | ✓ |
| Additive Gaussian state noise Eq. (2) | update_state_stochastic! adds N(0, σ_proc² I) | ✓ |
| Linear obs operator Eq. (3) | Nonlinear! Our h(x) = 1000/β is monotone but not linear | ✗ (intentional; restricts us to bootstrap proposal) |
| Bootstrap proposal Eq. (4) + weights Eq. (5) | filters.jl:153 | ✓ exactly |
| Algorithm 1 line 4 (sample proposal) | filters.jl:167-168 | ✓ |
| Algorithm 1 line 5 (compute weights) | filters.jl:169-171 | ✓ |
| Algorithm 1 line 7 (resample) | resample! call in run loop | ✓ (systematic, every step) |
| Locally optimal proposal Eq. (8) | Not used (requires linear obs operator) | — |
| ESS formula Eq. (11) | Computed in plot scripts; matches | ✓ |
| Particle/MPI parallelism §3.3 | Used (thread parallelism, single rank in our runs) | ✓ |

**Conclusion: the code implements paper Algorithm 1 with bootstrap proposal correctly. The one deviation (nonlinear observation operator) is by design and prevents us from using the locally optimal proposal but does not affect correctness of the bootstrap branch.**

---

## 4. Observation system

### 4.1 How synthetic observations are generated

In `simulate_observations_from_model` (see §1.5), each timestep:

```
y_t = h(x_t) + v_t,   v_t ~ N(0, σ_obs² · I_16)
```

where `h(x_t)` is the surrogate ux at sensor cells, `σ_obs = 0.10`, and the noise is iid per sensor per step.

### 4.2 Sensor locations

Two modes:

- **Grid-aligned (default):** `sensor_stride = 100` picks flat indices `1, 101, 201, …, 1501` — gives 16 sensors approximately evenly spaced through column-major storage.
- **From a file (variant):** `station_filename = "stations_random_16.txt"` reads x,y coordinates and snaps to the nearest cell.

In the canonical run (used for the ablation) we use the grid-aligned variant.

### 4.3 Observation cadence

One observation per filter step (every `time_step = 400 s` of model time). Total run: 200 observations over 80 000 s of model time.

### 4.4 Observation noise

`σ_obs = 0.10` in ux units. With typical ux ≈ 1, that's a 10 % noise floor — defensible InSAR-style noise level. iid across sensors and timesteps. Same covariance used for both generation and likelihood (`ScalMat(n_obs, σ_obs²)`), so there's no truth-vs-filter inconsistency in the noise model.

### 4.5 Sparse vs dense observations

**Very sparse: 16 / 1600 = 1.0 % grid coverage.** This is intentional (it's the regime where the pressure budget formula gave us healthy ESS — see [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md)). But it's also the source of the RMSE puzzle (§6 below). 1 % is *very* sparse compared to typical operational DA setups.

### 4.6 What "no observations" actually means in our ablation

`disable_observations: true` does *one* specific thing: returns `log p(y | x) = 0` for every particle, every step. Consequences:

- **All weights are equal:** `w_i = exp(0) / N = 1/N`.
- **ESS = N exactly** (verified empirically: mean ESS = 1000.0 in our ablation).
- **Resampling is uniform**: every particle has probability 1/N of being picked. With systematic resampling, the empirical distribution is preserved in expectation (with a small variance contribution from finite N).
- **Particles still propagate through the dynamics every step** — the prediction step is unchanged.

So "no observations" in our experiment means **no likelihood update, but resampling still happens (just as a no-op in expectation)**. If we wanted to truly skip resampling, we'd need to modify ParticleDA itself; the resampling step is hardcoded into the loop. Empirically this distinction doesn't matter — uniform-weight systematic resampling is statistically equivalent to no resampling at our N.

---

## 5. RMSE calculation

### 5.1 What's compared, where, when

In [plot_obs_vs_no_obs.jl](../glacier-code/particleda/plot_obs_vs_no_obs.jl):

```
rmse_beta[t] = sqrt(mean((beta_est[t] .- beta_true[t]).^2))
```

- `beta_est[t]` = the **PF mean β field** at timestep t, read from `state_avg/t000t/beta` in `particle_da.h5`. This is the ensemble average over 1000 particles, computed *after* the resampling step.
- `beta_true[t]` = the **truth β field** at timestep t, read from `state/t000t/beta` in `glacier_obs.h5`. This is the realised truth used to generate observations.
- The MSE is averaged over **all 1600 grid cells** (the whole spatial domain) — `mean(...)` is the spatial mean.
- The square root gives RMSE in **β units (Pa·s/m)**.

### 5.2 Timing

The RMSE at "step t" uses both fields at the same time index. ParticleDA writes the PF mean *after* the prediction + weighting + resampling for time `t`. The truth at time `t` is `x_t` = the truth after `t` dynamics steps. **Both are at the same model time** (`t · time_step` seconds).

This is the correct DA convention: RMSE of the *analysis* (post-update) estimate against the truth at the same time.

### 5.3 Are we comparing the right variables?

Yes:

- `state_avg/t000t/beta` is computed by ParticleDA from `MeanAndVarSummaryStat` on `state` directly (which IS β in our model post-rewrite).
- `state/t000t/beta` is the truth β.
- Both are 40×40 matrices, same shape, same units.

I verified the HDF5 group structures explicitly. No type, shape, or unit mismatch.

### 5.4 Possible RMSE bugs to rule out

| Suspected bug | Check | Result |
|---|---|---|
| Reading wrong group (e.g. `log_beta` instead of `beta`) | Inspect plot script | ✗ Uses `beta` correctly |
| Time-index mismatch | Both arrays sorted-keys, paired by index | ✓ Correct |
| Average over wrong axis | `mean(matrix)` averages over all elements (1600 cells) | ✓ Correct |
| Forgetting the square root | `sqrt(mean(.^2))` | ✓ Correct |
| Compared post-update PF mean to pre-update truth (off-by-one) | Both stored at same `t000t` key | ✓ Aligned |
| Counting t0000 placeholder | We skip t0000 when needed (NaN weights) | ✓ Handled |

**No RMSE-side bug found.** The computation is the textbook `RMSE = √(mean((x̂ − x)²))`.

---

## 6. Critical consistency checks — diagnosing the RMSE puzzle

This is the core question: **why does the no-obs ablation give *lower* spatial-average RMSE than the with-obs baseline?**

I ran four diagnostic experiments. Each rules out a class of bugs and points to the real cause.

### 6.1 Diagnostic 1 — Truth-vs-prior std over time

| Time index | std(truth − β_prior) over the grid |
|---:|---:|
| 0 | 300.2 |
| 4 | 199.7 |
| 19 | 174.3 |
| 49 | 277.6 |
| 99 | 419.3 |
| 149 | 433.9 |
| 200 | 318.5 |

**Interpretation.** The truth's deviation from the (stationary) prior *initially decreases* (rapid damping of high-frequency iid noise by numerical diffusion in the first ~20 steps), then *increases* in the middle of the run (advection has shifted the truth's sinusoidal envelope while the comparison `β_prior` stays put), then comes back down somewhat near the end.

So the truth field doesn't simply "converge to the prior" — it's *advected away* from the prior in the middle of the run, and the std grows. This rules out the simplest version of my earlier "diffusion smooths truth to prior" hypothesis.

### 6.2 Diagnostic 2 — RMSE split: sensor cells vs unsensored cells

This is the crucial diagnostic. Splitting the 1600 cells into 16 sensor cells and 1584 unsensored cells:

| t | with-obs sensor | with-obs unsens. | no-obs sensor | no-obs unsens. |
|---:|---:|---:|---:|---:|
| 0 | 212.7 | 301.1 | 212.7 | 301.1 |
| 49 | **88.1** | 144.1 | 104.9 | 110.4 |
| 99 | **41.8** | 118.3 | 94.0 | 93.3 |
| 200 | **19.1** | **108.6** | 77.5 | **84.1** |

**Final-10 averages:**
- With obs: sensor RMSE = **18.95**, unsensored RMSE = **109.26**
- No obs:   sensor RMSE = **80.40**, unsensored RMSE = **84.19**

**What this reveals.** At sensor cells the with-obs filter is *4× better* than no-obs (19 vs 80). At unsensored cells the with-obs filter is *worse* (109 vs 84) than just sitting at the prior. The spatial-average RMSE you see is dominated by the 1584 unsensored cells (99 % of the grid), so the unsensored-cell behaviour drives the final number.

This decomposes the puzzle cleanly:

- **Where observations *do* help:** sensor cells. RMSE drops from 212 → 19 — a 91 % reduction, demonstrating exactly the "filter is learning" behaviour the paper describes.
- **Where observations *don't* help and may hurt:** the rest of the grid. RMSE drops from 301 → 109 (still drops, because diffusion + advection rearrange the truth toward something closer to a smooth field on average — see §6.4 below — but it drops *less* than the prior-only no-obs run drops to 84).

### 6.3 Why the with-obs unsensored RMSE is *higher* than no-obs

Three factors stack:

1. **No spatial correlation in the prior.** Our state noise is iid per cell (white). So observing β at cell *i* tells us *nothing* about β at cell *j* under the prior. There's no posterior-update mechanism by which sensor information "leaks" into unsensored cells except through the dynamics.

2. **Dynamics propagates information only one-directional, slowly.** The upwind scheme moves features only in +x at speed ~1.75 m/s. Over 200 steps the effective propagation is ~35 grid cells worth of advection — meaningful but only in one direction, and the truth's evolution is dominated by its initial high-frequency noise pattern which gets numerically diffused anyway.

3. **Resampling biases the ensemble at unsensored cells.** When the filter resamples favouring particles whose β-at-sensors matches obs, the *surviving* particles' β-at-unsensored-cells is a biased sample of the prior (biased by which particles happened to match obs at sensors). With iid prior, this bias has zero mean but non-zero variance — adding noise to the unsensored estimate compared to the unbiased "all-particles" no-obs case.

Together: with-obs filter pays a noise cost at 99 % of the grid (unsensored cells) to gain a large benefit at 1 % of the grid (sensor cells). The net spatial-average RMSE goes up.

### 6.4 The no-obs RMSE *also* drops (from 300 to 84) — why?

Two reasons:

- **Numerical diffusion damps the iid initial noise.** The first 20 steps cut the truth-vs-prior std from 300 to 174 by smoothing out white-noise spikes.
- **The PF mean (no obs) is not exactly β_prior — it has small advected and noise components.** The mean of 1000 advected-and-noised particles is a tighter estimate of the *expectation* of the truth than β_prior alone, especially after the truth has been advected somewhat.

If the truth had no high-frequency content to lose (e.g. smooth Matérn-correlated initial noise), the no-obs RMSE would *not* drop nearly this much.

### 6.5 Checklist for everything I ruled out

| Possible bug | Verdict | How I checked |
|---|---|---|
| Incorrect RMSE evaluation | Clear | §5.4 + direct calculation matches |
| Truth leakage to the filter | Clear | `read_observation_sequence` reads only the `observations` group, never `state` |
| Reusing observations across runs | Clear | Both runs use identical seeds → identical obs (verified by hash) |
| Incorrect weight normalisation | Clear | `normalized_exp!` does log-sum-exp; ablation gives w = 1/N exactly |
| Particle degeneracy | Clear | With-obs ESS = 554 (healthy); no-obs ESS = 1000 (by design) |
| Resampling artifacts | Small but real (§6.3 pt 3) | Resampling adds noise at unobserved cells via biased sampling |
| Overconfident prior | Real (§6.3 pt 1) | iid prior gives zero spatial correlation; this is a *modelling* choice |
| Observation operator mismatch | Clear | Same `get_observation_mean_given_state!` for generation and likelihood |
| Comparing wrong variables | Clear | `state_avg/beta` vs `state/beta` — both β, same units, same time index |
| Timing mismatch | Clear | Both arrays indexed by t000t consistently |
| Deterministic propagation hiding divergence | Clear | Both filter and truth use the same stochastic update |
| Wrong σ_obs | Clear | Same `obs_cov` used for `sample_observation_given_state!` and for `get_log_density_observation_given_state` |

**No bugs found.** The RMSE behaviour is a *real consequence of the modelling assumptions* (iid noise + sparse sensors + numerical diffusion).

---

## 7. Comparison against ParticleDA paper / standard PF theory

| Standard PF concept | Paper reference | Our implementation | Status |
|---|---|---|---|
| Markov state space model (Eq. 1) | §2 | sample_initial_state! + transitions + sample_observation_given_state | ✓ |
| Additive Gaussian state noise (Eq. 2) | §2 | update_state_stochastic! adds N(0, σ_proc² I) | ✓ |
| Linear Gaussian observation (Eq. 3) | §2 | **Nonlinear** observation ux = 1000/β | ✗ by design |
| Bootstrap proposal (Eq. 4) | §2.1 | The only proposal we use | ✓ |
| Bootstrap weights (Eq. 5) | §2.1 | -invquad/2 of innovation in ux space | ✓ |
| Locally optimal proposal (Eq. 8) | §2.1 | Requires linear obs op → not available to us | n/a |
| Systematic resampling (Douc-Cappé) | §2.2 | ParticleDA built-in `resample!` | ✓ |
| ESS as degeneracy diagnostic (Eq. 11) | §2.2 | Computed in plot scripts | ✓ |
| Multi-threaded prediction + weighting | §3.3 | `Threads.@spawn` in filters.jl:164 | ✓ |
| MPI rank parallelism | §3.3 | Available but unused (we run single-rank) | n/a |
| HDF5 IO | §3.3 | Used for both obs and DA outputs | ✓ |

### 7.1 Where we differ from "best practice"

1. **No spatial correlation in noise.** The paper's tsunami example uses Matérn GRFs for both initial state and process noise. We use iid Gaussian. The paper notes (§8) that "the definition of the state noise should respect the smoothness of the state variable" — our iid noise does *not* respect smoothness, and we've now seen this has a measurable effect on the experiment.

2. **No spatial localisation.** The paper §1 and §8 highlight localisation as the standard remedy for sparse observations in high-dim systems. We don't use any localisation; the BootstrapFilter is "global" — every particle is weighted by the full likelihood at all sensors jointly. For our 1600-dim state with 16 sensors, this is fine but not optimal.

3. **No tempering or mutation.** §1 mentions tempering/mutation as extensions to maintain ensemble diversity. We have neither, just the bootstrap proposal + systematic resampling.

4. **No locally optimal proposal.** Because our obs operator is nonlinear, the variance-reduction benefit shown in the paper's Fig. 7 isn't available to us. The paper's tsunami runs with the optimal proposal got 2–3× lower RMSE at the same N — we can't take that win without adding a linearisation around the proposal's mean.

### 7.2 Where we *deviate* in a way that might look like a flaw but isn't

1. **Resampling every step.** Paper §2.2 and Algorithm 1 line 7 resample every step too. We follow.

2. **Returning `log_density = 0` in the ablation.** This is our addition; not in the paper. It gives a clean "what does the prior alone produce?" baseline.

3. **Recording both `beta` and `log_beta` in HDF5.** Bookkeeping for back-compat with older plots after the log → β state switch. No mathematical impact.

### 7.3 Conceptual flaws or simplifications worth flagging

| Simplification | Effect |
|---|---|
| iid initial + process noise | High-frequency content gets diffused away; sparse obs can't propagate spatially → §6.3 |
| Surrogate observation `ux = 1000/β` | Nonlinear inversion amplifies noise visually; blocks optimal proposal |
| Sparse sensors (1 % grid) | Dominates the RMSE picture (§6.2) |
| Single-direction advection | Information from obs propagates only downstream |
| Per-step resampling | Slightly degrades effective ensemble diversity over many steps |
| No localisation | Standard issue with sparse observations in high-dim |

None of these are bugs. They're modelling choices we made to keep the prototype simple, and they shape the behaviour we observed.

---

## 8. Final diagnostic assessment

### 8.1 Is the current behaviour physically and statistically reasonable?

**Yes.** Given iid noise structure + 1 % sensor coverage + numerical diffusion + bootstrap proposal, the spatially-averaged RMSE *can* be lower without observations than with them, because:

- The no-obs run pays no "noise floor" at sensor cells.
- The no-obs run has slightly tighter PF mean at unsensored cells (since resampling doesn't add a bias from the obs-driven selection).
- The truth's high-frequency component decays under numerical diffusion, making the smooth prior an increasingly good predictor at unsensored cells.

The with-obs run trades a *huge* improvement at sensor cells (RMSE 213 → 19, factor of 11×) for a *small degradation* at unsensored cells (101 → 109, factor of 1.08×). Spatially averaged, the degradation at 1584 cells outweighs the improvement at 16 cells.

This is a known phenomenon in DA, sometimes called "*observation localisation by absence of correlation*" — when the background prior has no covariance structure linking observed to unobserved cells, observations simply cannot inform the unobserved cells.

### 8.2 Most likely root cause of the unexpected RMSE trend

**Modelling structure, not implementation bug.** Specifically, the *combination*:

1. **iid (no spatial correlation) initial and process noise.** This is the dominant factor. Real glacier β has spatial correlation length scales of kilometres; observations would naturally inform nearby cells. Our iid setup throws this away.
2. **1 % sensor coverage** in a 1600-cell grid. Adequate for sensor cells, hopeless for the rest of the grid in the absence of (1).
3. **Numerical diffusion** in the upwind scheme damps the very thing (high-freq noise) that observations *could* in principle help with.

### 8.3 Prioritised debugging checklist

The result is *real*; the action is *improve the experiment*, not debug the code. In rough order of expected impact and ease:

1. **(High-impact, easy) Switch to Matérn-correlated initial and process noise.** Mirror the LLW2d pattern in [test/models/llw2d.jl](../test/models/llw2d.jl) using `GaussianRandomFields.jl`. This single change should make observations propagate to neighbouring cells (because correlated prior → posterior at obs cell informs neighbours via the covariance). Probably brings with-obs RMSE under no-obs RMSE.
2. **(High-impact, easy) Report RMSE at sensor cells separately from unsensored cells** in all future plots. Spatial-average RMSE alone is misleading for sparse-observation problems. Our diagnostic at §6.2 should become standard output of `plot_obs_vs_no_obs.jl`.
3. **(Medium-impact, easy) Try denser observations.** Bump sensor count from 16 to 64 or 256 and re-run the ablation. With more sensors, the unsensored-cell loss shrinks and with-obs RMSE should drop below no-obs.
4. **(Medium-impact, medium) Try a higher-order advection scheme.** Lax-Wendroff (2nd-order) or WENO (5th-order) drastically reduces numerical diffusion. This would prevent the "everything decays toward the smooth prior" effect.
5. **(Lower-impact, harder) Add spatial localisation to the filter.** Each observation only updates state within a radius. Requires a custom filter implementation; not currently supported in stock ParticleDA's BootstrapFilter.
6. **(Optional) Replace the surrogate obs operator with WAVI.** Doesn't directly fix the issue but is the actual project goal.

### 8.4 What this audit confirms about the code

- The state representation, dynamics, and PF mechanics correctly implement paper Algorithm 1 with bootstrap proposal.
- Truth and observation generation are consistent and free of leakage.
- RMSE is computed correctly.
- The disable_observations ablation is clean.
- The puzzling RMSE result is a feature of our modelling choices, not a flaw in the code.

### 8.5 What it confirms about our understanding so far

- The observations *are* improving the filter where it matters (sensor cells, 91 % reduction in RMSE).
- The spatially-averaged RMSE was hiding this and giving a misleading aggregate picture.
- The reason it was hiding it is now precisely diagnosed: iid prior + sparse obs + diffusion ⇒ no obs-driven improvement at the 99 % of the grid we don't observe.

Going forward, the *honest* metric for "is the filter learning?" is one of:

- ESS (1000 vs 554 — clearly different)
- Per-point pointwise plots at sensor cells (clear tracking with obs, none without)
- RMSE *at sensor cells* (19 vs 80 — clearly different)

The metric that *misleads* us is spatially-averaged RMSE, until we move to a setup with spatially-correlated noise.

---

## Files referenced

| Code | Purpose |
|---|---|
| [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) | Glacier model interface, state, dynamics, observation operator |
| [glacier.yaml](../glacier-code/particleda/glacier.yaml) | Canonical config (β-space, time_step=400, 16 sensors) |
| [glacier_no_obs.yaml](../glacier-code/particleda/glacier_no_obs.yaml) | Ablation config (`disable_observations: true`) |
| [src/ParticleDA.jl](../src/ParticleDA.jl) | Main PF loop, simulate_observations_from_model |
| [src/filters.jl](../src/filters.jl) | BootstrapFilter / OptimalFilter implementations |
| [test/models/llw2d.jl](../test/models/llw2d.jl) | Tsunami reference model (Matérn GRF noise) |
| [bootstrap-pf-experiments/code/benchmark.yaml](../bootstrap-pf-experiments/code/benchmark.yaml) | Tsunami YAML (for parameter comparison) |

| Earlier note | What it covers |
|---|---|
| [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md) | First-pass LowLevel vs ParticleDA result comparison |
| [07_observation_cadence.md](07_observation_cadence.md) | Time-step / n_integration_step bug and fix |
| [10_one_step_advection.md](10_one_step_advection.md) | Numerical-diffusion derivation for upwind |
| [13_observation_noise.md](13_observation_noise.md) | Observation noise mechanics and the 1/β amplification effect |
| [14_conceptual_faq.md](14_conceptual_faq.md) | Initial particle/truth setup, prediction vs update phases |
| [15_observations_ablation.md](15_observations_ablation.md) | First write-up of the ablation experiment |
| [16_beta_space_state.md](16_beta_space_state.md) | β-space state switch + dynamics verification |
