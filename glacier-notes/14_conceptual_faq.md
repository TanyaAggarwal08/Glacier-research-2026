# Conceptual FAQ — initial particles, observations, weights, and real-world cadence

> Four foundational questions about how the particle filter actually works, with concrete answers for **both** our glacier model and the tsunami model. Plain language; jargon explained inline.
>
> Skim §1 then jump to the question you want — they're independent.

---

## 1. Setup recap (so the rest makes sense)

There are three different things at play and they're easy to confuse. Let's name them.

| Name | What it is | Where it lives | When it exists |
|---|---|---|---|
| **The truth** | A single, ground-truth β field that evolves through time. | `glacier_obs.h5 → state/t####/beta` | We make it up via `simulate_observations_from_model`. In a real-data setting it'd be unknown. |
| **The observations** | Noisy measurements of one quantity (here `ux = 1000/β`) at the sensor cells, at each time step. | `glacier_obs.h5 → observations/t####` | Generated from the truth + measurement noise. Real-data setting would supply these. |
| **The particles** | 1 000 candidate β fields the filter carries forward. | In-memory while the filter runs; final mean and variance written to `particle_da.h5 → state_avg/state_var`. | Exist only during the filter run. |

A crucial property: **the filter never sees the truth.** It sees only observations. The truth is used twice — once to *generate* the observations, and once *after the run* to compute RMSE so we can grade the filter's performance.

In a real glacier project there is no "truth" file. You start with observations (InSAR / GPS), run the filter, and get a posterior estimate of β. We're using a synthetic truth just so we can check whether the filter is recovering it.

---

## 2. Q1 — Do particles start at the truth, or are they scattered?

**They are scattered.** Both the truth and every particle are independent random draws from the **same prior** distribution.

### 2.1 What the code does

Look at `sample_initial_state!` in [glacier_model.jl:74-89](../glacier-code/particleda/glacier_model.jl#L74-L89):

```julia
function ParticleDA.sample_initial_state!(
    state::AbstractVector{T}, model::GlacierModel,
    rng::Random.AbstractRNG, task_index::Integer=1,
) where {T<:Real}
    ParticleDA.get_initial_state_mean!(state, model)         # state ← θ_prior
    σ = model.parameters.init_std_theta                       # = 0.30
    @inbounds for i in eachindex(state)
        state[i] += σ * randn(rng)                            # add iid noise
    end
    return state
end
```

This function is called once **per draw**:

- **For the truth:** ParticleDA's `simulate_observations_from_model` calls it once with the `simulate_observations.seed` RNG to make the t=0 truth.
- **For each particle:** ParticleDA calls it 1 000 times (once per particle) with the `filter.seed`-derived RNG to make the initial ensemble.

Every call evaluates to:

```
state(i) = θ_prior(i)  +  0.30 · N(0, 1)
```

where `θ_prior` is the same sinusoidal log-β template `log(1000 + 500 sin(ωx) sin(ωy))` for everyone. The noise is independent for each cell, each particle, and the truth.

So at t = 0:
- The truth is `θ_prior` plus one random ripple of std 0.30.
- Each of the 1 000 particles is `θ_prior` plus a *different* random ripple of std 0.30.
- **Truth and particles are equally far from `θ_prior` on average; truth has no special status.**

### 2.2 What this means in numbers

`init_std_theta = 0.30` in log-β space. So β values for each cell are roughly `exp(log(β_prior) + 0.30·η)` where η ~ N(0,1). That's a 30 % multiplicative spread:

- A cell with β_prior = 1000 has truth and particles scattered roughly across [740, 1350] at 1σ.
- At 3σ tails: [400, 2460].

The 1 000 particles fill that distribution; one or two particles happen to be close to the truth by chance; most are not.

### 2.3 What happens after t = 0

Two things, every filter step:

- The truth evolves via deterministic dynamics (advection by `dt = 400 s`) + a fresh draw of process noise (std `0.007`).
- Each particle evolves the same way: same dynamics, *its own* fresh draw of process noise.

So the particles wander **independently** away from each other, just like the truth wanders away from its own t=0 state. There is no "shared kick" or "all particles drift the same way" — every random draw is independent.

The reason the filter eventually catches up to the truth is **not** that any particle "knows" where the truth is. It's that observations preferentially keep the particles that happened to wander in directions consistent with the data. That's the heart of the next section.

### 2.4 Same story for tsunami

LLW2d does the same thing structurally ([test/models/llw2d.jl:248-264](../test/models/llw2d.jl#L248-L264)) — `sample_initial_state!` adds a Matérn random-field perturbation to `θ_prior`. The truth and every particle are independent draws. Only difference: noise is *spatially correlated* (smooth ripples) instead of iid (white ripples like ours). Same independence story between truth and particles.

---

## 3. Q2 — The observations look far from the true β — is that broken?

**No, and this was the topic of [13_observation_noise.md](13_observation_noise.md).** Short recap because it's worth understanding clearly:

### 3.1 Noise is added in ux units, not β units

The line is `y(k) = ux_true(k) + σ_obs · η`, where η ~ N(0, 1) and `σ_obs = 0.10`. With ux ≈ 1 typically, that's a clean **±10 % noise**.

### 3.2 The β plot exaggerates the noise

When we plot observations on a β-equivalent axis (`β = 1000 / ux`), the 1/x inversion amplifies and skews the apparent noise. A noise event of size 0.10 in ux space turns into a β-equivalent error that's:

| Sign of noise | β-equivalent error |
|---|---|
| ux too low by 0.10 | β estimate too high by +111 |
| ux too high by 0.10 | β estimate too low by −91 |
| ux too low by 0.20 | β estimate too high by +250 |
| ux too high by 0.20 | β estimate too low by −167 |

So in the β plot the crosses look spread because the visual axis exaggerates the noise. In the **ux plot** ([ux_first_sensor.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/ux_first_sensor.png)) the crosses cluster cleanly within ±0.10 of the truth ux line — that's the honest picture.

### 3.3 "Won't the filter get confused?"

This was the user's specific worry: *"if observations look very far, the particle won't be able to understand whom to give more weight."*

Two reasons it works fine:

1. **The filter works in ux space, not β space.** When ParticleDA computes the likelihood for particle p, it asks: "Does the noisy observation y match this particle's predicted ux?" The match is in ux units. So the cosmetic distortion of the β plot is invisible to the filter — it never inverts the observation.

2. **σ_obs is calibrated to be informative without being too tight.** At σ_obs = 0.10, particles whose ux differs from the observation by ~0.2 (i.e. ~2σ) get likelihood about `exp(-(2)²/2) ≈ 0.14` — noticeably lower than the closest particle. Particles whose ux differs by ~0.5 (5σ) get `exp(-(5)²/2) ≈ 4·10⁻⁶`. So there's a clear "good vs bad" gradient. We picked 0.10 deliberately so this gradient was steep enough to inform the weights but not so steep that one particle wins everything.

### 3.4 Tsunami's observations look much tighter on its plots — why?

Tsunami observes height directly. Its `obs_noise_std` is 0.01 m (1 cm of wave noise on waves that peak at 30 m). That's a 0.03 % relative error — much tighter than our 10 %.

But tsunami's identity observation operator means there's no inverse-function amplification: an observation noisy by 0.01 m looks noisy by 0.01 m on any plot. Our 10 % relative noise in ux gets exaggerated into 10-25 % visual spread in β because of the inversion. **Different observation operators, different visual stories — same statistical structure.**

---

## 4. Q3 — Do particles give weight to observations or to the true β?

**To observations.** The truth is invisible to the filter; only observations matter. Here's the loop explicitly.

### 4.1 The actual update loop, one filter step

```
For each particle p in 1..1000:
    state[p] ← update_state_deterministic!(state[p])   # advect 400 s
    state[p] ← update_state_stochastic!(state[p])      # add process noise

# Read this filter step's observation y from glacier_obs.h5
y = observations[t]

# Compute each particle's likelihood: how well does its predicted ux match y?
for p in 1..1000:
    ux_predicted = surrogate_ux(state[p])[sensor_indices]
    log_w[p] = - sum((y - ux_predicted) .^ 2) / (2 * σ_obs²)

# Normalise so weights sum to 1
w = exp(log_w - max(log_w))
w = w / sum(w)

# Resample: 1000 indices drawn from the categorical(w) distribution
indices = sample(1..1000, prob=w, count=1000, replace=true)
state ← state[indices]
```

Two things to notice:

1. **`y` (the observation) is the only data input.** The truth never appears in this loop. Even though we have access to the truth in `glacier_obs.h5 → state/`, the filter doesn't read it.
2. **Weights are higher for particles whose predicted ux matches the observation closely.** This is purely a comparison of two ux vectors — the observation y, and each particle's prediction. Particles with predictions far from y get exponentially small weights.

### 4.2 What "correcting itself" means

The PF mean β (in `state_avg/t####/beta`) is `Σ_p w_p · particle_β_p`. After resampling, the weights are uniform (= 1/N) and the cloud is concentrated on the previously high-weight particles. So in practice, "the filter is correcting itself" means:

1. Each step, particles whose predictions match the observation get duplicated.
2. The duplicates then evolve independently (different noise draws).
3. Over many steps, the *only* particles that survive are the ones whose β fields keep producing observations consistent with the data.
4. Since the data is itself (truth + small noise), surviving particles cluster around the truth without any particle ever being *shown* the truth.

This is the magic of the particle filter: **it converges to the truth through the observations alone.** The observations are the "teacher"; the truth is hidden behind a noisy curtain that the filter has to peer through statistically.

### 4.3 If we removed the observations entirely

The filter would still have a `predict` step (particles drift via dynamics + noise), but no `update` step. Particles would diffuse outward from their starting positions, each one a freely wandering β field. The PF mean would basically stay near `β_prior` for the whole run, with growing posterior variance.

Try the thought experiment: what would the RMSE plot look like? It would *grow* over time (truth wanders too, but in a different random direction), not shrink. The fact that our [rmse_beta.png](../glacier-code/particleda/results/run04_grid_aligned_16obs/rmse_beta.png) shrinks from ~370 to ~115 is direct evidence that observations are doing real work.

### 4.4 So what are observations "telling" the particles?

Roughly: "Among your 1 000 candidate β fields, the ones that would have predicted *this* ux at *these* sensor locations get to survive to the next round. The others are deleted."

The observations are not "directly correcting" any particle. They're filtering out particles that disagree with reality. Over many steps, only realistic particles survive, and the PF mean ends up close to the truth.

The tsunami filter works exactly the same way — same loop, same logic, different physics and observation operator.

---

## 5. Q4 — Can we make the filter update every hour, like real observations?

**Yes, easily.** Real glacier observations come at hourly-to-weekly cadences. Our current 400 s cadence is unrealistically fast — that was a choice driven by toy-model convenience. To switch to an hourly cadence we change two parameters in the YAML; we don't change a single line of code.

But there's a CFL consequence, and that's what `n_integration_step` is for.

### 5.1 The numbers

Constants for our toy:
- `dx = 4000 m`
- `v_max ≈ 1.75 m/s` (at typical β ≈ 1500)
- CFL stability ceiling (with our 0.2 safety factor): `dt_safe ≈ 457 s`

Target cadences:

| Real-data analogue | `time_step` | minimum `n_integration_step` for safety | internal `dt` |
|---|---:|---:|---:|
| 1 minute (GPS sample) | 60 s | 1 | 60 s |
| 6.67 minutes (current canonical) | 400 s | 1 | 400 s |
| **1 hour (typical GPS interval)** | **3 600 s** | **9** (gives dt ≈ 400 s) or **16** (gives dt ≈ 225 s, safer) | **400 s** (or 225) |
| 1 day (cadence of cleaned daily products) | 86 400 s | 216 (dt = 400) or 432 (dt = 200) | 400 (or 200) |
| 6 days (Sentinel-1 InSAR repeat) | 518 400 s | 1 296 (dt = 400) | 400 |
| 12 days (Sentinel-1 long-baseline) | 1 036 800 s | 2 592 (dt = 400) | 400 |

Formula:

```
min n_integration_step  =  ceil(time_step / dt_safe)
                         =  ceil(time_step / 457)
```

That keeps internal `dt = time_step / n_integration_step` under the CFL ceiling.

For an hourly cadence: `time_step = 3 600`, `n_integration_step = 9`. Internal dt = 400 s, matching what the canonical setup already uses for stability. CFL satisfied.

Reasonable practical choice: `n_integration_step = 16` for a 2× safety margin (internal dt = 225 s).

### 5.2 What changes in behaviour with a longer cadence?

Three things, in order of how much they matter:

#### (a) The truth moves more between observations

At dt = 400 s the truth advects by `v · dt = 1.75 · 400 = 700 m` per sub-step. At 9 sub-steps per filter step, that's `6 300 m` per filter step — about **1.6 grid cells** of motion between observations.

For comparison, the canonical run (1 sub-step) moves the truth by only 0.18 cells per filter step. So at hourly cadence the truth has visibly moved between observations; the filter must extrapolate further with each `predict` step before the next `update`.

#### (b) Process noise accumulates

`update_state_stochastic!` is called once per filter step (independent of `n_integration_step`). So with one filter call per hour, β receives `σ_proc = 0.007` of noise per *hour*, not per sub-step. In effect, slower observation cadence means slower information injection into β.

If you wanted to keep "process noise per unit time" constant when comparing different cadences, you'd scale: `σ_proc_per_filter_step = σ_proc_base · √(time_step / time_step_base)`. We haven't done that yet — it would be the next refinement if hourly runs look too noisy.

#### (c) Wall-clock cost goes up linearly with `n_integration_step`

Currently 1 sub-step takes ~15 s of wall-clock for the whole run. At 9 sub-steps per filter step, expect ~120 s. At 16 sub-steps, ~240 s. Still very fast — our toy is cheap. For WAVI it'd matter more.

### 5.3 What about realistic observation noise at that cadence?

A real hourly GPS reading has its *own* noise level (~1-5 m/yr in surface velocity), which would translate to a different `obs_noise_std` than our current 0.10. We'd want to re-tune σ_obs to match the real instrument.

For an InSAR product (~10-50 m/yr surface velocity noise), σ_obs would need to be larger. Pressure budget then dictates how many sensors / how loose the prior need to be to keep ESS healthy.

### 5.4 How does this compare to tsunami?

Tsunami's `time_step = 5 s` is realistic for a tsunami warning system: ocean wave height observations from coastal gauges or DART buoys can be sampled every few seconds. Their CFL ceiling is ~5 s, so they need `n_integration_step = 10` to fit 10 sub-steps of 0.5 s under the ceiling.

The same pattern applies to us at longer cadences. Just the numbers differ:

| | Tsunami | Glacier (canonical) | Glacier (hourly) |
|---|---|---|---|
| Filter cadence (`time_step`) | 5 s | 400 s | 3 600 s |
| Internal dt | 0.5 s | 400 s | ~400 s |
| `n_integration_step` | 10 | 1 | 9 |
| Why this `n_integration_step`? | Tsunami CFL ≈ 5 s; safety factor → ≤ 0.5 s | Glacier CFL ≈ 457 s; 400 s already safe | Glacier CFL ≈ 457 s; 3 600 / 9 ≈ 400 s |

So **the design pattern is the same**: pick `time_step` to match real observation cadence, then pick `n_integration_step` so internal dt is under your CFL ceiling.

### 5.5 Want to actually try it?

Single YAML edit. Create `glacier-code/particleda/glacier_hourly.yaml`:

```yaml
filter:
  nprt: 1000
  verbose: true
  output_filename: "glacier-code/particleda/results/run06_hourly_cadence/particle_da.h5"
  seed: 42

model:
  glacier:
    nx: 40
    ny: 40
    x_length: 160000.0
    y_length: 160000.0
    station_filename: "glacier-code/particleda/stations_random_16.txt"
    init_std_theta: 0.30
    process_std_theta: 0.007
    obs_noise_std: 0.10
    advection_epsilon: 0.0005
    n_integration_step: 16          # ← was 1
    time_step: 3600.0               # ← was 400.0 (one hour now)

simulate_observations:
  seed: 123
  n_time_step: 200                  # 200 hours = 8.3 days of model time
```

Then:

```bash
mkdir -p glacier-code/particleda/results/run06_hourly_cadence
julia --project=test glacier-code/particleda/run_glacier_pda.jl  glacier-code/particleda/glacier_hourly.yaml
julia --project=test glacier-code/particleda/plot_glacier_pda.jl glacier-code/particleda/results/run06_hourly_cadence
```

Wall-clock budget: ~240 s. Say the word and I'll run it.

What we'd expect to see:
- ESS still healthy (same physics, same filter hygiene)
- RMSE convergence slightly different shape — fewer observations per unit model time so it might take longer to converge in wall-clock, but each one informs more model motion
- A CFL warning might fire if β briefly excursions upward and pushes `v_max` above 1.85 m/s (in which case bump `n_integration_step` to 24 and rerun)

---

## 6. Quick summary table

| Question | Plain answer |
|---|---|
| Q1: Do particles start at truth? | No. Truth and each of the 1 000 particles are independent random draws from the same prior. Particles do not "know" where the truth is at t = 0. |
| Q2: Why do obs look so far from truth in the β plot? | Visual artefact of the 1/β inversion. In ux space (the native space) observations are tightly within ±0.10 of truth, as designed. |
| Q3: Do particles weight by truth or by observations? | Observations. The filter never sees the truth — it sees only noisy observations and weights particles by how well their predicted observations match the noisy data. |
| Q4: Can we run at hourly cadence? | Yes — set `time_step = 3600`, `n_integration_step ≥ 9` (so internal dt stays under CFL ceiling). For longer cadences (daily, weekly) bump `n_integration_step` proportionally. Same pattern tsunami uses, different numbers. |

---

## 7. Glossary additions

| Term | Meaning |
|---|---|
| **Prior** | Our belief about β *before* any observation. Encoded as `θ_prior + σ_init · N(0, 1)`. Both truth and particles are drawn from it. |
| **Posterior** | Our belief about β *after* incorporating observations. The filter's PF mean approximates the posterior mean. |
| **Predict step** | Advance each particle via dynamics + process noise. Doesn't use observations. |
| **Update step** | Weight particles by their match to the current observation; resample. The "data assimilation" step. |
| **Likelihood** | A measure of how plausible an observation is given a candidate state. For Gaussian noise: `exp(-‖y − h(x)‖² / (2 σ_obs²))`. Higher = state explains data better. |
| **Filter cadence** | How often the filter does an update step. Set by `time_step`. |
| **Physics cadence** | How often the dynamics advance internally. Set by `time_step / n_integration_step`. Constrained by CFL. |
| **Realistic cadence** | Matching the real-world rate at which observations arrive — usually hours to days for glaciers, seconds for tsunamis. |
