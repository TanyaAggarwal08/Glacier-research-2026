# Parameters — what they mean, why we chose them, how they compare

> **What this note answers.**
> 1. *Was the LowLevel filter "wrong" too if the natural dt is 400 s?* — **No, and the answer is more interesting than that. Read §1.**
> 2. *What does `dt = 400` actually do in the code, step by step?* — §2.
> 3. *Are our ParticleDA parameters the same as LowLevel's? Where they differ, why?* — §3.
> 4. *What equations is the model actually solving?* — §4.
> 5. *How does this compare to the tsunami benchmark and why are the numbers so different?* — §5.
> 6. Glossary at the end (§6).
>
> This is intended to be the single page you re-read whenever the parameter names start to blur.

---

## 1. Was LowLevel wrong?

**Short answer: no.** Both filters were doing the same thing per step, just with different surface vocabularies. The "bug fix" was about exposing a configuration knob, not changing the underlying physics.

### How both filters actually computed dt

Open up [particlefilteringwithiceflow.jl:160-161](../glacier-code/testing_stage/particlefilteringwithiceflow.jl#L160-L161). Inside the `dynamics` function the very first thing it does each call is:

```julia
max_speed = maximum(1 .+ ε .* β)
dt = 0.2 * dx / max_speed   # CFL condition
```

That formula gives `dt ≈ 0.2 × 4000 / 1.75 ≈ 457 s`. Then it advects β by that dt and returns the new state. **Each call to LowLevel's `dynamics` = one CFL-derived sub-step ≈ 457 s of model time.**

Now look at our **pre-bug-fix** ParticleDA glacier model. The body of `update_state_deterministic!` had the same code copied across:

```julia
max_speed = maximum(1 .+ ε .* β)
dt = 0.2 * dx / max_speed
```

Same formula. Same ~457 s. The YAML parameter `time_step: 1.0` was silently ignored because the code never read it.

So **LowLevel and pre-fix ParticleDA were numerically identical** in their per-step dynamics. Both used the CFL formula to compute dt internally. Both advanced ~457 s per call. There was nothing physically "wrong" with either.

### So what was the bug?

The bug was a **mismatch between what the YAML claimed and what the code did**, not between what the code did and what physics requires.

ParticleDA's design pattern (inherited from the tsunami model LLW2d) is:
- The YAML's `time_step` says "this is how much model time one filter step covers."
- The YAML's `n_integration_step` says "split that time into this many smaller physics sub-steps."
- The model code is supposed to compute `dt = time_step / n_integration_step` and use that.

Our code wasn't doing that — it was using its own CFL formula and ignoring the YAML. So writing `time_step: 5.0` in the YAML gave you the *same* result as writing `time_step: 1000.0` or `time_step: 1.0`. Confusing, brittle, and broken once we wanted to experiment with cadence.

LowLevel has **no equivalent of this bug** because LowLevel has no `time_step` config in the first place. You call its `dynamics` and you get one CFL sub-step. There's no separation between "physics step" and "filter step" — they are the same thing. It's a simpler design, with the trade-off that you can't decouple them when you want to.

### What changed in v2

The bug fix made `update_state_deterministic!` honour the YAML:

```julia
dt = p.time_step / p.n_integration_step
```

With `time_step = 400.0` and `n_integration_step = 1` in the canonical YAML we now use **dt = 400 s** — chosen deliberately to (a) be safely below the CFL ceiling of ~457 s and (b) be close enough to LowLevel's effective dt that we can compare results without worrying about physical-scale differences.

The seed sweep in [08_seed_sweep_v2.md](08_seed_sweep_v2.md) confirms: at `time_step = 400`, ParticleDA's behaviour is within seed-noise of what we saw on the buggy code (which was using 457). So the bug fix is cosmetic when used at the right physical scale.

**Bottom line.** LowLevel is fine. Pre-fix ParticleDA was fine in the *physics*, broken in its *interface*. Post-fix ParticleDA does the right physics and lets you choose the cadence from YAML.

---

## 2. What does `dt = 400 s` actually do in the code?

Let's trace one filter step at the current canonical config (`time_step = 400, n_integration_step = 1, seed = 42`).

### 2.1 The big picture

`run_particle_filter` does, in a loop over `n_time_step = 200`:

```
for t in 1..200:
    for each of the 1000 particles in parallel:
        update_state_deterministic!(particle.state, model, t)  # the bit we're tracing
        update_state_stochastic!(particle.state, model, rng)
    weights = compute likelihoods from the new observation
    if needed, resample the ensemble
    write HDF5 snapshot
```

So "one filter step" means: every particle gets pushed forward through one call to `update_state_deterministic!`, then noise is added, then weights are recomputed against the just-arrived observation.

### 2.2 Inside `update_state_deterministic!`

In [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) lines 97–141 the function does, for one particle:

1. **Read the parameters.** `nx = ny = 40`, `dx = 160000 / 40 = 4000 m`, `ε = 0.0005`, `time_step = 400`, `n_integration_step = 1`.
2. **Compute `dt = time_step / n_integration_step = 400 / 1 = 400 s`.** This is the size of the internal physics sub-step.
3. **One-time CFL check.** Only on the very first call across the whole run (`_CFL_WARNED[]` flag):
   - Computes the current max wave speed: `max_speed = max(1 + ε·β) ≈ 1.75` for β around 1500.
   - Computes the stability ceiling: `cfl_safe = 0.2 × dx / max_speed ≈ 0.2 × 4000 / 1.75 ≈ 457 s`.
   - Warns once if `dt > cfl_safe`. At 400 vs 457 we're just under, with margin.
4. **Convert state to β.** The state vector holds θ = log β, so `β = exp(θ)` first.
5. **Run `n_integration_step = 1` sub-step** (just one, for our config — but if it were 2, the loop body below would run twice with the same dt):

   For each grid cell `(j, i)` in the 40×40 grid:

   ```
   # neighbour to the left (periodic boundary)
   im = i - 1, wrapping around if i == 1

   # spatial derivative (upwind)
   dβ/dx = (β[j,i] - β[j,im]) / dx

   # local advection velocity
   v = 1 + ε × β[j,i]

   # forward Euler step in time, upwind in space
   β_new[j,i] = β[j,i] - v × dt × dβ/dx
   ```

   So each cell's new β depends on its own value, its left neighbour, the local velocity, and dt.

6. **Convert back.** `state .= log(β_new)`.

That's it. **dt = 400 s** means the upwind step formula advances β by 400 s' worth of advection.

### 2.3 Why dt matters — what would happen at other values

| dt (per sub-step) | Effective motion per filter step | Filter outcome |
|---:|---|---|
| 1 s | `v·dt ≈ 1.5 × 1 = 1.5 m` (out of dx = 4000 m). β essentially **frozen**. | Filter has no signal to track. Locks onto one particle on step 1. RMSE never improves. (This is what happened in our first attempt — see [07_observation_cadence.md §2](07_observation_cadence.md).) |
| 400 s (canonical) | `v·dt ≈ 1.5 × 400 = 600 m` ≈ 15 % of a grid cell. β advects meaningfully. | Filter tracks truth, RMSE 330 → 115 over 200 steps. Healthy. |
| 800 s (every-2c variant) | `v·dt ≈ 1200 m` ≈ 30 % of a grid cell. Even more motion per step. | Still stable (still under CFL). Slightly better ESS because particles have more spread between obs. |
| 600 s | Just over CFL. Numerical wiggles start. | Risk of nonphysical β fluctuations. Would need `n_integration_step ≥ 2` to stay safe. |
| 5000 s | Way over CFL. | The upwind scheme blows up: β starts producing negative or huge values, exp/log explodes. |

So picking dt is a balance: large enough that the truth moves between observations, small enough that the numerics stay stable.

### 2.4 The CFL condition in plain words

The CFL (Courant–Friedrichs–Lewy) condition is a stability rule for explicit time-stepping of advection problems. It says:

> Information should not travel more than one grid cell per time step.

If `v` is the advection speed, `dx` is the grid spacing, and `dt` is the time step, the condition is `v · dt < dx`. Adding a safety factor (we use 0.2) gives `dt < 0.2 · dx / v`. Violate it and the numerical scheme amplifies tiny errors into infinity — your simulation explodes.

For our toy:
- `v_max = max(1 + ε · β) ≈ 1.75` (at β = 1500)
- `dx = 4000 m`
- CFL ceiling: `dt ≤ 0.2 × 4000 / 1.75 ≈ 457 s`

Our chosen `dt = 400 s` sits 14 % below the ceiling, which is comfortable. The one-shot warning catches any future config that crosses the line.

---

## 3. Parameter-by-parameter comparison

### 3.1 Full list of our current canonical config

From [glacier.yaml](../glacier-code/particleda/glacier.yaml) (the v2 / canonical setup):

| Block | Parameter | Value | Meaning in plain English |
|---|---|---:|---|
| `filter` | `nprt` | 1000 | Number of particles (parallel guesses) the filter carries. |
| | `verbose` | true | Write HDF5 snapshots after each filter step. |
| | `output_filename` | … | Where the filter's HDF5 goes. |
| | `seed` | 42 | RNG seed for resampling + process-noise draws. |
| `model.glacier` | `nx`, `ny` | 40, 40 | Grid is 40×40 cells = 1600 total. |
| | `x_length`, `y_length` | 160 000 m each | Physical domain size = 160 km × 160 km. So each cell is 4 km × 4 km. |
| | `sensor_stride` | 100 | Every 100th flat cell index is a sensor → 16 sensors. |
| | `init_std_theta` | 0.30 | Initial particle spread in log β: each particle's log β starts within ≈ 30 % of the prior mean. |
| | `process_std_theta` | 0.007 | Per-step "kick": add Gaussian noise of std 0.007 to log β in every cell every step (~0.7 % of β). |
| | `obs_noise_std` | 0.10 | Std of the noise added to observed velocities. ux ≈ 1, so this is ~10 %. |
| | `advection_epsilon` | 0.0005 | The ε in `v = 1 + ε·β`. Controls how strongly β feeds back into its own advection speed. |
| | `n_integration_step` | 1 | One physics sub-step per filter step. |
| | `time_step` | 400.0 | One filter step covers 400 s of model time → internal dt = 400 s. |
| `simulate_observations` | `seed` | 123 | RNG seed for generating the synthetic truth + observations. |
| | `n_time_step` | 200 | Run for 200 filter steps = 80 000 s total. |

### 3.2 LowLevel reference (`particlefilteringwithiceflow.jl`)

| Parameter | LowLevel value | ParticleDA v2 value | Same? | Why different? |
|---|---:|---:|---|---|
| `Nx`, `Ny` | 40, 40 | 40, 40 | ✅ | — |
| `L` (domain) | 160 000 m | 160 000 m | ✅ | — |
| `dx` | 4000 m | 4000 m | ✅ | — |
| `ε` (advection_epsilon) | 0.0005 | 0.0005 | ✅ | — |
| `β_prior` formula | `1000 + 500·sin(ωx)·sin(ωy)` | identical | ✅ | — |
| `process_noise_std_θ` | 0.007 | 0.007 | ✅ | — |
| `Np` (particles) | 10 000 | 1 000 | ❌ | We started at 200 (run-01), found Np-floor matters less than σ_obs, settled at 1000 as the sweet spot between absolute ESS and compute cost. LowLevel can afford 10 000 because each step is essentially free in that codebase. |
| `T` (filter steps) | 100 | 200 | ❌ | We run twice as long to see clearer convergence; not a fundamental difference. |
| `sensor_stride` | 16 | 100 | ❌ | LowLevel has 100 sensors → high pressure budget. We dropped to 16 in run-03 to bring pressure from ~900 down to ~144 — see [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md). |
| `vel_noise_std` (= `obs_noise_std`) | 0.05 | 0.10 | ❌ | We doubled it for ensemble health (run-02 change). LowLevel got away with 0.05 because Np = 10 000 — at our Np = 1 000 the tighter likelihood drove ESS into the floor. |
| `init_std_θ` | 0.05 | 0.30 | ❌ | We widened by 6× (run-02 change). The LowLevel toy with 0.05 had σ_init ≈ σ_obs, which forces all particles into a narrow band where the likelihood barely discriminates. Wider prior = healthier ensemble. |
| `dt` | CFL formula → ≈ 457 s | YAML → 400 s | ≈ | Same physical scale; ParticleDA's is just exposed in the config now. |
| `rng_seed` | 1234 | filter 42 / truth 123 | ❌ | Arbitrary; we vary across seed sweeps. |

**Take-away.** The *physics-defining* parameters (grid, domain, ε, β_prior, process noise) are identical. The *filter-hygiene* parameters (Np, init spread, obs noise, sensor count) are different — and each difference came from a specific lesson in earlier notes:

- `init_std_theta: 0.05 → 0.30` from [05](05_particleda_vs_lowlevel_comparison.md) ("wider σ_init is the LowLevel-style hygiene we should adopt").
- `obs_noise_std: 0.05 → 0.10` from [05](05_particleda_vs_lowlevel_comparison.md) (small obs-noise inflation matches LowLevel's 10 %-of-signal target).
- `nprt: 200 → 1000` from [05](05_particleda_vs_lowlevel_comparison.md) ("Np scales the absolute ESS floor; needed for usable uncertainty").
- `sensor_stride: 16 → 100` (i.e. 100 obs → 16 obs) from the pressure-budget reduction.
- `time_step: 1.0 → 400.0` from [07](07_observation_cadence.md) (the dt-bug fix needed a sensible value).

So the parameter differences aren't arbitrary — they're the documented record of what we learned along the way.

---

## 4. The backend equations

### 4.1 State and forward model

State vector per particle: θ = log β, a flat array of length 1600 (40×40).

**Deterministic dynamics** (`update_state_deterministic!`). At each sub-step of size dt:

For each cell (j, i):
```
v(j,i)  = 1 + ε · β(j,i)
∂β/∂x   ≈ (β(j,i) − β(j,i−1)) / dx     (upwind, periodic in x)
β'(j,i) = β(j,i) − v(j,i) · dt · ∂β/∂x
```

In continuous form this is the nonlinear advection PDE:

```
∂β/∂t + (1 + ε β) · ∂β/∂x = 0
```

This is essentially **Burgers' equation with a constant offset**: it's a wave equation where the wave moves faster wherever β is bigger. Two consequences:

- Smooth profiles get **sharpened** over time (faster bits catch up to slower bits) — that's why glaciology calls this nonlinear sharpening.
- Eventually shocks form. We don't run long enough for shocks to dominate, but the upwind discretisation copes with mild steepening.

There's no y-direction transport in our toy — only x. That's fine for a stress test; the spatial coupling through process noise and observations is enough to make the inverse problem non-trivial.

**Stochastic dynamics** (`update_state_stochastic!`). After the deterministic step:

```
θ(j,i) ← θ(j,i) + σ_proc · ξ          where ξ ~ N(0, 1) independent per cell
```

So each cell gets an independent Gaussian kick of std 0.007 in log β space (= ≈ 0.7 % of β per step). No spatial correlation. The LLW2d tsunami model uses Matern Gaussian random fields here for spatially-correlated noise; we use iid for simplicity. Could swap later.

### 4.2 Observation model

**Forward measurement** (`get_observation_mean_given_state!`):

```
ux(i)   = 1000 / (β(i) + 1e-6)        (surrogate ice-flow speed)
y(k)    = ux(sensor_indices[k])       for k = 1..16
```

Inverse relationship: high β (sticky bed) ⇒ slow flow; low β (slippery bed) ⇒ fast flow. The `+1e-6` guards against division by zero if a particle ever pushes β near zero (it won't under log-β parameterisation, but defensive coding).

**Observation noise** (`sample_observation_given_state!`):

```
y_observed(k) = ux(sensor_indices[k]) + σ_obs · η(k)    η ~ N(0,1)
```

independent across sensors.

**Likelihood** (`get_log_density_observation_given_state`):

```
log p(y | x) = − ‖y − h(x)‖² / (2 σ_obs²) + const
```

Standard isotropic Gaussian log-density. Returned without the constant (only differences in log-likelihood matter for weights).

### 4.3 Particle filter itself (handled by ParticleDA)

ParticleDA's `BootstrapFilter`, in each filter step, does:

```
For each particle p in 1..Np:
    state[p] ← update_state_deterministic!(state[p])
    state[p] ← update_state_stochastic!(state[p])
    log_w[p] = get_log_density_observation_given_state(y, state[p])

# normalise
w = exp(log_w − max(log_w))
w = w / sum(w)

# resample EVERY step (no adaptive threshold in current ParticleDA)
indices = sample(1..Np, prob=w, count=Np, replace=true)
state ← state[indices]
```

ESS is computed from the weights **before** resampling: `ESS = 1 / Σ w²`. That's what gets logged in the HDF5.

---

## 5. Why are our numbers so different from the tsunami benchmark?

The tsunami benchmark from [bootstrap-pf-experiments/](../bootstrap-pf-experiments/) uses the LLW2d (Linear Long Wave 2-D) model. Its YAML lives at [bootstrap-pf-experiments/code/benchmark.yaml](../bootstrap-pf-experiments/code/benchmark.yaml).

### 5.1 Side-by-side

| | LLW2d tsunami | Glacier surrogate (v2) |
|---|---|---|
| **Physics** | Shallow-water equations (linear) | Nonlinear advection (Burgers-like) |
| **State variables per cell** | 3 (height h, x-velocity u, y-velocity v) | 1 (β) |
| **Grid** | 51 × 51 | 40 × 40 |
| **Domain** | 200 km × 200 km | 160 km × 160 km |
| **dx** | ≈ 4 km | 4 km |
| **Wave speed** | √(g · depth) ≈ 170 m/s at 3 km ocean depth | 1 + ε · β ≈ 1.75 m/s |
| **CFL ceiling** | dx / wave ≈ 4000/170 ≈ 24 s | 0.2 × dx / wave ≈ 457 s |
| **YAML `time_step`** | 5 s | 400 s |
| **YAML `n_integration_step`** | 10 | 1 |
| **Internal dt** | 0.5 s (10 sub-steps of 0.5 s each = 5 s per filter step) | 400 s (one sub-step) |
| **`obs_noise_std`** | 0.01 m (height in metres) | 0.10 (dimensionless ux) |
| **Process noise structure** | Matern Gaussian random field (spatially correlated) | iid Gaussian per cell |
| **Initial noise structure** | Matern GRF | iid Gaussian per cell |
| **Observation operator** | Identity (read height directly at gauge cells) | Nonlinear (`ux = 1000/β`) |
| **n_time_step** | 250–260 | 200 |
| **Total model time** | 260 × 5 = 1300 s ≈ 22 minutes | 200 × 400 = 80 000 s ≈ 22 hours |

### 5.2 Why these differences exist

**Wave speed gap (170 vs 1.75 m/s).** That's the central difference. Shallow-water waves propagate at √(g·h) which for a 3 km deep ocean is about 170 m/s — a tsunami crosses our domain in ~20 minutes. β advection on a glacier "wave" velocity of 1.75 m/s means the same domain takes ~25 hours to cross. *Everything else flows from this.*

**CFL ceiling gap (24 vs 457 s).** Direct consequence of the wave-speed gap. Faster waves need smaller dt for CFL. CFL ceiling is inversely proportional to wave speed.

**Filter step (5 vs 400 s).** Both chosen at roughly 20 % of CFL ceiling, giving ~5× safety margin. Tsunami: 5 s = 21 % of 24 s ceiling. Glacier: 400 s = 87 % of 457 s ceiling. We're closer to the line, which is why the one-shot warning fires occasionally.

**`n_integration_step` gap (10 vs 1).** Tsunami chose `time_step = 5 s` (observation cadence) but the model needs dt = 0.5 s internally. So it splits each filter step into 10 sub-steps. Glacier's `time_step = 400 s` already satisfies CFL, so no splitting needed.

**Observation operator gap (identity vs 1/β).** Tsunami measures wave height directly — the most informative possible observation. Glacier measures *surface velocity*, which depends inversely and nonlinearly on β. So our inverse problem is fundamentally harder per observation. That's part of why we had to inflate σ_obs and reduce sensor count to keep the filter healthy.

**Process noise gap (GRF vs iid).** Tsunami uses Matern GRFs because shallow-water solutions are spatially smooth and uncorrelated noise would look unphysical. Our toy is also spatially smooth, so swapping to GRFs would be more realistic — flagged for a future iteration (see [05 §What to change next](05_particleda_vs_lowlevel_comparison.md)).

**State dimension gap (3 fields vs 1).** Tsunami needs height + 2 velocity components because it's solving a 2nd-order wave equation as a first-order system. Our 1st-order advection only needs β. So the tsunami state vector is 3 × 51 × 51 = 7803, ours is 40 × 40 = 1600.

### 5.3 What this means for "is glacier hard or easy?"

Our toy is *easier* in some respects:
- 1 state variable per cell (vs 3).
- Smaller state vector (1600 vs 7803).
- More forgiving CFL.

But *harder* in others:
- Nonlinear observation operator (sensitivity = `-1000/β²`, which is small when β is large, i.e. the very sensors most useful for sticky regions are least sensitive).
- We have to *invert* the observation to learn β (in tsunami you observe state directly).
- Fewer informative observations per particle update.

The pressure-budget formula (`n_obs × (σ_signal/σ_obs)²`) treats these uniformly, which is why our run-03 tuning (16 obs, σ_obs = 0.10) gave a pressure of ~144 — comparable to what the tsunami benchmark sweep needed for healthy ESS. The numbers translate even though the physics doesn't.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| **β (beta)** | The basal friction field. What we're trying to infer. Units of Pa·s/m. |
| **θ (theta)** | log β. We use this as the state because it keeps β positive automatically and makes the prior roughly Gaussian. |
| **ux** | Glacier surface velocity in the x direction. What we (synthetically) observe. |
| **dt** | Numerical time step. The interval the upwind scheme advances β by in one call. |
| **dx** | Grid spacing. Distance between adjacent cells. Here 4000 m. |
| **CFL** | Courant–Friedrichs–Lewy condition. The stability rule `dt · v ≤ dx` for explicit advection. Violate it and the scheme explodes. |
| **CFL ceiling / cfl_safe** | The largest dt the CFL condition allows for current state. We add a 0.2 safety factor. |
| **Upwind scheme** | Numerical method for ∂β/∂x that takes the derivative from the upwind side (left here, because velocity > 0). Conditionally stable. |
| **Periodic boundary** | Edge cells wrap around (cell `−1` is the rightmost cell). Avoids spurious boundary reflections. |
| **`time_step`** | (YAML) Total model-time covered by one filter step. |
| **`n_integration_step`** | (YAML) How many physics sub-steps fit into one filter step. dt = `time_step / n_integration_step`. |
| **`nprt`** | Number of particles (parallel candidate β-fields) in the ensemble. |
| **`init_std_theta`** | Standard deviation of the initial particle spread, in log β space. 0.30 means ≈ 30 % spread in β. |
| **`process_std_theta`** | Standard deviation of the per-step random kick added to log β to represent un-modelled physics. |
| **`obs_noise_std`** | Standard deviation of measurement noise on the observations. |
| **`sensor_stride`** | We pick every Nth flat grid index as a sensor. Larger stride = fewer sensors. |
| **`advection_epsilon` (ε)** | Strength of state-dependent velocity. `v = 1 + ε · β`. |
| **ESS — Effective Sample Size** | A diagnostic from 1 to `nprt` measuring weight balance. 1 = degenerate, `nprt` = uniform weights. |
| **`0.5·Np` line** | Conventional threshold for "filter is starting to struggle." We use it as a visual reference, not an automatic resample trigger (BootstrapFilter resamples every step). |
| **Pressure budget** | `n_obs × (σ_signal / σ_obs)²`. Predicts Bootstrap-PF collapse risk. Lower = healthier. Our v2 sits around 144. |
| **Bootstrap-PF** | The simplest particle filter: propose from the prior dynamics, weight by likelihood, resample. What `BootstrapFilter` in ParticleDA does. |
| **OptimalFilter** | A more sophisticated proposal that dodges the pressure formula but needs extra covariance methods on the model. Not yet implemented in our glacier model. |
| **Resampling** | Replace the particle cloud with N draws from itself, weighted by `w`. Kills bad particles, clones good ones. |
| **CFL warning** | One-shot console warning when our chosen dt exceeds CFL ceiling. Caught automatically in the bug-fixed code. |
| **Surrogate** | Our fast stand-in for the real ice-flow forward map: `ux = 1000/β`. The real WAVI call would be ~1000× slower. |
| **WAVI** | The actual ice-flow simulator we'll eventually plug in instead of the surrogate. |
| **Seed** | Integer that initialises the RNG. Same seed = same random draws = reproducible run. |
| **Seed-noise floor** | The natural spread in metrics just from drawing different random truths and particles. We measured ≈ 5–15 % for ours. |
