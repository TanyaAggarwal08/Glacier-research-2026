# dt and CFL — who computes it, who enforces it, what's the right choice

> **The question.** When LowLevel-style code computes `dt` internally from the CFL formula, it can't accidentally pick an unstable value. When tsunami-style code (and our v2 code) reads `dt` from the YAML, the user can pick anything — stable or not. Which approach is correct? What does the tsunami model *actually* do? Why does it use dt = 0.5 s with 10 sub-steps while we use dt = 400 s with 1 sub-step?
>
> This note answers all of that with numbers verified against [test/models/llw2d.jl](../test/models/llw2d.jl), [test/models/llw2d_timestepping.jl](../test/models/llw2d_timestepping.jl), [bootstrap-pf-experiments/code/benchmark.yaml](../bootstrap-pf-experiments/code/benchmark.yaml), and our own [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) — not from memory.

---

## 1. What the tsunami model actually does (verified from source)

### 1.1 The workflow in plain English

When ParticleDA runs the tsunami benchmark, here is what happens **per filter step**, for each particle:

1. `ParticleDA.run_particle_filter` calls `LLW2d.update_state_deterministic!` ([test/models/llw2d.jl:602-630](../test/models/llw2d.jl#L602-L630)).
2. Inside that function, **dt is computed as `dt = time_step / n_integration_step`**. Specifically line 612:
   ```julia
   dt = model.parameters.time_step / model.parameters.n_integration_step
   ```
3. A loop runs `n_integration_step` sub-steps, each calling `timestep!`. So a single filter step does multiple smaller physics updates.
4. `timestep!` ([test/models/llw2d_timestepping.jl:16-87](../test/models/llw2d_timestepping.jl#L16-L87)) is the actual shallow-water solver. It takes `dt` as an argument and uses it twice: once to update velocity from height gradients (lines 53-54), and once to update height from velocity divergence (line 79). **There is no CFL check inside `timestep!`. There is no CFL check anywhere in the tsunami code.** It trusts whoever picked the YAML values.
5. After the deterministic loop, `update_state_stochastic!` adds Matérn-GRF noise.
6. Then ParticleDA's outer loop compares to the observation, computes weights, resamples.

So **tsunami uses approach B**: YAML-controlled dt, no internal CFL check, full user responsibility.

### 1.2 The exact tsunami numbers

From [bootstrap-pf-experiments/code/benchmark.yaml](../bootstrap-pf-experiments/code/benchmark.yaml):

| Parameter | Value |
|---|---:|
| `x_length` | 200 000 m |
| `nx` | 51 |
| `dx` (computed as `x_length / (nx-1)`) | 4000 m |
| `bathymetry_setup` | 3000 m (ocean depth) |
| `time_step` | 5 s |
| `n_integration_step` | 10 |
| Internal dt | 0.5 s |

Wave speed in shallow water: c = √(g · h) = √(9.80665 × 3000) ≈ **171.5 m/s**.

CFL number used by tsunami: `c · dt / dx = 171.5 × 0.5 / 4000 ≈ 0.021`. The "unsafe" ceiling is c·dt/dx = 1, so they're sitting at 2.1 % of the unsafe limit — **extremely conservative**. They could mathematically use dt up to ~23 s before things become unstable, but they chose 0.5 s.

**Why so conservative?** Three plausible reasons (not stated in the code, but consistent with the practice):
- Wave height locally is non-uniform (the initial peak is 30 m); local velocities can spike well above the mean wave speed.
- Boundary effects (sponge layer, land masks) introduce local instabilities that need smaller dt to damp.
- The same model is used in unit tests with synthetic peaks; conservative dt makes the code robust across configurations.

### 1.3 *Why* tsunami needs 10 sub-steps

The natural answer to "why time_step=5 with n_integration_step=10 instead of time_step=0.5 with n_integration_step=1?" is:

**Because the filter cadence and the physics cadence are different requirements.**

- **Filter cadence** (observations per unit model time): set by **what we want to assimilate**. For tsunami, observations come every 5 s of model time. The PF should update every 5 s.
- **Physics cadence** (size of each numerical step): set by **what the numerics can do stably**. For tsunami, anything under ~5 s is fine, but 0.5 s gives a fat safety margin.

If you set `time_step = 0.5, n_integration_step = 1`, the filter would resample every 0.5 s — 10× more often than wanted, 10× more compute, 10× more weight-collapse risk (every-step resampling is hard on ESS — see [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md) §3).

If you set `time_step = 5, n_integration_step = 1`, the filter cadence is right but the physics step is 5 s — at the very edge of CFL stability. Local spikes in wave height push you over the edge and the simulation could blow up.

**The split (`5, 10`) gives you the best of both worlds: filter looks at observations every 5 s, but the physics is integrated in 10 safe sub-steps of 0.5 s in between.** That's exactly what `n_integration_step` is designed for.

---

## 2. What our glacier model does (verified from source)

### 2.1 The workflow

Per particle per filter step ([glacier_model.jl:97-141](../glacier-code/particleda/glacier_model.jl#L97-L141)):

1. ParticleDA calls `Glacier.update_state_deterministic!`.
2. Inside, line 105: `dt = p.time_step / p.n_integration_step`. **Identical pattern to tsunami.**
3. **Unlike tsunami, we have a one-shot CFL safety check** ([glacier_model.jl:108-115](../glacier-code/particleda/glacier_model.jl#L108-L115)): the first time through, it computes the current CFL ceiling and warns if our chosen dt exceeds it.
4. Then a loop of `n_integration_step` upwind sub-steps.
5. Returns; `update_state_stochastic!` adds iid Gaussian noise.

### 2.2 The exact glacier numbers

From [glacier.yaml](../glacier-code/particleda/glacier.yaml):

| Parameter | Value |
|---|---:|
| `x_length` | 160 000 m |
| `nx` | 40 |
| `dx` (computed as `x_length / nx`) | 4000 m |
| `advection_epsilon` (ε) | 0.0005 |
| Typical β_max | ~1500 |
| `time_step` | 400 s |
| `n_integration_step` | 1 |
| Internal dt | 400 s |

Wave speed: v_max = 1 + ε · β_max = 1 + 0.0005 × 1500 ≈ **1.75 m/s**.

CFL number: `v · dt / dx = 1.75 × 400 / 4000 = 0.175`. The "unsafe" ceiling is 1, so we're sitting at 17.5 % of the unsafe limit. With the 0.2 safety factor in our internal CFL check, we're at 87 % of the *safe* ceiling — meaning **we're at the edge** of what our safety formula allows.

### 2.3 Why we only need 1 sub-step

Because our filter cadence and our physics cadence both want roughly the same thing.

- Filter cadence: we want observations every ~400 s (this is what gives ESS-healthy behaviour at the current σ_obs and sensor count).
- Physics cadence: dt up to ~457 s is stable; 400 s is fine.

These match. No splitting needed. If we ever want filter cadence longer than ~457 s, we'd need `n_integration_step ≥ 2` to keep the internal dt below CFL.

---

## 3. Side-by-side workflow comparison

### 3.1 The same outer loop, different per-particle integration

Both models share the **same outer particle-filter loop** (this is ParticleDA's job, identical for any model):

```
for t in 1..n_time_step:
    for each particle:
        update_state_deterministic!(particle.state, model, t)   # ← model-specific
        update_state_stochastic!(particle.state, model, rng)    # ← model-specific
    weights = likelihood(obs[t], particle predictions)
    if needed, resample
    write HDF5
```

The **only thing that differs** is what each model does inside `update_state_deterministic!`. And both follow the same template:

```
dt = time_step / n_integration_step
for k = 1..n_integration_step:
    advance physics by dt
```

### 3.2 What's inside the per-sub-step physics

**Tsunami** (`timestep!` in [llw2d_timestepping.jl](../test/models/llw2d_timestepping.jl)):

```
For each cell:
    grad_x  = (eta0[i,j] - eta0[i-1,j]) / dx
    grad_y  = (eta0[i,j] - eta0[i,j-1]) / dy
    mm1     = mm0 - g · depth_x · grad_x · dt    # x momentum update
    nn1     = nn0 - g · depth_y · grad_y · dt    # y momentum update
    (apply land filters, sponge absorbers)
    div_x   = (mm1[i+1,j] - mm1[i,j]) / dx
    div_y   = (nn1[i,j+1] - nn1[i,j]) / dy
    eta1    = eta0 - (div_x + div_y) · dt        # height update
```

It uses dt **twice** per sub-step (once for momentum, once for height) — that's the leapfrog-flavoured shallow-water update.

**Glacier** (`update_state_deterministic!`):

```
For each cell:
    im      = i - 1 (wrap)
    grad_x  = (β[j,i] - β[j,im]) / dx
    v       = 1 + ε · β[j,i]
    β_new   = β[j,i] - v · dt · grad_x          # single state update
```

It uses dt **once** per sub-step. There's only one state variable (β), no momentum to track.

### 3.3 The full comparison table

| | Tsunami (LLW2d) | Glacier (ours, v2) |
|---|---|---|
| **Where dt is computed** | YAML, in `update_state_deterministic!` | YAML, in `update_state_deterministic!` |
| **Formula** | `dt = time_step / n_integration_step` | `dt = time_step / n_integration_step` |
| **Internal CFL check** | ❌ none | ✅ one-shot warning |
| **State variables per cell** | 3 (height, m, n — i.e. h, momentum_x, momentum_y) | 1 (β) |
| **Grid size** | 51 × 51 | 40 × 40 |
| **dx** | 4000 m | 4000 m |
| **Wave speed** | √(g·h) = 171.5 m/s | 1 + ε·β = 1.75 m/s |
| **CFL ceiling (unsafe)** | dx / c = 23.3 s | dx / v = 2286 s |
| **CFL ceiling (0.2 safety)** | 4.66 s | 457 s |
| **YAML `time_step`** | 5 s | 400 s |
| **YAML `n_integration_step`** | 10 | 1 |
| **Actual dt** | 0.5 s | 400 s |
| **Used CFL number (dt·c/dx)** | 0.021 (2.1 % of unsafe) | 0.175 (17.5 % of unsafe) |
| **Distance to safety limit** | 4.66 / 0.5 = 9.3× margin | 457 / 400 = 1.14× margin |
| **Total model time per run** | 250 × 5 = 1 250 s ≈ 21 min | 200 × 400 = 80 000 s ≈ 22 hr |
| **Filter steps per minute of physics** | 12 obs/min | 0.075 obs/min |

The two biggest take-aways from this table:

- **Wave speed ratio of ~100× drives everything.** Same grid, same dx, but speeds differ by 100. That makes CFL ceilings differ by 100. Glacier can use 800× larger dt than tsunami (400 vs 0.5) because nothing on our grid moves anywhere near as fast.
- **Tsunami runs conservatively (9× margin), we run aggressively (1.14× margin).** Our safety factor of 0.2 already does the "1× safety margin" math; we're using dt = 87 % of the safe ceiling. Tsunami's 5-step time_step with 10 sub-steps gives a 9× margin even ignoring their internal 0.2 (which they don't even check). They're being defensive.

---

## 4. Which approach is "correct"?

There are three honest answers depending on what we mean.

### 4.1 For data assimilation, YAML-controlled dt is structurally correct

Real observations come at fixed wall-clock intervals (satellite passes, GPS sensors, InSAR products). You can't ask nature to observe your glacier at exactly `0.2 · dx / max_speed` seconds. So the filter has to be told what cadence to use; it can't derive that from physics alone.

LowLevel's "compute dt from CFL each call" works because LowLevel doesn't separate filter cadence from physics cadence — they're the same thing in that codebase. Useful as a sanity check, but inflexible for assimilation.

ParticleDA (and tsunami, and our v2) make the right structural choice: **filter cadence is a user input, physics step is derived to fit**. That's why they have the `time_step / n_integration_step` knobs.

### 4.2 For stability, internal CFL guarding is the safety net

The downside of YAML-controlled dt is that a careless user can pick an unstable combination. The tsunami code doesn't guard against this at all; the user is fully on their own.

We added a one-shot CFL warning in [glacier_model.jl:108-115](../glacier-code/particleda/glacier_model.jl#L108-L115). That's strictly better than tsunami's silence. It catches the case where someone changes `time_step` upward without updating `n_integration_step`.

We could go further (see §5 below): per-step CFL check, automatic `n_integration_step` selection, hard error instead of warning. Each adds robustness at the cost of code complexity.

### 4.3 For "is 400 s the right dt for our problem?"

Looking at the table:
- Tsunami sits at 2.1 % of unsafe CFL with a 9× margin.
- We sit at 17.5 % of unsafe CFL with a 1.14× margin.

**Our dt is fine but tight.** The 1.14× margin means: if β fluctuates upward to ~1750 (which it can under our process noise σ_θ = 0.007 cumulatively over many steps), v_max rises to ~1.875, and CFL ceiling drops to 426 s. Our dt = 400 s is still under, but barely.

If we wanted to match tsunami's conservatism, we'd want dt around `457 / 9 ≈ 50 s` — say `time_step = 400` with `n_integration_step = 8` (internal dt = 50 s). That'd cost 8× more compute per filter step in `update_state_deterministic!` but be much safer.

**My recommendation:** keep `time_step = 400`, but bump `n_integration_step = 2` (internal dt = 200 s). That doubles the safety margin to 2.3× while only adding a small compute cost. Strictly better. And if we ever see the CFL warning fire on a real run, jump to `n_integration_step = 4`.

This isn't urgent — current runs are stable — but it's the disciplined choice and matches what tsunami does proportionally.

---

## 5. Concrete things we could do

In priority order:

1. **Bump `n_integration_step` from 1 to 2** in the canonical YAML. Doubles safety margin. Tiny compute cost (about 5–10 % more time). Strictly better with no downside.
2. **Make the CFL check fire per-step rather than one-shot.** Currently `_CFL_WARNED[]` makes the warning fire exactly once across the whole run. If β grows over time, we'd miss the moment of violation. Fix: check each step but only print once per (rounded) violation magnitude.
3. **Add an `auto_n_integration_step` flag.** If user sets it true, model computes the smallest `n_integration_step` that keeps dt under the safe CFL ceiling, based on current state. Tsunami doesn't have this; it'd be a nice add.
4. **Error instead of warn at extreme violation.** If dt > 5 × CFL ceiling, abort. That's deep into the explosion zone; the user definitely meant something else. We already have a warning; turning egregious violations into errors prevents wasted compute.

None of (2)-(4) are strictly necessary; the YAML pattern is correct as-is. (1) is a small immediate hardening I'd recommend doing before the next experiment.

---

## 6. Direct answers to the user's questions

> **Which is correct: dt computed internally for CFL (LowLevel) or dt set by YAML (tsunami/ours)?**

YAML-controlled is structurally correct for data assimilation because real observation cadence is a problem input, not something physics dictates. LowLevel's internal-CFL approach is fine for self-contained simulation but ties cadence to physics, which is the wrong constraint for assimilation. The tsunami benchmark uses YAML-controlled dt and doesn't even bother checking CFL internally — they trust the user to pick safe values.

> **What does tsunami actually do?**

`dt = time_step / n_integration_step` inside `update_state_deterministic!`. No CFL check. With YAML `time_step = 5 s` and `n_integration_step = 10`, internal dt = 0.5 s. Their CFL number is 0.021 — extraordinarily conservative (9× safety margin even ignoring their lack of safety factor).

> **Why does tsunami use 0.5 s with 10 sub-steps and we use 400 s with 1 sub-step?**

Two reasons:
1. **Physics speed.** Tsunami waves move at √(g·h) = 171.5 m/s. Our β advection moves at 1 + ε·β = 1.75 m/s. ~100× slower. CFL ceiling is inversely proportional to wave speed, so our ceiling is ~100× larger.
2. **Their filter cadence requires sub-stepping; ours doesn't.** Tsunami wants observations every 5 s (filter cadence). But CFL forces them under ~5 s for stability, so they split each filter step into 10 sub-steps of 0.5 s each. Our filter cadence (400 s) is already under our CFL ceiling (457 s), so 1 sub-step works.

> **Which way is better?**

For our project (and any DA project): the YAML-controlled pattern, with an internal CFL safety check that warns. We're already doing this. The one improvement worth making is `n_integration_step = 2` to widen the safety margin from 1.14× to 2.3×, matching the tsunami's conservatism level proportionally.

If you'd like, I can apply that change (bump to `n_integration_step = 2`) and re-run the seed sweep one more time to confirm nothing breaks. Single change, ~3 minutes of compute.
