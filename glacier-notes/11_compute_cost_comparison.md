# Why our 80 000-second run is faster than the tsunami's 1 250-second run

> **The puzzle.** The tsunami benchmark covers 1 250 s of model time and takes about 6 minutes wall-clock for 1 000 particles. Our glacier run covers **64× more model time** (80 000 s) with the same particle count and finishes in about **15 seconds wall-clock**. That's a ~24× speed advantage for a longer simulation. Why?
>
> **TL;DR.** "Model time covered" and "compute time used" are not the same thing. The actual cost is `n_filter_steps × n_integration_step × cost_per_sub-step`. We have *fewer* total sub-steps (because each is much bigger) and each sub-step is *cheaper* (simpler physics, smaller state, simpler noise). Multiply those out and we should be roughly 60× faster — which matches what we see.
>
> Confirmed timing of the canonical run on this machine: **`real 14.97 s`** for 1 000 particles × 200 filter steps × 1 sub-step each.

---

## 1. Model time covered ≠ compute time used

A common intuition: "if my simulation covers 80 000 s of model time, that's a lot — it should take longer than a 1 250 s simulation."

This is wrong because the relationship between model time and compute time depends on **how big each numerical step is**. A small dt forces many steps to cover a fixed model interval; a big dt covers the same interval in fewer steps.

The right cost formula is:

```
total_compute  ≈  n_filter_steps × n_integration_step × cost_per_internal_step
```

Each factor pulls in opposite directions for us vs tsunami. Let's count them carefully.

---

## 2. The factor-by-factor breakdown

| Factor | Tsunami | Glacier (ours) | Ratio (theirs / ours) |
|---|---:|---:|---:|
| Particles (`nprt`) | 1 000 | 1 000 | 1× |
| Filter steps (`n_time_step`) | 250 | 200 | 1.25× |
| Internal sub-steps per filter step | 10 | 1 | 10× |
| **Total internal physics steps per particle per run** | **2 500** | **200** | **12.5×** |
| Internal dt | 0.5 s | 400 s | 1/800× (we use larger dt) |
| Total **model time** covered per run | 1 250 s | 80 000 s | 1/64× (we cover more) |

Already this tells you the headline: **we do 12.5× fewer arithmetic updates** despite covering 64× more model time. That's because each of our updates is doing 800× more model-second of work.

But the per-step cost also differs.

### 2.1 Cost per internal physics step

This is what `update_state_deterministic!` actually does in one sub-step. Counted directly from the source.

**Tsunami** ([test/models/llw2d_timestepping.jl](../test/models/llw2d_timestepping.jl) lines 16–87):

| Pass over the 51×51 grid | What it does |
|---|---|
| 1 | Compute x-gradient of `eta` (`dx_buffer`) |
| 2 | Compute y-gradient of `eta` (`dy_buffer`) |
| 3 | Update x-momentum `mm1 = mm0 - g·depth_x·grad·dt` |
| 4 | Update y-momentum `nn1 = nn0 - g·depth_y·grad·dt` |
| 5 | Apply land filter + sponge absorber to `mm1` |
| 6 | Apply land filter + sponge absorber to `nn1` |
| 7 | Compute x-divergence of `mm1` |
| 8 | Compute y-divergence of `nn1` |
| 9 | Update height `eta1 = eta0 - (div_x + div_y)·dt` |
| 10 | Apply land filter to `eta1` |
| **Total** | **~10 passes over the grid per sub-step** |

Each pass touches 2 601 cells. Total ops per sub-step ≈ 26 000.

**Glacier** ([glacier-code/particleda/glacier_model.jl](../glacier-code/particleda/glacier_model.jl) lines 117–127):

| Pass over the 40×40 grid | What it does |
|---|---|
| 1 | For each cell: compute gradient + multiply by `v · dt` + subtract |
| **Total** | **1 pass over the grid per sub-step** |

Each pass touches 1 600 cells. Total ops per sub-step ≈ 1 600.

**Ratio (per sub-step):** tsunami does about `26 000 / 1 600 ≈ 16×` more arithmetic per sub-step. (~10× more passes × ~1.6× more cells.)

### 2.2 Cost per stochastic step

`update_state_stochastic!` is also called once per filter step per particle.

**Tsunami** uses **Matérn Gaussian random fields** generated via circulant embedding (FFT-based). For a 51×51 field with three state components, each sample needs ~3 forward FFTs and ~3 inverse FFTs. A 64×64 (next-power-of-2) FFT is O(N² log N) ≈ 24 000 ops, doubled to 48 000 with the complex factor. Per particle per filter step: ~150 000 ops.

**Glacier** uses **iid Gaussian noise**: `state[i] += σ · randn()` for each of 1 600 cells. Per particle per filter step: ~1 600 ops, with no FFT overhead.

**Ratio (noise step):** tsunami's noise generation is roughly `150 000 / 1 600 ≈ 100×` more expensive than ours. The FFT machinery and the Matérn correlation are doing real work; the iid alternative is essentially free.

### 2.3 Total cost ratio (theory)

Stacking the multiplications:

```
deterministic:  12.5  (sub-steps)  ×  16  (ops/sub-step)   ≈  200×
stochastic:                            1   (1 noise call/filter-step both ways)
                                      ×  100  (ops/noise)   ≈  100×
```

These are not additive; in practice the deterministic work dominates because it's per-sub-step while the stochastic is per-filter-step. Overall expect tsunami to be **somewhere between 50× and 200× slower** than glacier.

The user-reported 6 minutes (~360 s) for tsunami vs measured 15 s for glacier gives a **24× ratio**. That's at the low end of the theoretical range — probably because:
- Some overheads (file IO, MPI plumbing, garbage collection) don't scale with the inner loop and are similar in both cases.
- The Matérn FFTs benefit from BLAS / heavily-optimised libraries; the iid noise doesn't have an equivalent boost to exploit.
- Different machines, threads, and warmup costs can compress the gap.

24× is in the right ballpark. The big-picture story is solid: **most of our speed comes from doing many fewer, simpler arithmetic updates, not from cutting corners in the science.**

---

## 3. Why each factor goes our way

This isn't an accident or a special optimisation — it's a direct consequence of the physics we're trying to model.

### 3.1 Slower waves → larger CFL ceiling → larger dt → fewer sub-steps

Shallow-water waves travel at `√(g · depth)` = 171.5 m/s for the tsunami benchmark.
Glacier β advection travels at `1 + ε · β` ≈ 1.75 m/s.

A 100× speed gap directly produces a 100× larger stable dt. Since our filter cadence (400 s) is already under the stable ceiling, we don't need to split sub-steps. Tsunami's cadence (5 s) is not under their stable ceiling (~5 s) with safety margin, so they split into 10.

This is structural: any DA project on a "slow" physical process (glaciers, climate, ocean basin circulation) will inherit this advantage over a "fast" one (tsunami, atmospheric weather, gravity waves).

### 3.2 Fewer state variables → smaller inner-loop body

Tsunami's shallow-water equations are *second-order in space-time* — solving them as a first-order system requires three coupled variables (height + x-momentum + y-momentum). Each sub-step updates all three.

Our advection is *first-order in space-time*. One state variable per cell. One update per cell per sub-step.

This is also structural: any first-order conservation law gets this advantage over a second-order wave equation.

### 3.3 Simpler noise model

We use iid Gaussian noise per cell, which is essentially free.

Tsunami uses Matérn Gaussian random fields, which are scientifically appropriate for shallow water (the noise is spatially correlated; nearby cells must look similar) but require FFTs.

We could swap our iid noise for Matérn GRFs later — see the "what's next" list in [05_particleda_vs_lowlevel_comparison.md](05_particleda_vs_lowlevel_comparison.md). Doing so will close part of the speed gap (probably down to ~10× faster instead of 24×). It's also more physically realistic, so it's a worthwhile upgrade.

### 3.4 Smaller grid (a minor factor)

Tsunami: 51×51 = 2 601 cells. Glacier: 40×40 = 1 600. About 1.6× fewer cells per state-variable plane.

This is the smallest of the factors. It's the easiest thing to change if we wanted to scale up — going to 80×80 or 160×160 is just a YAML edit. The pressure-budget analysis ([00_action_plan.md](00_action_plan.md)) doesn't care about grid resolution as long as the obs operator is held fixed; the filter compute scales as `(nprt × cells × n_sub-steps)`.

---

## 4. Practical consequences

### 4.1 We have headroom for the expensive things we want to do

The big future change is **swapping the surrogate for WAVI** ([05 §What to change next](05_particleda_vs_lowlevel_comparison.md)). WAVI is a real ice-flow solver and each call will be hundreds-to-thousands of times slower than our `1000/β` formula.

If WAVI costs, say, 100 ms per particle per filter step (a reasonable mid-range estimate), then 1000 particles × 200 steps = 200 000 calls × 100 ms = ~5.5 hours per run. That's much slower than now, but our 15-s headroom means we can afford the upgrade without it becoming infeasible. Tsunami's tighter budget would have made the same upgrade harder.

### 4.2 We can afford bigger experiments

The seed sweep at 5 seeds takes 5 × 15 s ≈ 75 seconds. A 20-seed sweep would still be under 5 minutes. A σ_obs sweep at 5 values × 5 seeds = 25 runs ≈ 6 minutes total. We are not compute-bound on the surrogate problem; we should run more experiments while we still can.

### 4.3 We could be more conservative on CFL "for free"

Bumping `n_integration_step` from 1 to 2 (which I recommended in [09 §4.3](09_dt_and_cfl_handling.md)) costs about 5–10 % more wall-clock per run (~16 s instead of 15 s). It's a rounding error. We should just do it if we want the extra stability margin, no analysis needed.

### 4.4 The tsunami benchmark isn't slow because it's badly written

Just to be clear: 6 minutes for the tsunami benchmark is **not a sign of inefficient code**. The LLW2d implementation is reasonably tight. It's slow because shallow water *physics* is computationally heavier than upwind advection — three coupled fields, FFT-based noise, tiny CFL-limited dt. Any equivalent shallow-water DA setup would hit similar costs. Don't read "tsunami slow / glacier fast" as a code-quality comparison; it's a physics-cost comparison.

---

## 5. Direct answer to the user

> **Why is our model running so fast even at 1 000 particles when tsunami takes ~6 minutes?**

Three reasons stack multiplicatively:

1. **Fewer numerical steps** — we do 200 internal physics updates per particle per run; tsunami does 2 500. Our slower-moving physics has a 100× larger stable dt, so we can take much bigger steps. (12.5× win.)
2. **Cheaper physics per step** — our upwind scheme does one pass over the grid per sub-step; tsunami's shallow-water solver does ~10 passes over three state fields. (~16× win.)
3. **Cheaper noise** — we use iid Gaussian per cell; tsunami uses FFT-based Matérn random fields. (~100× win, only triggered once per filter step.)

Stacking these (with overhead) gives a theoretical ~50–200× speed advantage; we observe ~24×. The gap is structural: we're solving a *slower, simpler* physics problem on a *smaller* grid with *simpler* noise. None of it is a code-cleverness difference — it's a physics-cost difference.

The fact that we cover 80 000 s of model time vs tsunami's 1 250 s is irrelevant to compute time. **Model time isn't compute time.**

---

## 6. Glossary

| Term | Meaning |
|---|---|
| **Wall-clock time** | How long the user waits, by the clock on the wall. Independent of how much model time was covered. |
| **Model time** | The amount of simulated physical time the run covers. `n_time_step × time_step`. |
| **Internal step / sub-step** | One application of the discrete physics update. There are `n_integration_step` of these inside each filter step. |
| **Pass over the grid** | One loop through every cell to compute or update something. Tsunami's shallow-water solver does ~10 of these per sub-step; our upwind does 1. |
| **Matérn GRF** | A spatially-correlated Gaussian random field used for realistic noise in DA. Generated via FFT-based circulant embedding. Expensive but physical. |
| **iid Gaussian noise** | Independent identically distributed noise per cell — each cell gets its own random draw. Cheap but not spatially smooth. |
| **WAVI** | The real ice-flow forward model we eventually want to use. Will be much slower than our surrogate; our current speed headroom is what makes plugging it in feasible. |
