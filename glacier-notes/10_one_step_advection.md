# Why is n_integration_step = 1 OK at dt = 400 s?

> **The question.** With `time_step = 400 s` and `n_integration_step = 1`, the filter updates every 400 s of model time, but the underlying advection PDE has to *also* advance through that 400 s window. We're approximating that whole 400 s of physics with a *single* upwind step. Is that accurate enough? How is the PDE actually being updated between observations?
>
> **TL;DR.** Yes, it's accurate enough. For upwind-advection schemes there's a counterintuitive fact: splitting one big step into many smaller sub-steps does **not** automatically improve accuracy — it can actually *increase* numerical smoothing of the field. What matters is the **CFL number** (`v · dt / dx`), and at our value of 0.175 the upwind scheme is essentially as accurate as it can be at this discretisation order. Sub-stepping is a *stability* tool, not an *accuracy* tool. We're currently choosing accuracy over safety margin — both are defensible. This note walks through why.

---

## 1. What actually happens in one 400-second step

The continuous PDE we are solving is:

```
∂β/∂t + (1 + ε β) · ∂β/∂x = 0
```

In one call to `update_state_deterministic!` at the canonical config, the code does exactly **one** upwind step. For each grid cell (j, i):

```
v(j,i)   = 1 + ε · β(j,i)               # local velocity at start of step
dβ/dx    ≈ (β(j,i) − β(j,i−1)) / dx     # spatial derivative, upwind
β_new(j,i) = β(j,i) − v(j,i) · dt · dβ/dx
```

with **dt = 400 s** plugged in directly. That's it. One arithmetic update per cell.

To visualise: in 400 s a feature in β moving at the local velocity v ≈ 1.75 m/s travels `v · dt ≈ 700 m`. Our grid cells are 4000 m wide, so β moves about **17.5 % of one grid cell per filter step**. The upwind scheme handles that by reaching back one cell to the left and pulling forward a weighted blend.

So *during* the 400 s the PDE is not being integrated continuously — it's being approximated by this single algebraic update. The question is whether that approximation is good.

---

## 2. The three things to verify

Whenever you take a single big time step in an explicit numerical PDE solver, you should worry about three different things:

| Concern | What it means | Our status |
|---|---|---|
| **Stability** | "Will the scheme blow up?" Errors must not grow without bound from step to step. | ✅ Safe. CFL number 0.175 < 1 (the strict ceiling). Our safety formula uses 0.2 as a fudge factor; we're at 87 % of that. |
| **Accuracy** | "How close is the discrete update to the true continuous solution after one step?" | ✅ Acceptable at our parameters. The upwind scheme is first-order accurate in both dx and dt; the leading error is O(dt + dx²) per step. With dt = 400 s and a slowly evolving β, the per-step error is small (≪ 1 % of β values). |
| **Numerical diffusion** | "Does the scheme artificially smooth the field?" Upwind always adds some artificial diffusion — the question is how much. | ✅ At our CFL number this is **minimised**, not made worse — see §3 below. |

All three are about what happens *within* one update. If all three are OK, then doing one big step is fine; you don't need to sub-divide.

---

## 3. The counterintuitive bit — why more sub-steps doesn't automatically help

Here is the thing that surprised me when I first ran the math, and is probably the heart of the question.

For an **upwind advection scheme**, the *numerical diffusion coefficient* (the amount of artificial smoothing the scheme adds to the true PDE solution) is:

```
D_num  =  v · dx / 2 · (1 − CFL)        where CFL = v · dt / dx
```

This is the **modified equation** result for first-order upwind — it's saying that the discrete update is actually solving:

```
∂β/∂t + v · ∂β/∂x = D_num · ∂²β/∂x²       (instead of zero on the right)
```

The right-hand side is artificial. It's not in the original physics. It's an artefact of approximating ∂β/∂t and ∂β/∂x with one-sided differences.

**Key observation:** `D_num` is largest when `CFL = 0` (i.e. dt → 0) and **smallest when `CFL = 1`** (the stability boundary).

At our setup:
- `v ≈ 1.75 m/s`, `dx = 4000 m`, `dt = 400 s` → `CFL = 0.175`
- `D_num = 1.75 · 4000 / 2 · (1 − 0.175) ≈ 2 887 m²/s`

If we instead used `n_integration_step = 10` (so internal dt = 40 s):
- `CFL = 1.75 · 40 / 4000 = 0.0175`
- `D_num = 1.75 · 4000 / 2 · (1 − 0.0175) ≈ 3 440 m²/s`

So splitting the 400 s step into 10 sub-steps of 40 s each **increases** the per-step numerical diffusion by about 19 %. (You also do 10 more updates per filter step, but the diffusion accumulates over those steps in a similar ratio. Net result: smaller-dt is *more diffusive*, not less.)

Translated into plain English: **the upwind scheme is most faithful to the real advection when its CFL number is as close to 1 as stability allows.** Tiny sub-steps add more artificial smoothing than one big step.

This contradicts the intuition you'd carry over from, say, Runge–Kutta integration of ODEs, where smaller dt is always better. Upwind is different because its leading error mechanism is *spatial*, not *temporal*. Refining dt without also refining dx doesn't help; sometimes it hurts.

---

## 4. So when *would* you want more sub-steps?

Three legitimate reasons:

1. **Stability safety margin.** Our CFL is 0.175 → just 14 % below the safe ceiling our own formula computes. If β fluctuates upward (which our process noise allows it to do), `v_max` rises and the safe ceiling drops. With `n_integration_step = 2` (internal dt = 200 s) we'd halve the CFL number, gaining a margin from 14 % to 57 %. Cost: an 18 % increase in numerical diffusion, plus 2× compute on the deterministic step. Worth it if we ever push σ_proc or σ_init wider and β starts excursioning bigger.

2. **Strong nonlinearity.** Our advection velocity `v = 1 + ε β` depends on β. In one big step we're using v evaluated *at the start*; the true velocity drifts during the step because β is changing. For our ε = 0.0005 and β ~ 1000, the velocity changes very little over a single step (drift of order 0.001 over 400 s) — so this is negligible. If we cranked ε to 0.01 (β doubles its drift contribution), one big step would start to lose accuracy and sub-stepping would buy us something real.

3. **Filter cadence > CFL ceiling.** If we wanted observations only every 1000 s (longer than our ~457 s CFL ceiling), we'd be forced to split the integration into sub-steps under the ceiling — this is what tsunami does (`time_step = 5 s` exceeds their CFL of ~5 s with no margin, so `n_integration_step = 10` brings internal dt to 0.5 s).

None of these apply right now. Filter is healthy, β stays within bounds, observations come every 400 s. So **n_integration_step = 1 is genuinely the right choice for accuracy** *at this configuration*.

---

## 5. What this means for our results so far

The empirical evidence agrees with the theoretical argument:

- **RMSE plot** ([rmse_beta.png](../glacier-code/particleda/results/rmse_beta.png)) — drops from ~370 to ~115 over 200 steps. Smooth monotonic convergence. If the underlying physics step were grossly inaccurate, the truth and PF mean wouldn't even speak the same language; RMSE would plateau early or wander.
- **ESS plot** ([ess_evolution.png](../glacier-code/particleda/results/ess_evolution.png)) — settles above the 0.5·Np line. If physics were unstable, weights would collapse from one bad particle update; we don't see that pattern.
- **Multi-point plot** ([beta_multi_point.png](../glacier-code/particleda/results/beta_multi_point.png)) — at five non-sensor grid points, the dashed PF mean catches up to the solid truth over time. That's only possible if both are tracking similar underlying physics.

All three are consistent with the upwind step accurately propagating β across the 400 s window. If it weren't, the filter wouldn't be able to learn.

---

## 6. The honest trade-off

There are really two valid setups:

### Option A — accuracy-optimal (current)
```yaml
time_step: 400.0
n_integration_step: 1
```
- Internal dt = 400 s
- CFL number = 0.175
- Numerical diffusion: 2 887 m²/s
- Compute per filter step: 1 × upwind sweep
- Stability margin: 14 % below safe ceiling

### Option B — safety-optimal
```yaml
time_step: 400.0
n_integration_step: 2
```
- Internal dt = 200 s
- CFL number = 0.0875
- Numerical diffusion: 3 194 m²/s (+11 %)
- Compute per filter step: 2 × upwind sweep
- Stability margin: 57 % below safe ceiling

Option A is what we have. Option B was my recommendation in [09_dt_and_cfl_handling.md](09_dt_and_cfl_handling.md) §4.3 *for robustness*.

If the question is purely **"is the PDE being updated faithfully between observations?"**, Option A actually wins by a small margin (less numerical diffusion).

If the question is **"will the scheme survive a worst-case β fluctuation?"**, Option B wins (the bigger safety margin).

Both are defensible. The right call depends on what comes next:
- If we move to **WAVI** (real ice-flow model — much stiffer, with its own CFL), `n_integration_step` will jump anyway, probably to 5–20.
- If we **inflate σ_proc** to test the filter under more aggressive noise, β excursions get bigger and Option B becomes the safer choice.
- If we stay in the current canonical regime, Option A is genuinely the more accurate choice and there's no compelling reason to change.

---

## 7. Direct answer to "how is the PDE updating between intervals?"

It isn't, in a continuous sense. The PDE is being *replaced* by a discrete update rule that produces β at the end of the 400 s window directly from β at the start, in a single algebraic sweep. There's no "in-between" representation. The discrete rule has been chosen (first-order upwind) so that its result approximates the true continuous PDE solution at the end of the interval — accurate to first order in dt and dx, stable provided CFL < 1, with a known small amount of numerical diffusion that we just quantified.

If you wanted higher accuracy, the better answer would be a **higher-order scheme** (Lax–Wendroff at second order, or MUSCL/WENO at higher order) rather than more sub-steps of the same first-order scheme. That's a future enhancement worth considering, but not necessary at our current configuration.

---

## 8. Empirical verification we could run

If you want to *see* this rather than just trust the math, we could run three configs at the same seed:

| Config | n_int | Internal dt | Expected outcome |
|---|---:|---:|---|
| Current (A) | 1 | 400 s | Reference |
| Mid (B) | 2 | 200 s | Slightly more diffusive — β fields very slightly smoother, RMSE within ~5 % of A |
| Heavy | 10 | 40 s | Even more diffusive — β fields visibly smoother, RMSE slightly worse than A |

If the multi-point β plot shows the truth visibly smoother in the "Heavy" run than in "A", that's direct evidence of the numerical-diffusion effect. About 5 minutes of compute total. Say the word and I'll run it.

---

## 9. Glossary additions

| Term | Meaning |
|---|---|
| **First-order upwind** | The cheapest numerical scheme for ∂β/∂x: use a one-sided difference from the upstream direction. Stable under CFL < 1, but adds artificial smoothing. |
| **CFL number** | `v · dt / dx`. Ratio of how far a feature actually moves per step to the grid spacing. CFL = 1 means "exactly one cell per step." Stability requires < 1; minimum diffusion at CFL close to 1. |
| **Numerical diffusion** | Artificial smoothing introduced by a numerical scheme that isn't in the original PDE. For first-order upwind, equal to `v · dx · (1 − CFL) / 2`. Always non-negative. |
| **Modified equation** | The continuous PDE that a discrete scheme is *actually* solving (as opposed to the one you wrote down). Helps quantify accuracy and diffusion. |
| **Higher-order scheme** | A discretisation that's more accurate per step than first-order upwind, at the cost of more arithmetic per cell. Lax–Wendroff (2nd-order) and WENO (5th-order) are common choices. |
