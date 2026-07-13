# Why the wave amplitude decays — and how to reduce it

> The β field's sinusoidal amplitude shrinks over time in both the linear and nonlinear advection runs, even though we never wrote a diffusion term in the PDE and even though the boundary condition is exactly periodic. This note explains **why** with the math and then catalogues **what to do about it**.

---

## 1. The puzzle

The PDE we wanted to integrate is **pure advection**:

```
∂β/∂t + v · ∂β/∂x = 0          (1)
```

Exact solution: `β(x, t) = β₀(x − v·t)`. Shapes translate; amplitudes stay constant.

The boundary condition is **periodic** — implemented as `im = mod1(i − 1, nx)` in [glacier_model.jl](../glacier-code/particleda/glacier_model.jl). Mathematically equivalent to wrapping the domain on a circle, so no information is destroyed at the edges.

Yet in our simulations the sinusoid loses ~76 % of its amplitude over 100 h in the linear case and even more in the nonlinear case. Both the math on paper and the boundary condition are innocent. Where does the damping come from?

**Answer:** the upwind difference operator we use to *discretise* equation (1) is not equivalent to equation (1). It's equivalent to a different PDE — one that *does* have a diffusion term in it. The diffusion is a side effect of the finite-difference stencil.

---

## 2. The discrete scheme

First-order upwind, what we actually run ([glacier_model.jl:240](../glacier-code/particleda/glacier_model.jl#L240)):

```
β_i^{n+1} = β_i^n  −  v·Δt · (β_i^n − β_{i−1}^n) / Δx          (2)
```

Defining the **Courant number** `α ≡ v·Δt/Δx` (a.k.a. CFL), we can rearrange:

```
β_i^{n+1} = (1 − α) · β_i^n + α · β_{i−1}^n                       (3)
```

For stability we need `0 ≤ α ≤ 1`, so equation (3) is a **convex combination** of the cell's current value and its upwind neighbour. A weighted average. Averaging is smoothing. Smoothing kills amplitude.

That's the qualitative reason. The next section makes it quantitative.

---

## 3. Modified-equation analysis — what PDE does (2) actually solve?

We Taylor-expand both sides of (2) around `(x_i, t^n)` and see what continuous PDE the scheme approximates.

For the time step:

```
β_i^{n+1} = β + Δt · ∂_t β + (Δt²/2) · ∂_tt β + 𝒪(Δt³)           (4)
```

For the upwind neighbour:

```
β_{i−1}^n = β − Δx · ∂_x β + (Δx²/2) · ∂_xx β − 𝒪(Δx³)            (5)
```

So:

```
β_i^n − β_{i−1}^n = Δx · ∂_x β − (Δx²/2) · ∂_xx β + 𝒪(Δx³)        (6)
```

Substitute (4) and (6) into (2):

```
Δt · ∂_t β + (Δt²/2) · ∂_tt β
   = − v·Δt · ∂_x β + (v·Δt·Δx/2) · ∂_xx β + 𝒪(higher)            (7)
```

Divide by `Δt`:

```
∂_t β + (Δt/2) · ∂_tt β  = − v · ∂_x β + (v·Δx/2) · ∂_xx β        (8)
```

The leading-order equation (drop everything in `Δt, Δx`) is `∂_t β = −v · ∂_x β` — pure advection, as we wanted. But the `Δt`-order term `(Δt/2) · ∂_tt β` is non-negligible. To leading order in the discrete scheme, `∂_tt β ≈ v² · ∂_xx β` (differentiate pure advection in time once more). Substituting:

```
∂_t β + v · ∂_x β  =  ½·v·Δx·∂_xx β  − ½·v²·Δt·∂_xx β
                  =  ½·v·Δx·(1 − v·Δt/Δx)·∂_xx β
                  =  ½·v·Δx·(1 − α)·∂_xx β                          (9)
```

So the modified equation — the PDE the scheme **really** solves to leading order — is:

```
∂_t β + v · ∂_x β  =  D_num · ∂_xx β                              (10)

with    D_num = ½ · v · Δx · (1 − CFL)                            (11)
```

**Equation (10) is the advection–diffusion equation.** The diffusion coefficient `D_num` is purely a creature of the finite-difference truncation error. You did not write `D · ∂_xx β` anywhere. The discrete scheme introduced it for you.

Two observations from (11):

- **`D_num = 0` only when `CFL = 1` exactly.** Anywhere strictly inside the stable range `(0, 1)`, the scheme leaks diffusion.
- **`D_num ∝ Δx`.** A finer grid actually has *less* numerical diffusion at the same CFL. But pushing CFL toward 1 is the cheapest knob if you can't refine the grid.

---

## 4. Why amplitude decays exponentially

Plug a single Fourier mode `β(x, t) = A(t) · sin(k(x − v·t))` into (10). The advection part `∂_t β + v · ∂_x β` is exactly zero (sine moves rigidly with `v`). The diffusion part gives:

```
∂_xx β = −k² · β(x, t)
```

So `A(t)` evolves under `dA/dt = −D_num · k² · A`, hence:

```
A(t) = A₀ · exp(−D_num · k² · t)                                   (12)
```

Amplitude decays exponentially with rate `D_num · k²`. Short-wavelength modes (`k` large) die fast; long-wavelength modes die slowly.

---

## 5. Plug in the numbers

For our linear branch (`v = 1`, `Δx = 4 km`, `Δt = 360 s`, `n_modes = 3`):

| Quantity | Value |
|---|---:|
| `v` | 1 m/s |
| `Δx` | 4 000 m |
| `Δt = time_step / n_integration_step` | 360 s |
| `CFL = v·Δt/Δx` | 0.09 |
| `D_num = ½·v·Δx·(1−CFL)` | **1 820 m²/s** |
| `k = n_modes·2π/L` | 1.18 × 10⁻⁴ m⁻¹ |
| `D_num · k²` | 2.5 × 10⁻⁵ s⁻¹ |
| Run length | 100 h = 3.6 × 10⁵ s |
| Damping factor `exp(−D_num · k² · T)` | exp(−9.1) ≈ **1.1 × 10⁻⁴** |

Predicted: the sinusoid amplitude should drop to essentially zero over 100 h. That matches what the GIFs show.

For the nonlinear branch, `v ≈ 1 + ε·β` with β up to ~3 500 gives peak `v ≈ 2.75`. That **doubles `D_num`** and **increases `CFL`** by the same factor, so the damping is worse: `exp(−16.4) ≈ 7 × 10⁻⁸` — total decay.

---

## 6. Why the periodic BC is innocent

`im = mod1(i − 1, nx)` ([glacier_model.jl:243](../glacier-code/particleda/glacier_model.jl#L243)) treats the domain as a circle. No information leaks off the boundary; no flux is removed; total `Σ β` is conserved exactly (it's a convex combination of the same `Σ β` at every step).

You can verify this empirically: integrate the truth field `β(x, t)` over all 1600 cells at every step and confirm the sum is constant. It is. So no mass disappears — the field just *redistributes* itself, getting flatter as it does so.

The amplitude decay you see is **redistribution**, not **loss**. Total β is conserved; spatial *variance* of β decreases.

---

## 7. The cups-of-water analogy

Imagine 40 cups in a row, each with some water level (the value of β at that cell). You want the water-level pattern to slide to the right at velocity `v`.

Equation (3) says: at each time step every cup keeps `1 − α` of its own water and accepts `α` of its upwind neighbour's. With `α = 0.09`, every cup swaps 9 % of its level with its left neighbour, every step.

- A peak cup (β = 3 000) sitting next to a trough cup (β = 1 500) at the next step becomes `0.91 × 3000 + 0.09 × 1500 = 2 865`. The peak just lost 135 units.
- The trough cup similarly mixes upward.

Over hundreds of steps, peaks and troughs trade water until all 40 cups average toward the same value. The total water is preserved (no leaks); only the *contrast* between cups is destroyed. That's exactly equation (10): conserved mass, diffused contrast.

---

## 8. How to reduce the amplitude loss

In order of difficulty:

### A. Push CFL toward 1

`D_num ∝ (1 − CFL)`. So if we can keep `CFL` near 1, `D_num` shrinks toward 0.

For our linear case, raising CFL from 0.09 to 0.9 cuts `D_num` by 10×. Over 100 h that turns the damping factor from `exp(−9.1) ≈ 1×10⁻⁴` into `exp(−0.91) ≈ 0.40` — the wave survives at ~40 % amplitude. Big improvement, zero code change.

The knobs:

```
n_integration_step = 1
time_step          = 3600           # → Δt = 3600 s; CFL = 1·3600/4000 = 0.9
```

Caveats:
- **CFL ≤ 1** is required for stability. Going to CFL = 1 exactly removes diffusion but is on the edge of instability — any rounding error grows. Stay at CFL ≈ 0.9 in practice.
- For **nonlinear** advection, `v` varies in space and time. The local CFL can spike above 1 in high-β regions. You'd need a smaller `Δt` to keep the worst-case CFL safe.

### B. Lax–Wendroff scheme

Replace the 1-sided difference with a 2-sided difference + a correction term that **cancels the leading-order numerical diffusion**:

```
β_i^{n+1}  =  β_i  −  ½·α·(β_{i+1} − β_{i−1})           (centred-difference advection)
                  +  ½·α²·(β_{i+1} − 2·β_i + β_{i−1})    (correction term)
```

Modified-equation analysis of this scheme shows `D_num ≡ 0` to second order. Leading-order error is now *dispersion* (oscillations near sharp edges) at `𝒪(Δx²)`. Cheap: same loop structure, just two more indices touched per cell.

For our nonlinear case the natural generalisation is:

```
v_i        =  1 + ε·β_i
α_i        =  v_i · Δt / Δx
β_i^{n+1}  =  β_i  −  ½·α_i·(β_{i+1} − β_{i−1})
                  +  ½·α_i²·(β_{i+1} − 2·β_i + β_{i−1})
```

We've added this as a new `advection_type = "lax_wendroff"` branch in the model so you can flip it on with a single YAML/dict change. See §8.x below.

### C. Higher-order upwind with a flux limiter (MUSCL / Van Leer)

Second-order accurate in smooth regions, first-order near shocks. Robust workhorse in CFD. Implementation ~30 lines.

### D. Semi-Lagrangian / characteristic tracking

Trace each cell's value back along the velocity characteristic by `−Δt` and interpolate at that location. Has zero numerical diffusion in the limit, no CFL stability constraint. Most expensive change but the most accurate.

---

## 9. Verifying the prediction empirically

Three sanity-check experiments you can run that should match the theory:

1. **Bump `n_integration_step = 1` (CFL ≈ 0.9).** Re-run any tracking experiment. The truth's sinusoid should now retain ~40 % of its amplitude at 100 h instead of decaying to nothing.
2. **Halve `time_step` to 1800 s.** Δt halves, CFL halves to 0.045, `D_num` only changes ~5 % (since `(1 − 0.045) ≈ (1 − 0.09)`). Damping is essentially unchanged. Demonstrates that `Δt` per se isn't the issue — `CFL` is.
3. **Flip `advection_type = "lax_wendroff"` (after enabling).** Damping should drop *dramatically* — the leading-order diffusion is removed.

---

## 10. Where this shows up

- The cross-section GIFs in [results/run07_particle_tracking/](../glacier-code/particleda/results/run07_particle_tracking/) and [run08_linear_advection_run/](../glacier-code/particleda/results/run08_linear_advection_run/): truth's sinusoid visibly flattens over 100 h.
- The sensor-trail GIF in [results/run09_no_obs_linear_adv/](../glacier-code/particleda/results/run09_no_obs_linear_adv/): truth swings around 2000 driven by process noise, but the *expected value* (which the ensemble mean tracks since no obs) flattens to exactly 2000 — which is the prior at that cell, with the sinusoid already diffused away.
- [`glacier-notes/10_one_step_advection.md`](10_one_step_advection.md) had a partial version of this analysis; this note supersedes it with the full modified-equation derivation and quantitative predictions.

---

## 11. Glossary additions

| Term | Meaning |
|---|---|
| **Modified equation** | The continuous PDE that a discrete numerical scheme **actually** solves to leading order, found by Taylor-expanding the scheme. Distinct from the PDE the scheme was *designed* to solve. |
| **Truncation error** | The difference between the original PDE and the modified equation. For first-order upwind it manifests as a `D_num · ∂_xx β` diffusion term. |
| **Numerical diffusion** | Diffusion-like dissipation that comes from the discretisation, not from any physical term in the PDE. |
| **`D_num`** | Numerical-diffusion coefficient `= ½ · v · Δx · (1 − CFL)` for first-order upwind. |
| **CFL number** | `α = v · Δt / Δx`. Stability of upwind requires `0 ≤ α ≤ 1`. Damping is worst at CFL → 0 and disappears at CFL → 1. |
| **Lax–Wendroff** | Second-order explicit scheme that uses a centred difference for advection plus a `½·α²·∂_xx` correction. Cancels the leading-order numerical diffusion at the cost of introducing dispersion. |
| **Dispersion** | The error mode of higher-order schemes: different wavenumbers travel at slightly different speeds, causing wiggles near sharp gradients. Distinct from dissipation/diffusion. |
