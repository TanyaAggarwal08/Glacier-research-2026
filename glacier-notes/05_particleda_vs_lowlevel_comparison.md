# ParticleDA vs LowLevel — first-pass result comparison

> Snapshot of the very first ParticleDA glacier run vs the LowLevel toy that lived in `glacier-code/results/`. The headline observation — "ParticleDA's ESS looks much worse" — is **partly real and partly an artifact of comparing different experiments through different filter conventions.** This note pulls those apart so we know which knobs are worth turning.

## TL;DR

| | LowLevel toy (`glacier-code/results/`) | ParticleDA glacier (`glacier-code/particleda/results/`) |
|---|---|---|
| Driver | `particlefilteringnonlinear.jl` | `run_glacier_pda.jl` |
| State | β directly, range ≈ [−1, 1] | θ = log β, β ≈ [500, 1500] |
| Forward op | identity (obs = β at sensors) | nonlinear surrogate (obs = 1e3/β at sensors) |
| Np | 10 000 | 200 |
| T | 200 | 50 |
| σ_obs | 0.1 (10 % of signal) | 0.05 (≈ 5 % of ux ≈ 1) |
| σ_init | 0.8 (80 % of signal) | 0.05 in log β (≈ 5 % of β) |
| Resampling | **adaptive** (only when ESS drops) | **every step** (BootstrapFilter default) |
| ESS plot | pegged at Np, ~10 dips to ~10–15 % of Np | bouncing 1–30 (out of 200) every step |
| Max weight | mostly tiny | 0.1–0.65 every step |
| Posterior Var(β) | not logged | collapses to ~1.0 after step 2 |
| β RMSE | 0.65 → 0.19 (relative ~20 %) | 52 → 37 (relative ~4 %) |
| β pointwise track | — (different model) | PF mean tracks truth within ~5 % |

So: the **tracking** (RMSE, pointwise plot) is actually fine — *better* in relative terms than the LowLevel toy. The **ensemble health** is genuinely poor: a single particle holds 30–60 % of the mass most steps, and the posterior variance crashes after one step. That's real degeneracy. The ESS plot exaggerates the gap because the two filters report ESS at different moments in their cycles.

## Where the ESS-plot gap comes from

There are six effects stacked on top of each other. Four are *experimental confounders* (not telling us anything about the filter); two are *real* degeneracy issues we should fix.

### 1. (Confounder) Different problems

The LowLevel ESS plot you're comparing to is from `particlefilteringnonlinear.jl` — a toy advection of a smooth sinusoid in [−1, 1] with **identity observation operator**. The ParticleDA run is the glacier surrogate (`ux = 1e3/β`) — a **strongly nonlinear, decreasing** observation operator. These are not the same problem; calling them both "particle filter" is the only thing they share.

The fair LowLevel reference would be `glacier-code/testing_stage/particlefilteringwithiceflow.jl` (same surrogate, same log β state). That file doesn't ship an ESS plot in `glacier-code/results/`, so we cannot compare like-for-like yet. **Action:** if we want a clean head-to-head, run `particlefilteringwithiceflow.jl` once and save its ESS history.

### 2. (Confounder) 50× particle count

Np=10000 vs Np=200. The LowLevel paper finding from `01_lowlevel_Np10000_observations.md` was: at Np=10000 the dips bottom at 10–15 % of Np. Apply that ratio to 200 → expect dips to 20–30. That is **exactly** what the ParticleDA plot shows. So the absolute ESS difference (1000s vs 10s) is almost entirely the Np difference; in **relative** terms the two filters are in the same regime.

### 3. (Confounder) Resampling cadence and what ESS measures

This is the biggest visual confounder.

- **LowLevel `ParticleFilter`** resamples *adaptively* — only when ESS drops below a threshold. Between resamples, weights are carried over from prior steps, so ESS=Np means "weights are still well-balanced after several observations." Most of its 200 steps therefore show ESS=Np.
- **ParticleDA `BootstrapFilter`** resamples *every step*. After each resample the weights are uniform; the next observation hits a uniform ensemble; the reported ESS is "how informative was this one observation?" It can never carry healthy weights across steps because the weights reset.

So the LowLevel plot is "ESS at the moments the filter decided to resample" (mostly skipped → looks flat at full Np), and the ParticleDA plot is "per-step likelihood spread from uniform" (always 5–15 % of Np in this regime). Both numbers describe a filter that resamples roughly once every 6–10 steps, but the LowLevel one *hides* the dips behind its adaptive trigger.

### 4. (Confounder) σ_init in observation-space units

LowLevel toy: σ_init = 0.8 with σ_signal ≈ 1 → particles spread by ~80 % of the dynamic range; nearly anything the truth does is in-distribution. ParticleDA glacier: σ_init_θ = 0.05 in log β → β spread of ~50 around 1000 → **ux spread of ~0.05** at obs time. That equals σ_obs. So most particles fall within ~1σ_obs of each other and the likelihood barely discriminates — but the *one* particle that lands closest to the truth still wins by a factor of e¹ or more, which is enough to spike the weights. Combine with every-step resampling → degeneracy.

### 5. (Real degeneracy) Posterior variance collapses to 1.0 in two steps

`particle_variance.png` shows posterior Var(β) crashing from 1.0025 → 1.00005 by t=3 and staying there. With process noise σ_θ=0.007 and resample-every-step, the ensemble cannot regenerate diversity faster than resampling kills it. After a couple of steps every particle is a near-clone of the survivor.

This is a real problem. It means the filter is not actually using its 200 particles — effective N is roughly 1.

### 6. (Real degeneracy) Pressure budget

Same formula as the tsunami sweep: `pressure = n_obs × (σ_signal / σ_obs)²`. With 100 sensors, σ_signal ≈ ux spread ≈ 0.3 (β range 500–1500 → ux range 0.67–2.0), σ_obs = 0.05:

```
pressure ≈ 100 × (0.3 / 0.05)² = 3600
```

That's deep in the "Bootstrap collapses regardless of N" zone (`00_action_plan.md`, Phase B). The LowLevel run for the *same* problem (`particlefilteringwithiceflow.jl`) would face the same pressure; it just isn't logged.

## Why the RMSE is fine even though ESS is bad

Two reasons:

1. **The surrogate is forgiving.** ux = 1e3/β is monotone in β. Even a degenerate ensemble centered on the right β gives the right ux. So the *mean* tracks; only the *uncertainty* collapses.
2. **β doesn't move much per step.** σ_proc_θ = 0.007 → 0.7 % per step. After 50 steps the truth has wandered ~5 %. The PF mean only needs to track that slow drift, which a degenerate ensemble can do as long as it isn't biased.

This is exactly the "RMSE doesn't tell you the filter is healthy" warning from `00_action_plan.md` Phase C.

## Aside — why we didn't just port the LowLevel toy's parameters

**The question:** the simple LowLevel toy (`particlefilteringnonlinear.jl`) uses β directly (not log β), identity observation operator, β ∈ [−1, 1], σ_init = 0.8, σ_obs = 0.1. Why isn't the ParticleDA run set up with those same numbers, so we can compare like-for-like?

The honest answer is that **two different comparisons were getting tangled together**, and one of them does want the LowLevel parameter set.

### Two purposes that need separating

1. **Filter-library sanity check** — "Is ParticleDA doing roughly what a particle filter should do?"
   - For this you want both filters running on *identical, easy* problems. Identity obs, wide prior, β space.
   - The LowLevel toy is the right baseline. We should port *that* to ParticleDA, not modify it to match the glacier setup.

2. **Realistic glacier estimation** — "Will this configuration recover β from velocity observations under realistic noise?"
   - For this you need the parts that make the problem *glaciological*: positivity (log β), velocity observations (not β-at-sensors — that's literally what we're inferring), β values of O(1000), σ_obs matched to InSAR noise.
   - The current ParticleDA run is closer to this; the LowLevel toy is too synthetic to be the reference.

The first run I built jumped straight to (2), so the only filter we have a comparable LowLevel result for (`particlefilteringnonlinear.jl`) is the wrong baseline. That's the gap.

### Which parameter set makes more sense and why

| Choice | LowLevel toy convention | Glacier-realistic convention | Why |
|---|---|---|---|
| State variable | β | **log β** | β must be > 0. Sliding-law coefficients span orders of magnitude (10⁵–10⁸ Pa·s/m for real glaciers); log space makes the prior roughly Gaussian and avoids negative weights from impossible β values. |
| Observation operator | identity | **velocity (surrogate or WAVI)** | We *observe* velocity; we *infer* β. Using identity-on-β trivialises the problem — you don't need a filter for that, you'd just take the obs. The whole reason to use DA is the inverse-problem step. |
| β scale | [−1, 1] | **realistic units (Pa·s/m)** | Doesn't really matter mathematically (rescale and you're fine), but using realistic units forces us to pick σ_obs that's actually defensible from data sheets (InSAR ~10–50 m/yr, GPS ~1–5 m/yr) rather than a dimensionless 0.1 that means nothing. |
| σ_init | 0.8 of signal | needs to be **wide** (e.g. 20–50 % of log β) | The LowLevel toy got away with σ_init = 0.8 because that's "lots". Our 5 % is the actual mistake here — too tight for the problem. **This is the one LowLevel-style change we should keep.** |
| σ_obs | 10 % of signal | matched to **real data noise** | Our 0.05 (5 %) is plausible for clean InSAR; we should sweep this anyway (Phase B of [00_action_plan.md](00_action_plan.md)). |
| σ_proc | 1 % of signal | comparable in **log space** | 0.7 % per step is fine. |
| Np | 10 000 | start at 200, scale up only after B/C | LowLevel could afford 10 000 because each step was cheap. With WAVI eventually in the loop, every particle costs real money. Pressure-budget says larger Np doesn't fix degeneracy anyway. |
| Resampling | adaptive | adaptive *if* ParticleDA supports it | Adaptive is strictly better whenever available; nothing glacier-specific about this. |

**Rule of thumb:** keep glacier-realistic choices for the things that make the *problem* a glacier problem (state variable, observation operator, scale, σ_obs). Adopt LowLevel-style choices for the things that are about *filter hygiene* (wide σ_init, adaptive resampling, eventually larger Np).

### What this means for next steps

The **right** first action isn't to bolt ESS logging onto `particlefilteringwithiceflow.jl`. It's to do *both* comparisons properly:

- **Comparison 1 (filter sanity):** port `particlefilteringnonlinear.jl` to ParticleDA with its parameters preserved (β state, identity obs, σ_init = 0.8, σ_obs = 0.1, Np = 10 000). If ParticleDA's ESS plot looks like LowLevel's on that problem, the library is fine; any remaining gap is purely the resampling-cadence convention.
- **Comparison 2 (realistic):** keep the current ParticleDA glacier setup but apply LowLevel-style filter hygiene — wider σ_init, σ_obs sweep, adaptive resampling. Don't try to make the *problem* simpler; make the *filter* healthier.

The `## What to change next` list below is rewritten to reflect this.

---

## What to change next (ordered by expected impact, cheapest first)

### A. Filter-library sanity check on the LowLevel toy

Port `particlefilteringnonlinear.jl` to ParticleDA *unchanged* (β state, identity obs, σ_init = 0.8, σ_obs = 0.1, Np = 10 000, T = 200). Compare ESS plots directly. This tells us how much of the gap is *resampling cadence* (the every-step-vs-adaptive issue from §3) vs *the problem we picked*. Until we run this we don't know.

### B. Inflate σ_obs

The tsunami sweep showed this is by far the dominant lever. Try `obs_noise_std`: 0.05 → 0.1 → 0.2 → 0.5 → 1.0. Plot ESS / max-weight / RMSE. Expect a knee in ESS somewhere around 0.2–0.5; pick the smallest σ_obs that keeps ESS > 10 % of Np across most steps.

YAML-only change: edit `glacier-code/particleda/glacier.yaml`. No code edits.

### C. Widen σ_init

Currently 5 % in log β. Try 10 %, 20 %, 30 %. This buys the initial ensemble enough spread that the truth always sits inside the prior cloud. Especially important because every-step resampling can't recover diversity later — it has to be present at t=0.

### D. Switch resampling to adaptive (or every-other-step)

ParticleDA's `BootstrapFilter` resamples every step. Look into whether the filter accepts a `resample_threshold` parameter in `FilterParameters` (need to read `src/params.jl` / `src/filters.jl`). If so, set it to e.g. `0.5 * Np`. If not, we have two options:

1. Add it as a small upstream PR — `BootstrapFilter` is the natural place for adaptive resampling.
2. Live with every-step resampling and rely on B + C to make per-step weights healthy.

For the first pass, do (B) + (C) before touching (D).

### E. Switch summary stat to include particles (or sample subset)

`MeanAndVarSummaryStat` loses the particle ensemble. If we want LowLevel-style histograms or to track collapse modes by eye, we need a custom summary stat that retains a subsample of particles per step. Lower priority — only needed if (B)+(C) don't reveal what's happening.

### F. Larger Np

Only after (B)+(C)+(D). The tsunami sweep proved this is the *last* lever, not the first. But once σ_obs and σ_init are right, going from 200 → 1000 will give the visible-on-the-plot Np that makes ESS curves look like the LowLevel toy.

## Run-02 — LowLevel-style hygiene applied

After the discussion above we kept the glacier-realistic *problem* (log β state, surrogate ux observation, β ~ 1000) but adopted LowLevel-style *filter hygiene*: wider σ_init, slightly looser σ_obs, more particles, longer horizon. Saved run-01 to [results/run01_tight_prior/](../glacier-code/particleda/results/run01_tight_prior/) so we can A/B.

### Config diff (run-01 → run-02)

| Parameter | run-01 | run-02 | Reason |
|---|---:|---:|---|
| `nprt` | 200 | **1000** | More particles → absolute ESS in usable range |
| `init_std_theta` | 0.05 | **0.30** | Particles must cover the truth at t=0 (LowLevel had σ_init ≈ σ_signal) |
| `obs_noise_std` | 0.05 | **0.10** | Defensible InSAR-style ~10 % rather than aspirational 5 % |
| `process_std_theta` | 0.007 | 0.007 | Unchanged |
| `n_time_step` | 50 | **200** | Match LowLevel run length so we see whether the filter converges |

### What changed in the results

| Metric | run-01 (tight) | run-02 (LowLevel-style) |
|---|---|---|
| ESS | 1–30 / 200 (≈ 5–15 % Np), no upward trend | 50–400 / 1000 (≈ 5–15 % Np), **upward trend**, occasional spikes to 0.4·Np |
| max(weight) | 0.1–0.65 every step | spikes at the start, then **mostly < 0.1** after step 30 |
| Posterior Var(β) | crashed to ~1.0 by step 3 | takes ~30 steps to settle, stays higher than run-01 |
| RMSE(β) trajectory | 52 → 37 (no real learning) | **410 → 120** (clear convergence) |
| β at first sensor | offset ~30 from truth, no tracking | offset ~400 at t=1 (wide prior), **tracks truth by t≈120** |

Two important things to read from this:

1. **Relative ESS is the same (~5–15 % of Np), but absolute ESS is now usable.** With Np=200 the dips bottomed at 1 — completely degenerate. With Np=1000 the dips bottom at 50, which is enough to actually represent uncertainty. This matches the tsunami-sweep finding: Np doesn't *fix* the regime, but it scales the absolute floor.

2. **The filter now *learns*.** RMSE dropping 410 → 120 over 200 steps, with the pointwise plot showing the PF mean catching up to truth around step 120, is the qualitative behaviour the LowLevel toy showed. It was hidden in run-01 because the prior was too tight to leave any error to correct.

### What the residual ESS pattern means

ESS still bounces around 10–20 % of Np with the same spiky shape. That's not a tuning failure — it's the every-step resampling convention from §3 above. Each step measures "likelihood spread starting from uniform weights", and with `pressure ≈ 100·(0.3/0.10)² = 900` we're still in a high-pressure regime where one observation is enough to make a few particles win. The plot will keep looking spiky no matter how much we tune σ_init, until we either:

- get adaptive resampling into ParticleDA's `BootstrapFilter` (would need an upstream change in [src/filters.jl](../src/filters.jl); not exposed via `FilterParameters` per [src/params.jl](../src/params.jl)), **or**
- switch to ParticleDA's `OptimalFilter`, which uses the optimal proposal and dodges the Bootstrap pressure formula entirely (requires the covariance methods in Community 7 of the graph — a fair amount of extra interface work).

For now, **run-02 is healthy enough to move on with**. Next planned tweaks (only after we confirm convergence is genuine on a few seeds):
- σ_obs sweep at 0.05 / 0.10 / 0.20 / 0.50 to find the knee.
- Replace the surrogate with WAVI (the actual point of the project).
- Then consider Optimal proposal or upstream-PR adaptive resampling.

## Run-03 — reduce observation count (Phase B.3 lever)

After run-02 the per-step ESS was still bouncing around 5–15 % of Np despite the wide prior — pressure ≈ 900 from `100 × (0.3/0.10)²`. The cheapest remaining lever was `n_obs`: drop sensors 100 → 16 by changing `sensor_stride: 16 → 100`. Predicted new pressure: `16 × (0.3/0.10)² = 144` — a ~6× cut.

run-02 saved to [results/run02_wide_prior_100obs/](../glacier-code/particleda/results/run02_wide_prior_100obs/).

### Results

| Metric | run-02 (100 obs) | run-03 (16 obs) |
|---|---|---|
| Pressure | ~900 | **~144** |
| ESS | 50–400 / 1000 (5–40 % Np) | **400–800 / 1000 (40–80 % Np)**, mostly **above 0.5·Np** |
| max(weight) | spikes at start, then mostly <0.1 | one early spike at 0.55, then **mostly <0.02** |
| RMSE(β) | 410 → 120 | 350 → 125 (similar final) |
| Pointwise β tracking | PF mean catches truth by t≈120 | PF mean tracks truth from **t≈30** |

This is exactly what the tsunami pressure formula predicted: cutting `n_obs` by ~6× pushes the run from "high pressure, particles barely informative" to "moderate pressure, ensemble cooperates." The ESS line now sits comfortably above the 0.5·Np reference for the bulk of the run — the canonical "healthy filter" signature.

### Why fewer obs isn't free

Two real costs to keep in mind:

1. **Information thrown away**. With 16 sensors spread across a 40×40 grid we now see ~1 % of cells. The PF mean still tracks the truth because the dynamics couple neighbouring cells through advection, but the *uncertainty* in unobserved regions is genuinely larger than in run-02. We'd see this if we plotted spatial Var(β) — sparse regions wouldn't tighten as much.
2. **Sampling pattern matters**. We picked sensors at stride 100, which is essentially "every 2.5 rows of the grid" — geographically fine. Real InSAR coverage is usually dense in one region and absent in another. When we move to real data we'll need `station_filename` and explicit (x, y) coordinates, not a stride.

For the synthetic glacier benchmark this is the right call — the budget is satisfied and the filter is healthy. For the real-data run we'll likely have to:
- accept σ_obs inflation (Phase B.1) on top of obs reduction, or
- spatially pool obs (B.3 variant: 5×5 super-pixels) rather than subsample, or
- move to OptimalFilter or localised PF.

### Suggested next steps

The run is healthy enough that the *next* useful experiment isn't more PF tuning — it's:

1. **Confirm robustness**: rerun at 3–5 random seeds to make sure the convergence shape is generic, not a lucky obs realisation.
2. **σ_obs knee-search**: at fixed 16 obs, sweep `obs_noise_std` ∈ {0.05, 0.10, 0.20} to find the tightest σ_obs that keeps ESS healthy. Tighter σ_obs → better tracking accuracy when the filter can take it.
3. **Swap surrogate → WAVI** (the actual project goal). The PF plumbing is now provably working; replacing the forward map is a localised change in [glacier-code/particleda/glacier_model.jl](../glacier-code/particleda/glacier_model.jl) (`surrogate_ux!`).

## Files this note refers to

- This run's outputs: [glacier-code/particleda/results/](../glacier-code/particleda/results/)
- LowLevel toy outputs: [glacier-code/results/](../glacier-code/results/)
- Background: [00_action_plan.md](00_action_plan.md), [01_lowlevel_Np10000_observations.md](01_lowlevel_Np10000_observations.md), [04_code_maps.md](04_code_maps.md)
- Config to tune for steps B / C: [glacier-code/particleda/glacier.yaml](../glacier-code/particleda/glacier.yaml)
- Plot driver to rerun after each YAML tweak: [glacier-code/particleda/plot_glacier_pda.jl](../glacier-code/particleda/plot_glacier_pda.jl)
