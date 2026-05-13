# LowLevelParticleFilters — Np=10000 ESS Observations

> **Source experiment:** `particlefilteringnonlinear.jl` (nonlinear advection toy model, 40×40 grid = 1600 state dims), run with `Np = 10000` instead of the original 100.

## What the new plot shows

The ESS-evolution plot from this run has a very different *shape* from the earlier Np=100 run:

- **Most timesteps sit at full ESS = 10 000** (the line at the top of the plot)
- **About 10 sharp dips** scattered between t ≈ 90 and t ≈ 200
- **Each dip bottoms out at ESS ≈ 1 000–1 500** (i.e. 10–15 % of N)
- **Dips snap back immediately** (the spikes are one-timestep-wide)
- The 0.5 × Np threshold line (5 000) is crossed only at the dips

This is *visually* dramatic — those vertical lines look like cliffs. But numerically it's a healthier picture than what we saw with the tsunami benchmark and even a healthier picture than your Np=100 run.

## What "not that promising" actually means here

The intuition was: "more particles → fewer/smaller dips." What we got is: "more particles → much fewer dips, but the dips that remain are *proportionally* the same depth."

Quantitatively:

| Run | full-ESS fraction | dip depth (as % of N) | dip frequency |
|---|---:|---:|---:|
| Np=100 (earlier) | ~50 % of timesteps | 10–40 % of N | very frequent (multiple per 10 steps) |
| Np=10000 (this run) | ~95 % of timesteps | 10–15 % of N | rare (~10 events in 200 steps) |

So N **did** help — significantly — but it didn't *eliminate* the bad events. It just made them rarer. The depth of the remaining events scales with N (10 % of 100 = 10, 10 % of 10 000 = 1 000), so on an absolute scale 1 000 surviving particles is plenty.

**Verdict: this is a working, stressed-but-not-degenerate filter.** Not a failure. The interpretation "ESS doesn't seem that promising" is reading the visual drama rather than the numbers.

## Why this is EXACTLY what the tsunami sweep predicted

The pressure-budget formula:

```
pressure = n_obs × (σ_signal / σ_obs)²
```

For your advection toy model:

- `n_obs` = 100 (sensors)
- `σ_obs` = 0.1
- `σ_signal` ≈ 1 (sine wave amplitude)
- **pressure = 100 × (1/0.1)² = 10 000**

That puts you **well above the "healthy" band (≤ 100)** and well into the "Bootstrap will be stressed" regime. Going from N=100 to N=10 000 doesn't change pressure at all — pressure has no N in it.

So the prediction was: even at large N, you'll see ESS dips whenever a particularly informative observation lands. That's exactly what the plot shows: most observations are unremarkable (ESS stays at N), but the occasional sharp event still concentrates weights.

**Increasing N from 100 → 10 000 did the right thing for the wrong reason.** It didn't reduce per-event pressure; it just gave you more particles so each event leaves more survivors. The events themselves are still there.

## What this tells us for the glacier model

This is good news, with one caveat.

### Good news

1. **Bootstrap can work even with high pressure, IF N is large enough that 10-20 % survivors is still a meaningful ensemble.** Your 1 000 surviving particles at the bottom of a dip is more than enough for posterior inference.
2. **The pressure formula is predictive but not prohibitive.** It says "you're stressed", not "you're broken". Stress is recoverable via resampling.
3. **Your existing LowLevel setup is a viable template** for the glacier work. The wide σ_init (=0.8) and adaptive resampling are doing real work.

### Caveat

1. **You're paying 100× compute for ~5× fewer events.** Going from Np=100 to Np=10 000 made each timestep ~100× slower. If your model evaluations are expensive (real glacier forward model — ISSM, Elmer/Ice, etc.), this is brutal.
2. **For the same effort, you could try cutting pressure first** — and then large N becomes unnecessary. The tsunami sweep showed that going from σ_obs=0.01 to σ_obs=5.0 (loosening 500×) eliminated collapse at *just* N=500. That's the cheap fix.

## Concrete next step for the glacier project

Before porting anything, **run a tiny pressure-vs-N sweep on this same toy model** to verify the lesson on hardware you control:

| Run | Np | σ_obs | What it tests |
|---|---:|---:|---|
| A | 100 | 0.1 (your baseline) | reproduce your earlier run |
| B | 10 000 | 0.1 (current) | already done — that's the plot above |
| C | 100 | 1.0 (10× looser) | does loosening σ_obs alone fix it at low N? |
| D | 1 000 | 0.3 | middle ground |

You'd predict from the pressure formula:

- A: pressure 10 000, N=100 → frequent dips (✓ matches Np=100 plot)
- B: pressure 10 000, N=10 000 → rare dips (✓ matches this plot)
- C: pressure **100**, N=100 → healthy ESS throughout
- D: pressure **1 111**, N=1 000 → intermediate

If C beats B at 100× less compute, you've proven the σ_obs lever quantitatively on your own toy model, and the glacier-model plan can confidently lean on it.

## Practical recommendations

1. **Keep your Np=10 000 LowLevel run as the "raw power" baseline** — useful for verifying everything else against.
2. **Run experiment C** (Np=100, σ_obs=1.0) as your next ESS-diagnostic experiment. ~100× faster than Np=10 000, and if the prediction is right, it should give a flatter ESS curve (no dips at all). If it does, you've replicated the tsunami finding on your own hardware.
3. **Run experiment D** (Np=1 000, σ_obs=0.3) as a compromise check — moderate compute, moderate pressure.
4. **Move to glacier physics only after** this 4-run sweep on the toy model confirms the methodology end-to-end. The toy is cheap; the glacier won't be.

## What I'm logging into the action-plan file

In `00_action_plan.md` under **Phase B**, the lever order remains:

1. σ_obs (still the best lever)
2. σ_init (your existing run already uses a wide one — good)
3. n_obs (last lever before throwing N at it)
4. N (only after the above; expensive)

Your Np=10000 plot reinforces this ordering: option 4 alone is expensive and only partially effective. Options 1-3 should always be tried first.

---

## TL;DR for the notebook

- Np=10 000 ESS plot **looks dramatic but is fine**: 95 % of timesteps healthy, dips bottom at ~10 % of N (= 1 000 survivors), adaptive resampling handles them.
- Confirms the tsunami sweep's prediction: **N alone doesn't fix pressure, it just makes the consequences less catastrophic per event.**
- For glacier: **do the σ_obs sweep on the cheap toy model first** (C and D above) to verify the methodology, then port to glacier physics.
- Pressure for the toy model = 10 000 (huge). Pressure for glacier will be your first calculation once you know your real σ_obs.
