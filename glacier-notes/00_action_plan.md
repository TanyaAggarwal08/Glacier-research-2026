# Glacier Model — Particle Filter Action Plan

> **Companion file:** Bootstrap-PF findings from the LLW2d tsunami benchmark are in `~/.claude/plans/check-if-the-code-typed-shell.md`. Refer back to it for the pressure-budget formula derivation, numerical sweep results, and the FAQ on RMSE / `n_integration_step` / dt mistakes.
>
> **Sibling files in this folder:**
> - `01_lowlevel_Np10000_observations.md` — analysis of the toy-model run at N=10 000 and what it confirms about the tsunami findings.

## Context

Goal: assimilate **surface velocity observations** into a glacier model to estimate the **basal friction coefficient β** via the underlying ice-flow physics. The Particle Filter is the algorithm of choice because (a) we expect non-Gaussian posteriors on β, and (b) we want a principled uncertainty quantification, not just a point estimate.

Two candidate Julia packages:

- **LowLevelParticleFilters.jl** — already prototyping in. Multi-threading, no multi-node. Adaptive resampling built in.
- **ParticleDA.jl** — multi-threading **and** multi-node MPI. We benchmarked it on a tsunami model and learned what makes Bootstrap-PF degenerate; the lessons transfer.

This file is the running notebook for that decision and for the glacier-specific parameter sweep.

---

## What we learned from the tsunami benchmark (transferable)

(Full details in the companion plan file. Key transferable points only here.)

### The pressure formula

```
pressure  =  n_obs × (σ_signal / σ_obs)²       per timestep
```

Predicts Bootstrap-PF collapse risk:

- **pressure ≪ 10**  → Bootstrap is fine
- **pressure ~ 10–100** → borderline (the "knee")
- **pressure ≫ 100** → Bootstrap collapses regardless of N

### Three knobs that control collapse (in order of impact)

1. **σ_obs** — the dominant lever. Loosening σ_obs by 10× can shift you from collapsed to healthy.
2. **σ_init** (initial-state spread) — under-appreciated. A wide initial prior gives the ensemble somewhere to start. The user's existing LowLevelParticleFilters run works partly because σ_init = 0.8 was used vs σ_signal ≈ 1 — particles cover the prior well.
3. **n_obs** — fewer observations means less pressure, but only large reductions (e.g. 15 → 1) matter; moderate cuts (15 → 5) don't help.

### What does NOT help, in order of futility

1. **More particles**. Going N = 50 → 10 000 on tsunami produced *zero* improvement in ESS. The cost was 200× compute for no benefit. (See `01_lowlevel_Np10000_observations.md` for why your own LowLevel run at N=10 000 confirms this on a different toy model: events become rarer but proportionally just as severe.)
2. Switching libraries (LowLevel vs ParticleDA). Both implement vanilla Bootstrap. Library choice is about *scale*, not *correctness*.

### Why your existing LowLevel run looks OK with Np=100

Your code at `particlefilteringnonlinear.jl` works because:

- σ_init = 0.8 vs σ_signal ≈ 1 → genuinely diverse starting ensemble
- σ_obs = 0.1 vs σ_signal ≈ 1 → pressure per obs is moderate
- Adaptive resampling triggers only when ESS drops — efficient
- State dim 1600 is small enough that 100 particles cover the prior meaningfully

The ESS plot showing oscillation between full N and ~10–40 is the **healthy stress regime** — not degeneracy. Don't break what's working.

---

## Phase A — Set up the minimum viable glacier test

Before doing any sweep, build a **single** representative test case you can run cheaply (≤ a few minutes) and reproducibly.

### Decisions to make first

| Item | What to choose | Notes |
|---|---|---|
| Spatial domain | Smallest glacier / patch that exhibits the physics you care about | Single mountain glacier, ~50×50 grid is enough to start |
| State vector | β field (the thing you're inferring) + any prognostic variables (ice thickness h, velocity components if part of the state) | Start with β-only if your physics can hold the velocity solve outside the filter (i.e. β → u via solving the momentum equation each step) |
| Time horizon | Few simulated years, sub-annual timestep | Enough for one or two distinct seasonal signals if relevant |
| Forward model | Whatever solver you're committing to (ISSM, Elmer/Ice, custom, …) | Wrap it so the PF can call `forward(state) → state'` |
| Observations | Realistic synthetic obs first: take a "true" β, simulate the forward model, sample velocity at chosen locations with realistic noise | Use synthetic before real data — lets you compute ground-truth RMSE |
| Observation noise σ_obs | **The actual noise of your real data source.** InSAR vel: ~10–50 m/yr typically. GPS: ~1–5 m/yr. Feature tracking: tens of m/yr. | This is the number you'll need to defend later |
| σ_init for β | Wide enough to cover physically plausible β range (~10⁵ – 10⁸ Pa s/m for typical sliding laws). A factor 2-3× on log-β is reasonable. | DO NOT pick a narrow Gaussian around your initial guess |
| n_obs | All velocity observation pixels you intend to use | Don't decimate yet — let the sweep tell you if you need to |
| N (initial) | 100 | Cheap; same as your existing LowLevel run |

### What to record on this baseline run

- ESS over time
- RMSE of posterior-mean β against true β (since this is a synthetic test, you have the truth)
- Wall-clock per filter step
- Max weight per step (collapse indicator)
- Visual snapshot of posterior-mean β field at a few times

If your baseline ESS plot looks like your existing tsunami plot (oscillating between full N and ~10–40), you're in the working regime — go to Phase C. If ESS pegs at 1 throughout, go to Phase B first.

---

## Phase B — If the baseline collapses: apply the levers, in order

Run **one knob at a time** so attribution is clean. Each variant should reuse the same seeded synthetic-truth and seeded filter — only one parameter changes.

### B.1 — Inflate σ_obs

Try `σ_obs_filter = α × σ_obs_real` for α = 1, 2, 5, 10, 20. Plot ESS-over-time for each. The "knee" is where ESS first stays healthy throughout.

This is **observation error inflation**, a standard technique in operational data assimilation. It's not cheating — it acknowledges that filter likelihoods are over-confident in high-dim systems and need calibration.

**Decision criterion:** smallest α that gives ESS > 10% of N for ≥ 90% of timesteps **and** RMSE not noticeably worse than smaller-α runs.

### B.2 — Widen σ_init (if B.1 didn't fully fix it)

Try `σ_init_β = β × {0.5, 1, 2, 5}` of your default spread. Wider = more diverse ensemble. Cost: more model evaluations spent on implausible starting points, but in a single-shot filter that's a few percent.

### B.3 — Reduce n_obs (last resort before going to bigger N)

If your real data has 10⁵ velocity pixels, you're not actually using 10⁵ independent observations — neighbouring pixels are correlated. Try:

- Random spatial subsampling: pick 10², 10³, 10⁴ observation points
- Spatial pooling: average velocity in 5×5 super-pixels before assimilation
- Temporal subsampling: assimilate every 5th obs instead of every one

Each of these reduces effective n_obs without losing much real information.

### B.4 — Only after B.1–B.3: try larger N

If you still need more particles after fixing the pressure budget, then scale N. *But* if pressure is still > 100 after tuning σ_obs / n_obs / σ_init, more N won't help (tsunami sweep proved it; your own Np=10 000 LowLevel run also confirmed it — see `01_lowlevel_Np10000_observations.md`). At that point Bootstrap is structurally wrong for your problem and you need:

- **Localised PF** (split the state into spatial blocks, run separate PFs per block) — not in ParticleDA or LowLevel out of the box
- **Hybrid EnKF + PF** — also not directly available
- A different inference framework (e.g. EnKS for ice-sheet inversions — see ITSLIVE assimilation papers)

---

## Phase C — When the baseline works: tune for accuracy

Healthy ESS doesn't automatically mean good β estimates. Now optimise for **RMSE** against the synthetic truth.

1. **Tighten σ_obs** (within the regime where ESS stays healthy) — this trades ensemble health for tracking accuracy. Find the sweet spot.
2. **Test sensitivity to N** at the chosen (σ_obs, σ_init): does going from N=100 to N=500 reduce RMSE? If yes, scale up. If no, save the compute.
3. **Check posterior calibration**: at known truth values, does the posterior credible interval cover the truth ~95% of the time across many synthetic trials? This is the test that says your uncertainty quantification is honest.

---

## Phase D — Infrastructure decision: ParticleDA from day one

> **Updated decision** — see `03_decision_use_particleda.md` for the full rationale. The earlier "prototype on LowLevel, port later" advice has been superseded.

**Recommendation: use ParticleDA from the start**, with these justifications:

- The LLW2d model in `test/models/llw2d.jl` is a near-perfect template for your glacier model (2D field, scattered observations, sub-stepping, Matérn GRF noise, HDF5 output)
- Forward-model code (β + ice-flow physics) is library-agnostic — the wrapper is the only difference, and ParticleDA's ~200 lines of wrapper give you HDF5 logging, scattered obs, sub-stepping, MPI, and YAML config for free
- Ice-flow forward models will be expensive per particle → you'll need MPI eventually → start with the library that supports it

**LowLevel role:** kept as a sanity-check tool only. Don't build production code on it. The `particlefilteringnonlinear.jl` toy advection run remains useful as a quick 2D-field cross-check.

**Caveats that don't go away:**
- Bootstrap-PF degeneracy is library-agnostic. The pressure formula (`n_obs × (σ_signal/σ_obs)²`) still governs collapse risk. MPI gives you faster runs, not collapse immunity.
- MPI in ParticleDA is **opt-in via `mpiexec`**. Launching with plain `julia` gives single-rank ParticleDA, which is slower than LowLevel (see `02_particleda_mpi_explanation.md`).
- HDF5-write-per-step (`verbose: true`) adds ~25 % overhead at single-rank. Turn off for production sweeps.

The phased implementation plan now is:

1. Verify MPI works locally — `mpiexec -n 4 julia --project=test test/mpi_filtering.jl`
2. Clone LLW2d, rename to GlacierModel, replace physics
3. Run Phase A baseline at 1 rank for fast iteration
4. Engage MPI when ice-flow cost makes 1-rank infeasible
5. Cluster deployment with `verbose: false` and Slurm/PBS

Full details: `03_decision_use_particleda.md`.

---

## Operational checklist (for the actual implementation)

- [ ] Define synthetic-truth setup (true β field, forward model, observation simulator)
- [ ] Implement RMSE diagnostic against synthetic truth
- [ ] Implement ESS / max-weight diagnostics per timestep
- [ ] Implement CSV-row-after-each-run logging (same pattern as tsunami sweep)
- [ ] Phase-A baseline run, plot ESS + RMSE
- [ ] If collapsed: Phase-B sweeps (σ_obs first)
- [ ] If healthy: Phase-C accuracy tuning
- [ ] Phase-D infrastructure decision (only after the science is validated)
- [ ] Replication of the chosen config with **real** velocity data (not synthetic)
- [ ] Posterior calibration check (does CI coverage match nominal?)

---

## Open questions to resolve before Phase A

These determine the cost and viability of every subsequent step:

1. **What is your forward model?** ISSM, Elmer/Ice, IcePack.jl, custom shallow-ice / SSA / Stokes solver? Wall-clock per single forward step determines feasibility of N×T model evaluations.
2. **How large is the state vector?** Just β (one scalar per grid cell) = nx × ny dimensions. Or β + ice thickness + temperature = 3× that. Bigger state = more memory = sooner you need MPI.
3. **What is the real σ_obs of your velocity data?** InSAR speckle tracking ~ tens of m/yr; offset tracking ~ tens of m/yr; GPS ~ m/yr.
4. **Time-dependent or steady-state inversion?** Time-dependent → you need the PF to track. Steady-state with one observation epoch → consider variational / MCMC over PF; PF shines when you have a time series.

Sketch answers in this file as you decide.

---

## Files / location

- This file: `~/.claude/glacier-model/00_action_plan.md`
- Tsunami findings (companion reference): `~/.claude/plans/check-if-the-code-typed-shell.md`
- Future per-experiment files should go in this directory too:
  - `01_lowlevel_Np10000_observations.md` — ✓ already there
  - `02_phaseA_baseline.md` — once you've defined the synthetic setup and run the baseline
  - `03_phaseB_sigma_obs_sweep.md` — results of the σ_obs sweep
  - `04_phaseC_accuracy_tuning.md`
  - `05_infrastructure_decision.md`
  - etc.

The numbering keeps them sortable in a file browser.
