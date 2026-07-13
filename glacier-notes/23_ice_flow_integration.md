# Ice-flow model integration — β → velocity RMSE pipeline

Purpose: wire the WAVI ice-flow forward model into the particle-filter
workflow so we can evaluate whether the filter's β-space learning
translates into velocity-space accuracy.

## What was built

**Two files, two environments.**

- `glacier-code/particleda/ice_flow.jl` — thin module `IceFlow` wrapping
  WAVI.jl. Cached grid, bed, and initial conditions match `main.jl` but
  rescaled to the 40×40 grid the PF uses.
- `glacier-code/particleda/rmse_experiments/experiment_09_velocity_rmse_pf.jl`
  — **Stage 1**. Runs under `--project=test`. Executes the bootstrap PF
  and dumps truth β + ensemble-mean β trajectories + β-RMSE + ESS to
  HDF5.
- `glacier-code/particleda/rmse_experiments/experiment_09_velocity_rmse_post.jl`
  — **Stage 2**. Runs under the **default** Julia env (needs WAVI, whose
  HDF5/NetCDF stack isn't compatible with the `test` env). Loads the
  HDF5, runs WAVI once on the truth β and once on the mean β at every
  step, computes velocity RMSE, plots both β-RMSE and velocity-RMSE on a
  twin-axis figure.

Why two stages, two envs: WAVI's underlying `libhdf5` / `libnetcdf`
artifacts conflict with what ParticleDA pulls in. Splitting the
pipeline means neither side needs to know about the other's
dependencies. HDF5 is the handoff.

## Key API adaptation

`main.jl` passed the basal-drag field to WAVI via `Params(beta = …)`.
The currently-installed WAVI (`~/.julia/packages/WAVI/688R8`) renamed
this to **`weertman_c`** — same field, new keyword. The wrapper does
the rename so downstream code can keep saying "β".

## First-run numbers (linear advection, run10-ish defaults, T=100h)

| Metric | t = 0 | t = 100 h | Δ |
|---|---:|---:|---:|
| β-RMSE | 509.5 Pa·s/m | 347.0 Pa·s/m | ↓ 32 % |
| velocity-RMSE (speed magnitude) | 1717 m/yr | 1453 m/yr | ↓ 15 % |

Output figure: `results/rmse_analysis/exp09_velocity_rmse.png` — β-RMSE
as a continuous blue line (left axis, Pa·s/m) and velocity-RMSE as red
dots (right axis, m/yr) on a shared time axis.

Compute cost: **202 WAVI solves in 3.9 min** (one for truth + one for
mean, at every step of T+1=101). Each solve is ~1.55 s after
compilation.

## Interpretation

**β-RMSE improvement translates to velocity-RMSE improvement, but not
1-to-1.** The filter is doing what the prompt hoped for — better β
estimates *do* yield more accurate velocity predictions. But the
transfer is lossy: β dropped 32 %, velocity dropped 15 %.

Why the ratio is < 1: WAVI's basal-drag → velocity map is a smoothing,
elliptic operator (momentum balance is diffusive). Local β errors get
averaged out by the solve — nearby cells partially compensate for each
other. So even a badly wrong β can produce a reasonable velocity if the
errors cancel spatially. This is a general property of ice-flow
inversion and one of the classic reasons β estimation is hard: you can
be wrong in β-space and right-ish in velocity-space, or vice versa.

Absolute magnitudes: mean speeds are in the thousands of m/yr
(shelf-like). 1400 m/yr RMSE at the end is ~30 % of typical speed —
not spectacular, but the direction is right.

## Two things worth remembering

1. **Cosmetic bug in the post script's "mean truth-speed" print.**
   Assignment inside a `for` loop doesn't escape without a `global`
   declaration in Julia scripts. The RMSE numbers themselves are
   correct — the diagnostic line at the end prints `NaN` because of
   this. Doesn't affect the plot or the metric.
2. **WAVI's boundary conditions dominate velocity RMSE.** We inherited
   `u_iszero = ["north"], v_iszero = ["south"]` from `main.jl`. Applied
   to a domain with strong sinusoidal β shear, the v-component reaches
   ±30 000 m/yr near the boundaries. Not a bug — but it means our
   velocity RMSE includes those boundary effects. Options for later:
   mask the boundary cells before computing RMSE, or switch to a
   different velocity metric (e.g. u-only at sensor points).

## What this unlocks

The pipeline is a **general diagnostic**, not just a one-shot
experiment. Any PF run whose truth + ensemble-mean β trajectories we
dump to HDF5 can be post-processed to yield velocity RMSE.

Natural extensions:
- Sweep obs cadence / n_obs / σ_obs and ask "how do velocity-RMSE
  curves compare across configs?"
- Multi-seed averaging: repeat Stage 1 with different seeds, do Stage 2
  on each, plot velocity-RMSE bands.
- Different priors (e.g. Evensen pseudo-random β) once the wave code is
  wired in — Stage 2 doesn't care what generated the trajectory.

## See also

- `22_lw_rpf_reference.md` — the LW + RPF setup Stage 1 uses (linear
  advection variant is a straightforward branch).
- `21_numerical_diffusion.md` — why LW beats first-order upwind.
- `rmse_experiment_log.md` — Stage 1 registers a row for exp09 there.
