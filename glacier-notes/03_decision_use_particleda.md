# Decision: Use ParticleDA from the start for the glacier model

> Supersedes the "prototype on LowLevel first, port later" advice in earlier files. After thinking through the actual cost/benefit, **ParticleDA is the right tool for this project from day one**, not a future migration target.

## Your argument, restated

You raised five points. They're all correct, with some nuance on one of them:

1. **Avoid the porting risk** — building on LowLevel and then re-doing it on ParticleDA when scale demands is real friction.
2. **`n_integration_step` for sub-stepping** — your glacier forward model (SIA, SSA, or full-Stokes) will need internal sub-steps for CFL stability when you assimilate every 5 s or 10 s of simulated time. ParticleDA's LLW2d shows exactly this pattern.
3. **Scattered observation locations via a `.txt` file** — InSAR pixels / GPS stations / feature-tracking points → arbitrary (x, y) coordinates. The LLW2d station-loading pattern is the right abstraction.
4. **LLW2d is a near-perfect template** — 2D field with Matérn GRF noise, scattered observations, sub-stepping, HDF5 output. ~700 lines of clean Julia you can copy + modify rather than writing from scratch.
5. **LowLevel doesn't document a 2D field example** — true. Your toy advection works, but you're off the documented path. ParticleDA's documented use case *is* 2D PDE-driven systems.

The only point that needs nuance is your sixth one: *"Bootstrap with MPI can help with the calculations as ice-flow is expensive."* True — MPI gives you a faster Bootstrap. But MPI does **not** fix particle degeneracy. The pressure formula (`n_obs × (σ_signal/σ_obs)²`) still applies. So MPI lets you afford more particles in wall-clock terms, but you still have to tune σ_obs / σ_init / n_obs to actually make Bootstrap viable. **Library choice ≠ degeneracy fix.**

## Why I now agree with switching

Three concrete reasons that tip the balance:

### 1. Inheritance from LLW2d isn't optional — it's the killer feature

The LLW2d model in `test/models/llw2d.jl` already implements every piece of infrastructure you need:

- **Multi-variable state** (height + 2 velocity components on a 51×51 grid) — exactly the shape of your glacier state (β + ice thickness + maybe velocity components)
- **Sub-stepped deterministic update** (the `n_integration_step` loop) — exactly the pattern for your CFL-constrained ice-flow solver
- **Gaussian random field stochastic forcing** — for your sub-grid process noise (uncertain accumulation, melt, sliding parameters)
- **Scattered observations** via `station_filename` → real-world observation pixel coords
- **HDF5 output** for state mean / variance / weights at every timestep
- **MPI distribution** of particles across ranks, with custom variance reductions
- **YAML configuration** so you can run sweeps without recompiling

You'd reimplement all of this for LowLevel. That's weeks of engineering you avoid by using the template.

### 2. The forward-model code is the same in both libraries

This is the key insight that makes the "port later" argument weaker than I made it sound earlier:

Your ice-flow + β-advection physics — the actual scientific code — is **library-agnostic**. It's a function that takes (state, dt) and returns (new state). Whether you call that function from a LowLevel `dynamics` callback or from a ParticleDA `update_state_deterministic!` method, the physics is identical.

So the choice between libraries is really:

| | LowLevel | ParticleDA |
|---|---|---|
| Lines of wrapper code | ~10 (two callbacks) | ~200 (full model interface) |
| Lines of physics | thousands | thousands |
| HDF5 logging | write yourself | built-in via `verbose: true` |
| Scattered obs handling | write yourself | built-in via `station_filename` |
| MPI scaling | impossible | built-in, opt-in via `mpiexec` |
| Sub-stepping | write yourself | template in LLW2d |

The 190 extra lines of wrapper are a one-time cost. The infrastructure you get for them recurs every time you change a config, scale up, or want a new diagnostic.

### 3. Ice-flow forward model = expensive per particle = MPI matters

When you add an ice-flow solver (anything beyond toy advection), the per-particle per-timestep cost goes up significantly:

- Toy advection (your current LowLevel): ~10 µs per particle-step (40×40 grid, ~1600 cells × cheap update)
- LLW2d shallow water: ~3.7 ms per particle-step (51×51 grid × 10 sub-steps × moderate update) — 370× slower than toy
- SIA (shallow-ice approximation) glacier: ~10–100 ms per particle-step (depending on grid + nonlinear viscosity solve)
- SSA (shallow-shelf approximation) glacier: ~100 ms – 1 s per particle-step (elliptic solve per step)
- Full-Stokes glacier: ~1–10 s per particle-step (full 3D nonlinear solve)

At SIA/SSA cost levels, even N = 500 particles for 100 timesteps becomes 5 min – 50 min serial. With 32 MPI ranks, divide by 32 → 10 s – 100 s. **That's the difference between "iterate ten times a day" and "one run a day".**

You only get that speedup with MPI. LowLevel can multi-thread but not multi-node.

## Where ParticleDA still costs you something

Being honest about what you'll pay:

### Upfront cost
- **One-time learning curve**: the model interface has ~10 functions to implement vs LowLevel's 2. Use LLW2d as a template, but you'll still spend ~1 week getting comfortable with it.
- **MPI setup**: `MPI.jl` needs to know which MPI implementation to use (system OpenMPI/MPICH or the JLL bundled one). On a cluster, this is non-trivial — needs talking to the sysadmin or reading docs. On your laptop, `mpiexec` from the bundled MPItrampoline works out of the box.

### Recurring cost
- **HDF5 writes per timestep** when `verbose: true` — ~25 % of wallclock at single-rank. Acceptable when MPI is on (it gets amortised) but switch to `verbose: false` for production sweeps where you don't need per-step state snapshots.
- **The `optimize_resampling: true` default** runs HiGHS LP every timestep — ~10 % overhead at single-rank. Useful at multi-rank to minimise inter-node particle copies, but pure overhead at 1 rank. Default is fine for your eventual scaled runs.

### Things to verify before committing
- Confirm `mpiexec -n 4 julia --project=test runtests.jl` works on your laptop. If it doesn't, MPI installation needs work *before* you start writing glacier code.
- Confirm your eventual cluster has MPI.jl-compatible MPI (most do — OpenMPI, MPICH, Cray MPI all work).

## Concrete plan

### Phase 0 — verify MPI works locally (1 day)

```bash
cd ~/Desktop/particleDA/ParticleDA.jl
mpiexec -n 4 julia --project=test test/mpi_filtering.jl
```

If this runs without error, you're set. If MPI.jl complains, fix the install before doing anything else. (Most likely cause: MPI.jl falling back to `MPItrampoline` instead of system OpenMPI. Usually fixable with `JULIA_MPI_BINARY=MPItrampoline` or `=system` env var depending on what you want.)

### Phase 1 — copy LLW2d, rename, gut the physics (1 week)

```bash
cp -r ParticleDA.jl/test/models/llw2d.jl YourProject.jl/src/glacier_model.jl
```

Then:

1. Rename the module: `LLW2d` → `GlacierModel`
2. Rename the parameter struct: `LLW2dModelParameters` → `GlacierModelParameters`
3. Keep ALL the infrastructure (`init`, `sample_initial_state!`, `write_model_metadata`, etc.) — you just need to replace the physics
4. Replace `update_state_deterministic!` with your ice-flow + β-advection update. **Reuse the `n_integration_step` loop pattern.**
5. Keep `update_state_stochastic!` as-is (Gaussian random field forcing — appropriate for glacier process noise)
6. Adapt `sample_observation_given_state!` to map state → surface velocity at observation points (your H operator)
7. Keep the station-file pattern; supply your real GPS / InSAR coordinates

At the end of this week you should be able to run:

```bash
julia --project=test your_glacier_benchmark.jl
```

…and get the same kind of HDF5 output as we got for tsunami — `state_avg/tNNNN/{β, h, u, v}`, `weights/tNNNN`, etc.

### Phase 2 — sweep methodology applied to glacier (1–2 weeks)

This is the **`00_action_plan.md` Phases A–C**, just executed on your glacier model instead of LowLevel toy:

- Phase A: baseline run at small N, real σ_obs, observe ESS
- Phase B: σ_obs / σ_init / n_obs sweep if it collapses
- Phase C: accuracy tuning + posterior calibration

All running on **1 MPI rank initially** for fast iteration. Don't engage MPI yet — your model is still cheap (SIA or simple SSA), so single-process is fine.

### Phase 3 — MPI scaling (when forward model gets expensive)

When you upgrade to a more expensive forward model (full SSA, full Stokes, finer grid, or just need more particles):

```bash
mpiexec -n 8 julia -t 4 --project=test your_glacier_benchmark.jl
```

Validate scaling on a problem you've already solved at 1 rank — same input, more ranks, should give same answer in less time (modulo a different RNG path per rank).

### Phase 4 — Production / cluster deployment

When ready to go to a real cluster (Archer2, etc.):

- Build a Slurm/PBS script that launches `mpiexec` with the right rank count
- Set `verbose: false` in YAML for the final run — no need for per-step HDF5 in production
- Output only at coarser intervals (we could add a "save every N timesteps" parameter to the model module if needed)

## The key insight that changes my advice

I'd been treating "LowLevel vs ParticleDA" as "easy now / complicated later" vs "hard now / capable later." But the actual trade is "**no infrastructure / write everything**" vs "**inherit infrastructure / write only physics**." For a 2D field model with scattered observations, sub-stepping, HDF5 logging, and eventual MPI needs — ParticleDA's infrastructure is exactly what you'd build anyway in LowLevel, just less battle-tested. So you might as well inherit it.

The earlier "prototype on LowLevel first" advice was right *if* your project was: small state, simple obs, one-shot research code. For a project that's: 2D physical model + scattered real observations + planned scaling to ice-flow + multi-year research effort — **ParticleDA is the right choice from day one.**

## Caveats that don't go away

- **Bootstrap-PF degeneracy is library-agnostic.** Both LowLevel and ParticleDA implement vanilla Bootstrap. The pressure-budget formula applies equally to both. Library choice doesn't fix collapse; σ_obs / σ_init / n_obs choices do.
- **MPI requires explicit `mpiexec`.** Without it you're running ParticleDA in its slowest mode (see `02_particleda_mpi_explanation.md`).
- **LowLevel still has uses.** Specifically: sanity-checking ParticleDA results on a simplified 1D version of your problem; quick experiments where the model interface boilerplate isn't worth the friction. Keep it in your toolkit, just don't build production code on it.
- **The HDF5-write-per-step overhead** is real at single-rank. Turn off `verbose: true` for sweeps once you trust the filter.

## Updates to other files

- `00_action_plan.md` Phase D — flip the default recommendation. Was "LowLevel for prototyping, ParticleDA for scaling." Now: "ParticleDA from day one; LowLevel only for sanity checks."
- `01_lowlevel_Np10000_observations.md` — keep as historical record of the toy-model experiment. The σ_obs lever finding still applies; the library choice for production has moved.

## One-liner for the notebook

> Use ParticleDA from day one. The LLW2d model in `test/models/llw2d.jl` is your template — clone it, rename, replace the physics, keep all the infrastructure. The library choice doesn't fix Bootstrap degeneracy (σ_obs / σ_init / n_obs tuning still required), but it gives you HDF5 logging, scattered observations, sub-stepping, YAML config, and MPI scaling for free.

---

## Action items

- [ ] Phase 0: verify `mpiexec -n 4 julia --project=test test/mpi_filtering.jl` works on your laptop
- [ ] Phase 1: clone `test/models/llw2d.jl` → `glacier_model.jl`, replace physics with β + ice-flow
- [ ] Phase 1: write a glacier-specific YAML config, with real GPS/InSAR station coordinates in a separate `.txt`
- [ ] Phase 2: run Phase A baseline from `00_action_plan.md` on the new glacier model
- [ ] Phase 2: σ_obs sweep if needed (Phase B)
- [ ] Phase 3: enable MPI when forward model cost demands it
- [ ] Phase 4: cluster deployment
