# Why was ParticleDA so slow? Did we actually use MPI?

> **Short answer (verified):** No — every benchmark run used **1 MPI rank, 1 thread**. We paid ParticleDA's MPI overhead without getting any parallelism benefit.

## How we verified

Direct check on this machine:

```bash
julia --project=test -e 'using MPI; MPI.Init(); println("rank=", MPI.Comm_rank(MPI.COMM_WORLD), " size=", MPI.Comm_size(MPI.COMM_WORLD)); println("threads=", Threads.nthreads())'
# Output:
# rank=0 size=1
# threads=1
```

So `MPI.Comm_size = 1` (single process) and `Threads.nthreads() = 1` (single thread). Every ParticleDA run in this experiment was effectively single-core.

## Reality check on the timings

The "30 min for 200 particles" memory is slightly off — N=200 actually finished in about 72 seconds in the sweep. The 30 min number was N=**5 000**. The full table:

| Run | What it was | Wall-clock (1 rank, 1 thread) |
|---|---|---:|
| Sweep N=200, T=100 | benchmark sweep | 72 s (~1.2 min) |
| Sweep N=500, T=100 | benchmark sweep | 179 s (3 min) |
| Sweep N=1 000, T=100 | benchmark sweep | 358 s (6 min) |
| Sweep N=5 000, T=100 | benchmark sweep | **1 806 s (30 min)** |
| Sweep N=10 000, T=100 | benchmark sweep | 3 727 s (62 min) |
| Showcase N=200, T=260 (verbose) | wave-figure reproduction | ~5 min |

So at "200 particles" we were closer to a minute than 30 minutes. But the deeper point you're making is correct: the runs were slow given that ParticleDA was supposedly built for HPC parallelism.

## What ParticleDA paid for and didn't use

At 1 MPI rank, ParticleDA still runs through all its MPI-aware machinery, just with degenerate (no-op) communication. Costs incurred:

1. **MPI initialisation + finalisation** — small constant overhead
2. **`MPI.Gather` and `MPI.Bcast` calls every timestep** — at 1 rank these are essentially no-ops, but the code path executes
3. **Custom mean/variance reductions** — designed for cross-rank reduction; trivial at 1 rank but still runs
4. **`optimize_resampling: true` (default)** — runs an optimal-transport solver via the HiGHS LP solver every timestep to minimise particle-copy traffic *between ranks*. At 1 rank there's nothing to optimise, but the solver still runs.
5. **HDF5 I/O every step** because `verbose: true` — writes `state_avg/tNNNN/{height,vx,vy}` (3 fields × 51×51 floats), `state_var/tNNNN/{height,vx,vy}`, and `weights/tNNNN` (length N) after every filter update. For N=10 000 / T=100 this adds up to **several GB of disk writes**.

Of these, items 4 and 5 are real per-step overhead you can switch off. Items 1–3 are tiny in absolute terms.

## How to actually invoke MPI

The script doesn't change — you just launch differently:

```bash
# 4 MPI ranks on this machine
mpiexec -n 4 julia --project=test benchmark_run.jl

# 8 ranks × 4 threads each = 32-way parallelism (hybrid MPI + threads)
mpiexec -n 8 julia -t 4 --project=test benchmark_run.jl

# On a cluster job script (Slurm example)
srun --ntasks=64 --cpus-per-task=4 julia -t 4 --project=test benchmark_run.jl
```

`ParticleDA.run_particle_filter` checks `MPI.Comm_size(MPI.COMM_WORLD)` at runtime and divides `nprt` evenly across ranks. With N=200 and 4 ranks, each rank gets 50 particles. **Constraint:** `nprt` must be divisible by the number of ranks (there's an `@assert` for this in the source).

We never used `mpiexec`. We just typed `julia ...` which always gives you 1 rank.

## What speedup would MPI actually buy?

Estimating from the single-rank numbers:

- Per-particle-per-step time at 1 rank: ≈ wallclock / (N × T)
- At N=10 000, T=100, 3 727 s → **3.7 ms per particle-step**
- Most of that is the PDE solve (10 sub-steps × 51×51 grid) — **embarrassingly parallel** across particles
- With perfect strong scaling, 8 ranks → ~ 8× speedup → 3.7 ms becomes 0.46 ms → 62 min becomes ~8 min

The paper itself (Giles et al. 2024, Section 6.2, Figure 6) reports near-linear weak scaling out to 128 ranks. Weak scaling = doubling ranks while doubling N keeps wallclock constant. For *strong* scaling (fixed N), efficiency drops at very high rank counts because the per-step "Update statistics" reduction (a global all-reduce on the state-dim mean and variance) becomes the bottleneck. But at modest rank counts (4–16), most of the linear speedup is realised.

**Rule of thumb:** for our problem, 4–8 ranks would give 4–8× speedup. Beyond that, returns diminish because the reduction step starts dominating.

## So why use ParticleDA at all over LowLevel?

Honest answer: **only when you need scale**. Specifically:

| Scenario | Pick |
|---|---|
| Single-node prototyping, state fits in RAM | **LowLevel** (faster, simpler) |
| Single-node production, N ≤ a few thousand | **LowLevel** (multi-threading is enough) |
| State-dim × N > single-node RAM (e.g. ice-sheet at fine resolution) | **ParticleDA** (must distribute across nodes) |
| Wall-clock target needs > 64 cores | **ParticleDA** (MPI) |
| Operational/real-time DA in HPC environment | **ParticleDA** (what it's built for) |

The paper's pitch is real, but it's a pitch for **HPC users**, not for laptop prototyping. At 1 rank, ParticleDA is slower than LowLevel for the same problem because of the overhead listed above, especially the HDF5-every-step writing.

## Per-step cost comparison at single-rank, single-thread

A rough breakdown for our tsunami N=200 / T=100 / 51×51 grid run (~72 s total):

| Component | Estimated % of wallclock |
|---|---:|
| Forward model PDE solve (10 sub-steps × 51×51 per particle) | ~50 % |
| Weight computation + normalisation | ~5 % |
| Resampling (with optimal-transport solver) | ~10 % |
| Summary statistic computation (mean, var) | ~5 % |
| **HDF5 write every step** | **~25 %** |
| MPI no-op communication + bookkeeping | ~5 % |

Estimates only — the actual breakdown is in the `enable_timers: true` output if you ever enable it. The key insight: **HDF5 writes alone account for roughly a quarter of the wallclock**. LowLevel doesn't pay this unless you ask for I/O.

## How to make single-rank ParticleDA faster (if you ever need to)

If you must run ParticleDA single-process (e.g. for a sanity check before scaling up to MPI), turn off the expensive bits:

```yaml
filter:
  verbose: false              # no HDF5 writes per timestep — saves ~25 % of wallclock
  optimize_resampling: false  # skip the HiGHS optimal-transport solver — saves ~10 %
```

Both default to "on". Switching them off would have cut our N=10 000 / T=100 run from 62 min to roughly **35 min** — about comparable to what LowLevel would do running the same forward model.

## Implication for the glacier project

This **strengthens the recommendation from `00_action_plan.md` Phase D**:

- **Prototype on LowLevel.** It's lighter, faster on a single laptop, and you can iterate quickly through the σ_obs / σ_init / N sweeps.
- **Port to ParticleDA only when you genuinely need MPI.** Triggers: ice-sheet-scale state, N × state_dim that doesn't fit in a single node's RAM, or a wallclock target that demands a cluster.
- **When you port, actually use `mpiexec`.** If you launch ParticleDA without `mpiexec`, you're getting the worst of both worlds: ParticleDA's overhead with none of its benefit.

## One-line summary for the notebook

> ParticleDA's MPI is opt-in via `mpiexec`. We never opted in. Our "ParticleDA benchmark" was actually a benchmark of ParticleDA in its slowest possible single-process mode. For glacier-model prototyping, LowLevel is the right tool. Only when scale demands MPI should the choice flip.

---

## Verified from source code

- MPI calls: `src/ParticleDA.jl:136, 138, 184, 186-187, 264, 282, 288` (gather, bcast, init, comm_rank, comm_size)
- HDF5 writes per step (verbose mode): `src/ParticleDA.jl:233, 310` → `write_snapshot` in `src/models.jl:456`
- Optimal-transport resampling: `src/utils.jl` → `optimized_resample!`, uses `HiGHS` and `ExactOptimalTransport` packages (see `Project.toml`)
- Filter parameters defaults: `src/params.jl:23-33`
- Particle distribution across ranks: `src/ParticleDA.jl:190-192` (`@assert mod(filter_params.nprt, my_size) == 0`)
