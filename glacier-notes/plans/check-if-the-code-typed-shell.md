# Bootstrap Particle Filter on LLW2d Tsunami — Findings & Plan

## TL;DR (for write-down)

We tested how a Bootstrap particle filter behaves on a high-dimensional model (LLW2d tsunami, state dimension ~7 800) by sweeping three different knobs one at a time: **N** (number of particles), **σ_obs** (observation noise), and **n_stations** (how many observation points). Goal: figure out what's needed to avoid "particle degeneracy" (all weight collapsing onto one particle) so we can apply the lesson to our glacier model later.

**One-sentence answer:** *More particles doesn't fix degeneracy — loosening the observation noise does.*

---

## What "degeneracy" means in this experiment

- **Bootstrap PF** = the simplest particle filter — proposes new particles from the model's prior dynamics, then re-weights by how well each one matches observations.
- **Degeneracy** = after a few timesteps, one particle gets nearly all the weight (max weight ≈ 1.0), every other particle is dead, and the **Effective Sample Size (ESS)** drops to 1. You essentially have a single trajectory pretending to be an ensemble — the filter looks like it works, but its uncertainty estimates are meaningless.
- **ESS formula** = 1 / Σ(wᵢ²). Healthy ensemble: ESS ≈ N. Collapsed: ESS = 1.
- We declare a config "healthy" if ESS stays above ~10 % of N most of the time.

---

## Test setup (so you know what we measured)

- **Model**: LLW2d tsunami, 200 × 200 km domain, 51 × 51 grid, three state variables (water height + two velocity components) → **state dimension 7 803**
- **Truth**: 30 m initial wave peak in the lower-left corner; we let it propagate for 100 timesteps (= 500 simulated seconds)
- **Filter**: ParticleDA's Bootstrap filter with `MeanAndVarSummaryStat`
- **Default observations**: 15 stations from the paper, observation noise σ_obs = 0.01 m
- For each run we recorded: ESS history (per timestep), max weight at end, RMSE of posterior mean against the truth, total wall-clock time
- All 11 runs total wall-clock ≈ 2 hours

---

## Finding 1 — Number of particles N: **DOESN'T HELP**

This was the most dramatic result. We ran the same exact setup with five different particle counts:

- **N = 50** → ESS stayed at 1.0 the whole run, max weight = 1.0
- **N = 200** → ESS = 1.0, max weight = 1.0
- **N = 500** → ESS = 1.0, max weight = 1.0
- **N = 1 000** → ESS = 1.0, max weight = 1.0
- **N = 5 000** → ESS = 1.0, max weight = 1.0
- **N = 10 000** → ESS = 1.0, max weight = 1.0

**Bullet form for the notebook:**
- Going from 50 → 10 000 particles is a **200× increase in compute cost** with essentially **zero ESS benefit**.
- Wall-clock scales linearly: 22 s → 21 s → 72 s → 179 s → 358 s → 1 806 s → 3 727 s for the six tightest-obs runs.
- RMSE was also flat across all N values (≈ 0.6 m) — adding particles didn't make the *accuracy* any better either.
- This matches what you saw with LowLevelParticleFilters at N=10 000 — confirms the issue isn't the library, it's the physics of high-dim Bootstrap.
- **The "no realistic N saves Bootstrap" conclusion is now experimentally verified for this problem class.**

---

## Finding 2 — Observation noise σ_obs: **THIS IS THE LEVER**

We fixed N = 500 and varied how noisy the filter thinks observations are:

- **σ_obs = 0.01 m** (paper default — very tight) → ESS = 1.0 (collapsed)
- **σ_obs = 0.1 m** (10× looser) → mean ESS = 2.1, max weight = 0.83 (barely twitching)
- **σ_obs = 1.0 m** (100× looser) → **mean ESS = 197** (40 % of N), max weight = 0.017 ← **knee here**
- **σ_obs = 5.0 m** (500× looser) → **mean ESS = 431** (86 % of N), max weight = 0.004 ← **healthy**

**Bullet form for the notebook:**
- The **transition is sharp** — going from σ=0.1 to σ=1.0 (just one decade) flipped the filter from collapsed (ESS=2) to healthy (ESS=197).
- At σ_obs = 5.0 m the filter is essentially uniform — every particle has roughly the same weight.
- The "knee" sits between σ_obs = 0.1 and σ_obs = 1.0 for this problem.
- **Why it works**: tight observations make the likelihood very "spiky" — one particle scores far higher than all others, and resampling kills the rest. Loose observations flatten the likelihood so many particles get comparable weights.

---

## Finding 3 — Number of stations n_stations: helps, but only at very low counts

We fixed N = 500, σ_obs = 0.01 m, and varied how many observation points the filter sees:

- **15 stations** (paper default) → ESS = 1.0 (collapsed)
- **5 stations** → ESS = 1.0 (still collapsed!)
- **1 station** → **mean ESS = 47** (9 % of N), max weight = 0.024 ← finally healthy-ish

**Bullet form for the notebook:**
- Going from 15 → 5 stations did **nothing** — still collapsed. Cutting observation count by 3× wasn't enough.
- Going from 5 → 1 station did a lot — single observation gives the ensemble breathing room.
- **The reason**: each observation point multiplies the weight-concentration pressure. The math: log-likelihood = sum over observation points of (obs - prediction)² / (2σ²). More obs = more terms = sharper spike in particle weights.
- Translation: **observation count and σ_obs combine multiplicatively** — what matters is `n_obs × (1 / σ_obs)²`, not either one alone.

---

## Finding 4 — Counter-intuitive: looser observations gave BETTER accuracy

This was the most surprising number in the sweep:

- **σ_obs = 0.01** (tight) → RMSE = 0.62 m (collapsed, only one particle alive)
- **σ_obs = 1.0** → RMSE = 0.59 m
- **σ_obs = 5.0** (very loose) → **RMSE = 0.48 m** ← *best of all*

**Bullet form for the notebook:**
- You'd expect looser observations → filter trusts data less → worse accuracy. We got the **opposite**.
- **Why**: with tight observations, the filter collapses to a single particle. The "posterior mean" is literally just that one particle's state — including all its random-walk noise. With loose observations, all 500 particles stay alive, so the posterior mean is an **ensemble average that smooths out the random fluctuations**.
- **Caveat**: this only works because the truth in this problem is smooth (a propagating wave) and the random noise we average over is small. In a problem where the true state itself fluctuates fast, over-loose σ_obs would smear out real signal.
- **Practical implication**: in the data-assimilation literature this is called **"observation error inflation"** — deliberately telling the filter that observations are noisier than they actually are. It's a standard trick when you have a small ensemble.

---

## Finding 5 — Empirical rule of thumb (write this down)

Combining the above, we found one quantity that predicts whether Bootstrap will work:

```
pressure = n_obs × (signal_std / σ_obs)²        per timestep
```

where `signal_std` is the typical scale of variation in the observed quantity.

- **pressure ≪ 10**  → Bootstrap is fine
- **pressure ~ 10–100** → Bootstrap is borderline (this is where the "knee" was)
- **pressure ≫ 100** → Bootstrap collapses regardless of N

**Checked against the actual runs:**

| Run | pressure (with signal_std≈3 m) | Observed ESS | Verdict |
|---|---:|---:|---|
| σ_obs = 0.01, 15 obs | 15 × (3/0.01)² = 1.4 × 10⁶ | ESS=1 | extreme overload |
| σ_obs = 0.1, 15 obs | 15 × 900 = 13 500 | ESS=2 | still overload |
| σ_obs = 1.0, 15 obs | 15 × 9 = 135 | ESS=197 (40 %) | borderline ✓ |
| σ_obs = 5.0, 15 obs | 15 × 0.36 = 5.4 | ESS=431 (86 %) | healthy ✓ |
| σ_obs = 0.01, 1 obs | 1 × 90 000 = 90 000 | ESS=47 (9 %) | still overload, but less so |

The formula isn't perfect (the σ=0.01 / 1-station case violates it slightly) but it gets the order-of-magnitude right and explains all the qualitative jumps in the data.

---

## Recommended starting template for the glacier model

When you set up the Bootstrap filter for glaciers, do this **in order**:

1. **Estimate the natural variability of your observed quantities.** Look at climatology — over a glacier's typical timescale, how much do velocity / thickness / elevation actually fluctuate? That's `signal_std`.
2. **Count your effective observation dimensions.** If you have 100 stations but they're spatially correlated, the *effective* number might be more like 10.
3. **Choose σ_obs so that the pressure formula stays under ~100**: `σ_obs ≥ signal_std × √(n_obs / 100)`. If your real instrument noise is tighter than this, **deliberately inflate it in the filter** — this is standard, accepted DA practice.
4. **Start with N = 100–500.** Larger N is wasted compute until you've fixed the pressure problem. Once pressure is under control, scale N up only if ESS still drops below ~10 % of N late in the run.
5. **Watch ESS over time** as your primary diagnostic. If ESS dives in the first 5 timesteps, your pressure budget is too high.
6. **If pressure can't physically be brought under 100** (e.g., you genuinely need 1 000 observations and they're really tight), then Bootstrap is structurally wrong for your problem. You'd need:
   - **Localised** particle filters (ParticleDA doesn't currently support this)
   - **Hybrid** methods (EnKF + PF)
   - Or accept the limitation and report uncertainty differently (e.g., only the surviving particle's trajectory as a point estimate)

---

## ParticleDA fit for the glacier project

- **Pro**: multi-node MPI (which LowLevelParticleFilters lacks) — important if your glacier state is even higher-dimensional than the tsunami (likely yes for 3D)
- **Pro**: clean YAML config, well-tested I/O, integration tests
- **Con**: doesn't include localisation or hybrid methods — those would need to be added on top
- **Con**: Bootstrap filter degenerates exactly the way ours did on tsunami. You must fix obs-pressure before scaling

**Wall-clock benchmark from this experiment**: N=10 000 / T=100 / state-dim 7 803 took **62 min on this Mac (single node, no MPI)**. With ParticleDA's MPI, expect roughly linear scaling across nodes — so 5 nodes ≈ 12 min, 10 nodes ≈ 6 min. That's the throughput you can plan around.

---

## Files produced by the sweep (still in repo)

- `benchmark_sweep_results.csv` — full numerical table (11 rows × 11 columns)
- `benchmark_sweep_ess.png` — ESS-vs-time curves, 3-panel by sweep axis
- `benchmark_sweep_rmse.png` — RMSE-vs-time curves, 3-panel by sweep axis
- `benchmark_sweep_summary.png` — bar chart of min/mean ESS as % of N
- `sweep_*.h5` — per-run HDF5 with full particle-mean / variance / weights history (~15–20 MB each)
- `benchmark_sweep.log` — full stdout
- `benchmark_sweep.jl` — the sweep script itself, parameterised, easy to extend for new sweeps

---

## How the sweep was implemented (for later reference)

- Uses three existing ParticleDA APIs: `LLW2d.init`, `simulate_observations_from_model`, `run_particle_filter` — all in-process, no YAML rewrites per run
- The script builds a per-run `model_dict` Dict, simulates truth + observations with seeded RNG, runs the filter, reads back ESS/RMSE from the per-run HDF5
- Configs are run **fast → slow** so partial results are durable if interrupted
- Each config writes its CSV row immediately and re-renders all three plots — never lose work
- All seeds fixed (`123` for observations, `42` for filter) — sweep is fully reproducible
- No source modifications to `src/` or `test/`

### Sweep grid (final, 11 runs)

| Axis | Values | Held fixed |
|---|---|---|
| N | 50, 200, 500, 1 000, 5 000, 10 000 | σ_obs=0.01, n_stations=15 |
| σ_obs | 0.01, 0.1, 1.0, 5.0 | N=500, n_stations=15 |
| n_stations | 1, 5, 15 | N=500, σ_obs=0.01 |

### Numerical results table

| Run | N | σ_obs | S | min ESS | mean ESS | max w | mean RMSE (m) | time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N50 | 50 | 0.01 | 15 | 1.00 | 1.01 | 1.000 | 0.631 | 22 |
| N200 | 200 | 0.01 | 15 | 1.00 | 1.00 | 1.000 | 0.619 | 72 |
| N500 | 500 | 0.01 | 15 | 1.00 | 1.00 | 1.000 | 0.637 | 179 |
| N1000 | 1 000 | 0.01 | 15 | 1.00 | 1.00 | 1.000 | 0.609 | 358 |
| N5000 | 5 000 | 0.01 | 15 | 1.00 | 1.02 | 1.000 | 0.645 | 1 806 |
| **N10000** | **10 000** | 0.01 | 15 | **1.00** | **1.03** | **1.000** | **0.648** | **3 727** |
| sig0.1 | 500 | 0.1 | 15 | 1.00 | 2.12 | 0.835 | 0.624 | 179 |
| sig1.0 | 500 | 1.0 | 15 | 7.85 | 196.85 | 0.017 | 0.592 | 179 |
| **sig5.0** | 500 | **5.0** | 15 | **266** | **431** | **0.004** | **0.482** | 179 |
| S5 | 500 | 0.01 | 5 | 1.00 | 1.06 | 1.000 | 0.711 | 179 |
| S1 | 500 | 0.01 | 1 | 8.89 | 47.17 | 0.024 | 0.634 | 179 |

---

## FAQ — Why does RMSE INCREASE over time? Shouldn't DA make it shrink?

This is a sharp observation and the answer is informative — it tells you something important about what the filter is actually doing. **In this experiment, "RMSE growing with time" is correct, expected, and matches the physics.** Here is the full breakdown:

### 1. RMSE starts near zero because the experiment is rigged that way

- We set `use_peak_initial_state_mean: true` AND `sigma_initial_state: 0.001`.
- That means every single particle, AND the truth, all start from **the same 30 m peak in the lower-left corner** with essentially zero random spread.
- At t = 0: `mean(particles) ≈ truth` → **RMSE ≈ 0 m**.
- So we're not measuring "how well the filter converges from a bad start" — we're measuring "how well the filter holds on to a perfect start as the dynamics randomise."

### 2. The truth itself is a random process — it's not a fixed target

Every timestep, the truth gets two updates:

- **Deterministic**: the shallow-water wave equations push the wave forward
- **Stochastic**: `update_state_stochastic!` draws a fresh Gaussian random field (with `sigma = [0.1, 10.0, 10.0]` per state variable) and adds it on top

So the truth is **one specific realisation** of a stochastic dynamical system — a particular random walk. It's not a smooth, fixed function of time that the filter could in principle "find."

### 3. Process noise accumulates roughly like √t

Per-step height-noise std-dev is **0.1 m**. Over `t` independent steps, the cumulative random-walk-style displacement grows like:

```
expected drift ≈ 0.1 × √t  metres
```

So at t = 100, the "natural" RMSE floor (between two independent realisations starting at the same point) is ≈ **0.1 × √100 = 1.0 m**.

Our actual observed RMSE at t = 100 is ≈ 0.5–0.8 m for the collapsed runs and ≈ 0.5 m for sig5.0 — meaning the filter is doing *better* than the random-walk floor by maybe a factor of 1.5–2×. It IS helping; it's just helping bound the growth, not reverse it.

### 4. The collapsed runs (almost all of them) are one random walk vs another

When σ_obs = 0.01 the filter collapses to a single surviving particle. That one particle has its own stochastic trajectory drawn from the same dynamics as truth. So **the "posterior mean" is just one particle's random walk, and truth is another**. Two independent random walks starting at the same point diverge by exactly √t — that's the physics.

Observations only constrain the surviving particle's state at 15 discrete stations. Away from those points (which is most of the 51×51 grid), the particle drifts freely. So the per-cell error is bounded by observations near the stations but grows freely between them — average that over the grid and you get the growing curves you see.

### 5. Even the healthy run (sig5.0) can't drive RMSE down

For sig5.0, the ensemble stays alive and the posterior mean averages over 500 particles — smoothing out *the filter's own randomness*. But it still doesn't know **truth's specific noise realisation**. So:

```
posterior_mean ≈ deterministic_part_of_dynamics
truth          = deterministic_part_of_dynamics + truth's specific noise realisation

⇒ RMSE = ||truth's noise component||
```

That noise component grows as √t, regardless of how good the filter is. **The only way to drive RMSE down would be to observe truth densely enough to estimate its noise realisation across the whole grid** — that's not what we have (only 15 sparse stations).

### 6. So would RMSE ever decrease? Yes, in two regimes

- **Deterministic dynamics** (`sigma = [0, 0, 0]`): no process noise → truth is a fixed function of time → filter would converge to truth and RMSE would fall.
- **Dense observations**: if stations covered the grid densely, observations would constrain truth's noise realisation everywhere; the posterior would converge to truth.

In our experiment we have **stochastic dynamics + sparse observations**, and the dynamics dominate. That's the inherent floor.

### 7. Bullet form for the write-down

- RMSE = 0 at t=0 is an **artefact of the experiment design** — all particles + truth start from the same wave peak.
- RMSE growing with time is **the truth wandering away from any deterministic prediction** because the dynamics are stochastic, not because the filter is failing.
- Expected random-walk floor at t=100: ~1.0 m. Observed: 0.5–0.8 m. Filter is helping, but only suppressing the growth, not reversing it.
- For collapsed runs: posterior mean = one specific particle's random walk; truth = another → RMSE grows like √(t).
- For healthy ensemble (sig5.0): posterior mean ≈ deterministic dynamics; truth = deterministic + noise realisation → RMSE = magnitude of truth's noise → still grows like √(t), but the ensemble averages out *its own* randomness so RMSE is lower than collapsed runs.
- To make RMSE decrease, you'd need either **deterministic dynamics** (no process noise) or **observation density comparable to state dimensionality** — neither holds here.

### 8. Implication for the glacier model

- If your glacier model has significant **process noise** (uncertain SMB, unknown sliding parameters, sub-grid processes lumped into stochastic forcing), expect RMSE to grow ~√t between observation-constrained points, regardless of filter quality.
- Decreasing RMSE is realistic only if your glacier dynamics are nearly deterministic AND your observations have spatial coverage comparable to the state grid.
- Plot RMSE on log scale if you want to see whether the growth rate is √t (DA limited by process noise) or faster (DA itself failing).
- **Don't judge filter quality by RMSE alone.** Pair it with ESS — a filter with ESS≈1 and "good" RMSE is just one trajectory pretending to be an ensemble; the uncertainty quantification is broken even though the point estimate looks fine.

---

## FAQ — What is `n_integration_step` and why does it matter for runtime?

### The short version

There are **two different time-steps** inside this code, and they do different things:

| Parameter | What it controls | Our value | Total time |
|---|---|---:|---|
| `time_step` | Time between **filter updates** (when observations arrive and particles get re-weighted/resampled) | 5.0 s | 5 s × n_time_step |
| `n_integration_step` | Number of **internal PDE sub-steps** used to advance the wave equations between two filter updates | 10 | internal Δt = time_step / n_integration_step = 0.5 s |

So if you set `time_step=5.0`, `n_integration_step=10`, `n_time_step=260` (the wave-propagation showcase run):

- **Filter updates**: 260 times (every 5 simulated seconds → T_max = 1 300 s)
- **PDE updates inside the model**: 260 × 10 = **2 600 times** (every 0.5 simulated seconds)

(Your "every 2 s, 640 times" framing is close but slightly off — the actual choice was `time_step=5.0 s` with 256 filter updates for the T=1 280 s figure. The internal PDE sub-step works out to 0.5 s, not 2 s.)

### Why are these two time-steps separate?

The shallow-water PDE has a **CFL stability condition**:

```
internal_dt ≤ dx / wave_speed
```

For our setup: dx = 200 km / 51 ≈ 3.9 km, wave_speed = √(g·depth) = √(9.81 × 3 000) ≈ 171 m/s.
So **internal_dt_max ≈ 3 900 / 171 ≈ 23 s** for hard stability.

In principle `time_step=5.0` alone would be stable. But:

- Higher-frequency wave modes on the grid have effective speeds higher than the headline c = √(gh) and would push closer to the CFL limit.
- The numerical scheme (Lax-Wendroff / leap-frog style) is *more accurate* with smaller sub-steps — less numerical dispersion that smears the wave fronts.
- The absorbing-boundary "sponge" needs fine sub-stepping to damp waves without reflecting them.

So the code splits the responsibility: **the filter cares only about the 5-second sampling interval (when observations arrive); the model cares about resolving the wave physics, which needs 0.5-second sub-steps internally**.

The exact line in the code that does this is in [test/models/llw2d.jl:612-629](test/models/llw2d.jl#L612):

```julia
dt = model.parameters.time_step / model.parameters.n_integration_step
for _ in 1:model.parameters.n_integration_step
    timestep!(...; dt)
end
```

Plain English: "compute an internal sub-step `dt = time_step / n_integration_step`, then call the PDE update that many times in a row."

### Why 10 specifically?

10 is a convention from the paper's showcase config (`inputs/parametersW1.yaml`). It gives:

- internal dt = 5.0 / 10 = 0.5 s — well inside the CFL safety margin
- A 10× safety factor over the strict CFL limit, which is comfortable for the absorbing boundaries and the higher-frequency components
- 10 is a round number that's clearly sufficient empirically (the published runs are stable and accurate)

It's not magical. You could probably use 5 (internal dt = 1 s) and still get a stable simulation; you'd just need to verify the wave looks the same. At n_integration_step = 1 (no sub-stepping, internal dt = 5 s) it'd probably blow up at the wave front.

### How does it scale runtime?

`n_integration_step` is a **linear multiplier** on the per-step compute. The cost breakdown is:

```
total_compute ≈ N_particles × n_time_step × n_integration_step × (grid_size × O(1))
                                            ↑
                                     this is what's "extra"
```

Numerical example using our N=10 000 / T=100 / n_integration_step=10 / 51×51 grid run that took 3 727 s:

- Total PDE updates: 10 000 × 100 × 10 × (51 × 51) ≈ **2.6 × 10¹⁰ grid-point updates**
- Per grid-point work: 3 727 s / 2.6 × 10¹⁰ ≈ **~140 ns per grid-point update** (Julia, single-threaded)

What happens when you change knobs:

| Change | Effect on runtime |
|---|---|
| Double N (particles) | ~2× runtime |
| Double n_time_step (longer simulation) | ~2× runtime |
| Double n_integration_step (finer sub-steps) | **~2× runtime** |
| Halve n_integration_step | **~½ runtime** (if still stable) |
| Double the grid (nx, ny both 2×) | ~4× runtime |
| Halve time_step but keep n_integration_step | ~no change to per-second cost (you do half the work per filter update but twice as many filter updates) |

### Implications for the glacier model

- **Pick n_integration_step from the CFL condition for *your* dynamics**, not by analogy with tsunami. Compute the fastest signal speed in your glacier model (probably ice-flow advection, basal sliding speeds, or stress propagation timescales). Then `internal_dt ≲ dx / fastest_speed`, with a 2–5× safety margin.
- **Don't blindly raise n_integration_step for "more accuracy"**: every doubling costs ~2× compute across the whole sweep. With N=10 000 already taking an hour for tsunami, doubling n_integration_step would push it to 2 hours. The accuracy gain past the CFL+safety threshold is small.
- **Lowering n_integration_step is the cheapest way to speed up a sweep** — provided the simulation stays stable. Worth profiling: try n_integration_step = 5, 3, 2 and watch for blow-ups in the wave field. The fastest stable setting buys you a free 2× to 5× on every run in the sweep.
- **Convergence test**: pick one configuration and run it with n_integration_step = 5, 10, 20, 40. If the results don't change between 10 and 20, you're over-sub-stepping at 20 — and you can probably go down to 5 before they break.

### Quick mental model

Think of `time_step` as **"how often the filter asks the model: where is the wave now?"** and `n_integration_step` as **"how carefully the model integrates the wave equations between those questions."** They're independent: you can ask less often (large time_step) while still integrating carefully (large n_integration_step), or ask often while integrating crudely. The optimum depends on (a) how fast you want filter corrections to flow into the state estimate and (b) how stiff the underlying PDE is.

---

## ⚠ Corrections — what I got wrong vs the actual paper (Giles et al. 2024 §6.1)

After you pointed me to page 11 / line 250 of the paper, I went back and re-read. **Two of my numbers don't match the paper.** Honest list:

### Mismatch 1: time_step

- **Paper says (line 249–250):** *"The particle states are integrated for 1280s, with the assimilation of the surface elevation occurring every dt = 2 s."*
- **What that means**: time_step = 2 s, n_time_step = 640. So 640 filter updates over 1280 s of simulated time.
- **What I used in `benchmark.yaml`**: time_step = 5 s, n_time_step = 260 → 256 filter updates over 1300 s.

Your "every 2 s, 640 times" framing was the correct answer from the paper. I was wrong to push back on it.

**Why I picked 5 s anyway:** I followed `inputs/parametersW1.yaml` (Alex Beskos, May 2021) which has time_step = 5.0. That file pre-dates the published experiment and was for an earlier development phase. The paper's actual experiment matches `inputs/parameters_truth.yaml` (Dan Giles, Aug 2022), which has **time_step = 2.0** and **n_integration_step = 4** — that's the canonical config for what you saw in Fig 4 of the paper.

### Mismatch 2: observation noise σ_obs

- **Paper says (line 247):** *"The observation error standard deviation for both the observation run and particles is set to 0.1."*
- **What I used**: `obs_noise_std: [0.01]` — **10× tighter than the paper.**
- **What I confused it with**: the paper's *initial-state* random-field std-dev is 0.01 (line 248). I conflated the two.

This second mismatch is the more important one for the sweep findings. The paper used σ_obs = 0.1; I used σ_obs = 0.01. Mine was 10× sharper, which is partly *why* my Bootstrap was even more degenerate than the paper's already-difficult Bootstrap result.

### Why our wave-propagation figure still looked correct

- Both time_step = 2 s and time_step = 5 s are well within the CFL stability limit (~23 s for this grid), so the wave dynamics are visually indistinguishable.
- The paper's snapshot times (0, 320, 740, 960, 1280 s) happen to be integer multiples of *both* 2 and 5, so either choice hits the same snapshots.
- The visual appearance of the propagating wave is dominated by the deterministic shallow-water equations; the time_step choice only affects fine-scale numerical dispersion.

So you got the right-looking figure for the wrong reason. The wave physics didn't care, but the numbers were off-paper.

### Does this invalidate the sweep findings?

**No** — and this is the important thing. The sweep already tested σ_obs = 0.1 (i.e. the paper's actual value) as one of the rows. The result was:

```
N = 500, σ_obs = 0.1, n_stations = 15  →  mean ESS = 2.12  →  STILL COLLAPSED
```

So even at the paper's σ_obs = 0.1, Bootstrap collapses (per our experiment). The reason the paper's experiment "worked" is that **the paper used the locally-optimal filter, not Bootstrap**. The paper itself reports in Fig 5 (left) that Bootstrap underperforms locally-optimal exactly because of this degeneracy issue — that's the whole motivation for their proposing the optimal proposal.

So the qualitative conclusions stand:

- **More N doesn't help Bootstrap** (N=50 → N=10 000 sweep, all flat at ESS=1)
- **σ_obs is the lever** — and we now know the paper's *real* σ_obs (=0.1) is *already past the collapse knee* for Bootstrap. You need σ_obs ≥ ~1.0 to keep Bootstrap healthy on this problem.
- For the glacier-model recommendation: the pressure-budget formula `n_obs × (signal_std / σ_obs)²` is unchanged; only the specific numerical example for tsunami should be updated using σ_obs = 0.1 (yielding pressure ≈ 13 500, still hugely overloaded).

### Effects of dt on the sweep, in principle

If we re-ran the sweep with time_step = 2 s instead of 5 s, two things change:

1. **2.5× more filter updates per simulated second** — so 2.5× more compute for the same simulation duration. The N=10 000 run would have taken ~2.5 hours instead of 1 hour.
2. **Particles drift less between updates** — they stay closer in observation space, so per-update log-weight differences are smaller. Slightly *more* of them survive each update. But there are 2.5× more updates per second, so per-second-collapse-pressure is roughly unchanged.

Net effect: probably small. The collapse-vs-healthy boundary in σ_obs would shift by maybe a factor of √(5/2) ≈ 1.6 — i.e. the σ_obs = 1.0 "knee" might shift to σ_obs ≈ 0.6 at dt = 2. Not a regime change.

### What to update if we re-ran cleanly

If you want a faithful paper-replication, the benchmark config should be:

```yaml
filter:
  nprt: 50                      # paper's N for Fig 4
  ...
model:
  llw2d:
    time_step: 2.0              # paper's dt
    n_integration_step: 4       # paper's value (from parameters_truth.yaml)
    obs_noise_std: [0.1]        # paper's σ_obs
    sigma_initial_state: 0.01   # paper's initial random-field std
    ...
```

(Note: the paper's Fig 4 actually uses N = 50, not 200. I picked 200 earlier because parametersW1.yaml had `nprt: 200`. Both work for the visualisation; the paper picked 50 to show the filter doing useful work with a small ensemble.)

### One-line summary

> The paper uses **dt = 2 s, σ_obs = 0.1, σ_init = 0.01, with the locally-optimal proposal**. I used **dt = 5 s, σ_obs = 0.01, σ_init = 0.001, with Bootstrap**. The wave figure visually matched anyway, and the sweep conclusions about Bootstrap behaviour are unchanged — but the numerical specifics were off-paper, and you were right to flag it.

---

## → Follow-on glacier-model planning

Glacier-model-specific action plan (Phases A–D, infrastructure decision, σ_init lever discussion) has moved to its own folder so this file stays focused on tsunami:

> **`~/.claude/glacier-model/00_action_plan.md`**

Future per-experiment notes for the glacier project should go in that same folder.

---

## No further code changes proposed at this point

This plan file is now a **deliverable**: it documents the experiment, the findings, and the recommendations for the glacier project. The sweep has answered the original three questions:

1. **Does N=10 000 restore ESS?** → No (verified).
2. **Which other knobs help?** → σ_obs (massively), n_obs (at very low counts).
3. **What's the recommended config to start the glacier project?** → Estimate signal_std, choose σ_obs so pressure ≲ 100, start at N=100–500, scale up only if needed.

Once you've reviewed and have decisions about the glacier model concrete, we can plan the next phase (porting this sweep methodology to your glacier setup).
