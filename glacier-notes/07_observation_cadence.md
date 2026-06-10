# Observation cadence — how often should the filter look at the data?

> **What this note covers.** We changed how often the particle filter "looks at" the observations, while keeping the underlying physics exactly the same. Result: fewer-but-spaced-out observations gave a slightly *healthier* filter than frequent observations. This note explains why, with all the technical words defined in plain English.
>
> **Quick glossary at the end** if you ever need to re-look up a term.
>
> **Files:** [run_obs_cadence.jl](../glacier-code/particleda/run_obs_cadence.jl), [plot_obs_cadence.jl](../glacier-code/particleda/plot_obs_cadence.jl), results in [results/obs_cadence/](../glacier-code/particleda/results/obs_cadence/).

---

## 1. The motivation in everyday terms

A particle filter does two things in a loop:

1. **Predict** — push each candidate guess (each "particle") forward in time using the physics model.
2. **Update** — compare each candidate's predicted observation against the real observation, then keep the candidates that agreed best.

So far we've been doing **predict → update → predict → update → …** with one prediction step followed by one update step. The update happens *every* model second.

But real glacier observations don't come every second. Satellite passes are days apart; GPS readings might be hourly. The natural pattern is **predict many times → update once → predict many times → update once → …**.

The user asked: what happens if we make the filter only update every 2 (or n) model time units, while the physics still steps forward continuously? Does the filter still work? Does ESS — the "ensemble health" number — get better or worse?

That's the experiment in this note.

---

## 2. The hidden bug we found while setting this up

While planning the experiment I noticed something wrong in the model code. The YAML config had two parameters:

- `time_step` — supposed to mean "how many seconds does one filter step cover?"
- `n_integration_step` — supposed to mean "split that filter step into this many smaller physics sub-steps."

But the actual code in `update_state_deterministic!` was completely **ignoring** `time_step` and computing its own internal dt from a stability formula (`dt = 0.2 × dx / max_speed`). So writing `time_step: 1.0` in the YAML did *nothing*.

That gives an internal dt of about **457 seconds** for our toy parameters (grid spacing 4 km, advection speed ≈ 1.75). Every previous run (run-01, run-02, run-03, seed sweep) was actually advancing the model by ~457 seconds per filter step, regardless of what `time_step` said.

This had two consequences:

1. **Run-03 looked healthy partly because the truth was *moving*.** With dt=457s and velocity ≈1.5, β advects ~700 m per filter step — non-trivial motion across a 4 km grid cell. The filter could meaningfully track this motion.
2. **My first attempt at this experiment failed instantly.** I set `time_step=1.0, n_int=1` thinking that would give me "filter every 1 second." With the bug fixed, internal dt became 1.0 instead of 457. Now β advects only 1.5 metres per step — basically nothing. The truth was effectively frozen. The filter latched onto one lucky particle on the very first observation (max weight = 0.9998) and never recovered, leaving RMSE stuck around 460.

So the real lesson is: **internal dt has to be big enough that the physics actually moves between observations.** I picked **dt = 400 s** going forward — just under the stability ceiling of 457 s, with a small safety margin. The fixed code now uses `dt = time_step / n_integration_step` and prints a warning the first time it sees dt above the stability limit.

This bug fix is committed in [glacier_model.jl](../glacier-code/particleda/glacier_model.jl) — search for `_CFL_WARNED`.

---

## 3. The actual experiment

We compared two configurations, with everything else identical (1000 particles, 16 sensors, wide prior σ_init=0.30, σ_obs=0.10, σ_process=0.007, seed=42):

| Tag | `time_step` | `n_integration_step` | Internal dt | Filter steps | Observations |
|---|---:|---:|---:|---:|---:|
| **run-04 every-1c** | 400 s | 1 | 400 s | 200 | **200** |
| **run-05 every-2c** | 800 s | 2 | 400 s | 100 | **100** |

Both runs cover the *same total model time* (200 × 400 = 100 × 800 = 80 000 s). Both use the *same internal physics step* (dt = 400 s). The only difference is how often the filter pauses to look at an observation.

The label "1c" is just shorthand for "every 1 cadence unit" where 1 cadence ≈ 400 s. The user's original phrasing was "every 2 seconds" but in our toy each "second" is meaningless — what matters is the **ratio** of dt to physical change. So "every 2 cadence" = filter sees half as many observations over the same physical interval.

---

## 4. What we found

### 4.1 Headline numbers

| Metric | run-04 (every 1c) | run-05 (every 2c) | Better |
|---|---:|---:|---|
| Mean ESS | 537 / 1000 (54 %) | **613 / 1000 (61 %)** | run-05 |
| Fraction of steps with ESS > 0.5·Np | 0.63 | **0.82** | run-05 |
| Minimum ESS | 1.9 | 3.0 | both have crisis dips |
| Final RMSE (avg of last 10 steps) | 123 | **114** | run-05 (barely) |

### 4.2 What the plots show

- **[ess_cadence.png](../glacier-code/particleda/results/obs_cadence/ess_cadence.png)** — run-05 (orange) sits noticeably *higher* than run-04 (blue) for most of the run. Both clear the 0.5·Np reference line most of the time, but run-05 clears it more often.
- **[rmse_cadence.png](../glacier-code/particleda/results/obs_cadence/rmse_cadence.png)** — both curves drop from ~330 (the prior misfit) to ~115 over the run. Run-05 is slightly faster to converge, and ends lower. The gap is small (~8 % of final value, near the seed-noise floor we measured earlier).
- **[maxweight_cadence.png](../glacier-code/particleda/results/obs_cadence/maxweight_cadence.png)** — both runs have an early spike to ~0.6 (the wide-prior moment where one particle wins big), then collapse to below 0.05 within a few steps. No late-run collapse.
- **[variance_cadence.png](../glacier-code/particleda/results/obs_cadence/variance_cadence.png)** — posterior Var(β) settles to ~1.0 (in β-squared units) in both cases. Run-04 has slightly higher post-spike variance because it's resampling more often, but both flatten out.

### 4.3 The CFL warning

When run-04 started, the model printed `┌ Warning: CFL violation likely`. That happened because at the first step β briefly went a bit higher than expected, pushing max_speed up and dropping the stability ceiling below 400. After one step, the values settled and the warning didn't repeat (the warning is one-shot by design — see `_CFL_WARNED[]` in the code).

In practice nothing exploded; the dynamics stayed stable. But the warning is real signal that we're operating close to the limit. If we ever bump σ_proc up so β can take larger excursions, we should also drop `time_step` down or raise `n_integration_step`.

---

## 5. Why fewer observations gave *better* ESS

This is the counterintuitive bit. The instinct is "more data = better filter." Why does run-05 (half the obs) beat run-04?

Picture the particle cloud (the 1000 candidate β-fields) as a swarm.

**With frequent observations (run-04):**
- After each obs, the filter does *resampling* — it discards the worst-fitting particles and duplicates the best-fitting ones. Diversity drops.
- Process noise then re-injects a tiny amount of random spread (σ_θ = 0.007 in log space).
- Next observation arrives quickly; the swarm hasn't had time to re-spread; weights concentrate again on one or two members; ESS drops.
- Net effect: a tight, fast-converging swarm that nonetheless suffers small-ESS dips between resamplings.

**With less frequent observations (run-05):**
- After each obs and resample, the swarm has **two** sub-steps' worth of process noise to spread out before the next observation hits.
- That extra spread means more candidates are within likely-distance of the truth when the next observation arrives.
- Weights are more balanced across particles → fewer "one particle wins" moments → higher ESS.

Concretely: process-noise spread grows like √(number of sub-steps). Two sub-steps gives √2 ≈ 1.4× more spread than one. That's enough to noticeably change how skewed the weight distribution becomes at the next observation.

There's a well-known trade-off in PF literature: **observation frequency vs ensemble health.** Frequent observations give the best mean tracking accuracy *if* the ensemble survives. Spaced observations give a healthier ensemble at the cost of brief tracking drift between observations. With wide-prior, low-pressure-budget setups like ours, spaced observations win on both metrics in this regime.

This is not always true. In a low-process-noise, high-information-per-observation regime, frequent observations would still win. But ours is the right regime to space them.

---

## 6. What this implies for the glacier project

1. **Real-data glacier observations are spaced out anyway.** Satellite passes (Sentinel-1, MEaSUREs ITS_LIVE) are days to weeks apart. GPS is hourly to daily. We were never going to do "filter every model step" on real data. So this experiment is *reassuring*: the realistic-cadence regime is the comfortable regime.

2. **`time_step` and `n_integration_step` now actually mean what their names say.** Anyone reading [glacier.yaml](../glacier-code/particleda/glacier.yaml) and changing those parameters will get the behaviour they expect — not whatever the buggy CFL formula decided.

3. **CFL is a real constraint, not just a docstring word.** If we move to a stiffer dynamics (e.g. real WAVI ice flow), the internal dt limit could be much smaller than 400 s. We'll need to bump `n_integration_step` so that `time_step / n_integration_step` stays under the stability ceiling. The one-shot warning is now there to catch that automatically.

4. **The previous "runs 01–03" used different physics than this run.** Their internal dt was ~457 s; ours is 400 s. They're broadly comparable but not identical. Going forward, the bug-fixed dynamics is the canonical baseline. If we want to redo the seed sweep on the new code, that's an extra 10–15 minutes of compute.

---

## 7. What to do next

Two natural next steps. My recommendation in **bold**.

1. **Re-do the seed sweep on the bug-fixed dynamics.** Currently the only multi-seed evidence we have ([06_seed_sweep_robustness.md](06_seed_sweep_robustness.md)) was on the buggy code path. We should redo it with `time_step=400, n_int=1` at 5 seeds so we have a clean reference for everything downstream. ~15 minutes of compute.
2. **σ_obs knee-search at the new physics + 2c cadence.** With cadence locked to "every 2c", sweep σ_obs ∈ {0.05, 0.10, 0.20, 0.50} and find the tightest σ_obs that still keeps ESS healthy. Trades ensemble health for tracking accuracy. ~20 minutes of compute.
3. Swap the surrogate for WAVI (the actual ice-flow forward model). This is the project goal but it's a substantive change — the WAVI call is several orders of magnitude slower than the surrogate. Probably the right move *after* (1).

---

## 8. Glossary — terms used in this note

| Term | Plain meaning |
|---|---|
| **particle** | One candidate guess for the unknown field (β here). We have Np = 1000 of them. |
| **ensemble** / **swarm** | The whole collection of candidates. |
| **predict step** | Each particle runs the physics forward by one sub-step. |
| **update step** | Compare each particle's predicted observation to the real observation; assign a *weight* proportional to how good the match was. |
| **weights** | Numbers from 0 to 1, one per particle, summing to 1. Big weight = "this particle is plausible." Small weight = "this particle is unlikely to be the truth." |
| **resampling** | Replace the swarm with N draws (with replacement) from itself, where the probability of picking a particle equals its weight. Effectively kills bad particles and clones good ones. |
| **ESS — Effective Sample Size** | A number from 1 to Np that says "if these weights were turned into equal-weight draws, how many independent draws would they be worth?" If one particle has weight 1 and the rest weight 0, ESS = 1 (terrible). If all particles share the weight equally, ESS = Np (perfect). |
| **0.5·Np threshold** | A common rule of thumb: when ESS drops below half the ensemble, the filter is "starting to struggle." |
| **dt** | The size of one physics step in time. Smaller dt = more accurate but more compute. |
| **CFL condition** | A stability rule for advection problems: dt must be small enough that information doesn't move more than one grid cell per step. Concretely `dt ≤ k × dx / max_speed` where k is a problem-dependent safety factor (here 0.2). Violate it and the numerical scheme explodes. |
| **`time_step`** | (YAML config) The model-time covered by ONE filter step. |
| **`n_integration_step`** | (YAML config) How many internal physics sub-steps fit in one filter step. Internal dt = `time_step / n_integration_step`. |
| **σ_obs** | Standard deviation of the observation noise. How wrong our measurements are. |
| **σ_init** | Standard deviation of the initial spread of particles around the prior mean. How wide our initial guess is. |
| **σ_proc** (process noise) | Standard deviation of the random kick added to each particle per filter step. Represents un-modelled physics. |
| **prior** | What we think β looks like *before* seeing any observation. Here: 1000 + 500·sin(ωx)·sin(ωy) plus a wide spread. |
| **posterior** | What we think β looks like *after* updating with observations. The filter computes this. |
| **truth** | The β field that the synthetic-observation simulator used as ground truth. We can compare our posterior mean to it because we made it up ourselves. |
| **RMSE** | Root-mean-square error. √(average of squared differences between estimate and truth). Same units as the thing being estimated (β here, in Pa·s/m). |
| **pressure budget** | Our shorthand from earlier notes: `n_obs × (σ_signal / σ_obs)²`. Predicts when Bootstrap-PF collapses. Lower is better. |
