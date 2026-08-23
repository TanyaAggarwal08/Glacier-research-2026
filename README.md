# glacier-code — what is in here

Everything in this folder is about one question:

> **The bed under a glacier grips the ice in places and lets it slide in others.
> We cannot dig down and measure that grip. But we CAN measure how fast the ice
> surface moves. Can we work backwards from the speed to the grip?**

The grip is called **basal drag**, written **β** (beta). The ice-flow model
**WAVI** goes forwards — give it β and it tells you the speed. Everything here
goes **backwards**, using a **particle filter**: keep hundreds of guesses for β,
run each one through WAVI, keep the guesses whose predicted speed matches the
real measurements, and repeat every hour.

This file tells you which folder does what. Each folder also has its own README
with the full detail.

---

## The short version

| folder | what it does | results land in |
|---|---|---|
| **`temperature_advection_run/`** | ⭐ **Start here.** β is driven by temperature and carried downstream. The most complete experiment. | `results/temperature_advection/` |
| **`double_bump_run/`** | β is a simple 2×2 bump pattern pushed around by advection. The original working setup. | `results/double_bump_sidewalls/` |
| **`temp_run/`** | No filter at all — just shows how temperature → β → ice speed, forwards. | `results/noise_temperature_forcing/` |
| `temp_pf_run/` | Estimates 2 numbers (the temperature→β recipe) instead of the whole field. | `results/temperature_pf_24h/` |
| `temp_field_run/` | Temperature only, **no** transport. A control experiment. | `results/temperature_field/` |
| `temp_advect_run/` | First version of the transport idea. Superseded by `temperature_advection_run`. | `results/temperature_advect/` |
| `particleda/` | The big working folder — every earlier experiment and plot. | `particleda/results/` |
| `testing_stage/` | Very early particle-filter practice scripts. | — |

Each of the six run folders is **self-contained**: one file, one command, all
results. You do not need anything else to run them.

> ⚠️ Use plain `julia`, **not** `julia --project=.`. The repository project
> environment has a broken NetCDF library and WAVI will not load from it.

---

## Things that are the same in every experiment

Rather than repeat these six times, they are listed once here. Any folder that
differs says so in its own README.

**The glacier**

| | |
|---|---|
| domain | 160 km × 160 km square |
| grid | 40 × 40 cells, so each cell is 4 km |
| ice surface | `z_s = 1060·√(1 − x/L)` — a ramp, thickest at `x = 0`, thinning downstream |
| bed | flat |

**What we measure**

| | |
|---|---|
| sensors | 16, on a 4×4 grid at x and y = 20, 50, 110, 140 km |
| measured | `log(surface speed)`, once per hour |
| measurement error | σ = 0.5 in log units (so roughly a percentage error) |

We are trying to find **1600 numbers** (β in every cell) from **16 numbers per
hour**. That imbalance is the central difficulty in all of this.

**The sliding law** — `weertman_m = 1`

This makes the physics as simple as possible: `τ_b = β·u`, so

```
speed  ≈  driving stress / β
```

Higher β ⇒ stickier bed ⇒ slower ice. With `m = 1` the number we estimate *is*
exactly the drag coefficient, with nothing hidden in between.

**Edges of the domain**

```julia
u_iszero = ["north"]                 # a wall at x = 0 (the ice divide)
v_iszero = ["south","east","west"]   # free-slip side walls
```

⚠️ WAVI's names are **array positions, not compass directions**: "north"/"south"
are the **x** edges, "east"/"west" are the **y** edges. Zero the component that
*crosses* each edge. Leaving the side walls out makes them behave like a cliff
edge, and the ice there runs ~32× too fast.

**The filter**

| | |
|---|---|
| particles | 500 guesses |
| tempering stages | 10 |
| resampling | when the effective sample size drops below N/2 = 250 |
| jitter after resampling | smooth noise, amplitude 20 |
| noise smoothness | 15 km correlation length |

*Tempering* means the measurement is applied in 10 gentle nudges instead of one
hard shove. Without it, one particle grabs all the weight immediately and the
whole ensemble collapses at the first step.

*Jitter* means shaking the copies slightly after resampling. Without it the
copies are identical and the ensemble stops being able to explore.

**How the guessing starts** (in every filter experiment)

```
background = truth + 300·(a smooth random wave)   ← ONE wrong field, same for everyone
particle   = background + 200·(a smooth random wave)   ← a different guess each
```

So the filter starts about 300 Pa·s/m away from the answer, and the 500 particles
are scattered around that wrong starting point.

---

## The experiments

### ⭐ `temperature_advection_run/` — the main one

**The question:** can we recover the β field when β is driven by something
physically believable (temperature) instead of an invented pattern?

**How β changes each hour:**

```
β_next(x) = β_now(x − v·dt)   −   k·(T_next − T_now)   +   noise
            └─ carried along ─┘   └─ warm = slippery ─┘
```

Two things happen at once:

1. **Warm bed = slippery bed.** `β = C − k·(T − T_ref)`, so when the surface
   warms 1 °C, β drops by 40. Over a day/night cycle that swings β by ±400.
2. **The pattern gets carried downstream** at `v = 2800/β` — a slippery bed
   carries its own pattern faster.

Part 2 is what makes it work. Part 1 alone does nothing for the filter, because
it is added to the truth and to every guess equally and so cancels out.

**Parameters**

| | value | meaning |
|---|---|---|
| `C_MAP` | 2000 | β at the reference temperature |
| `K_MAP` | 40 | β drops this much per °C of warming |
| `T_REF` | −10 °C | the reference temperature |
| `V_SCALE` | 2800 | sets the transport speed, `v = 2800/β` ≈ 2 m/s |
| `V_MIN, V_MAX` | 0.5, 5.0 m/s | safety clamp so the numerics stay stable |
| `NSUBSTEP` | 10 | advection substeps per hour |
| temperature | ±10 °C daily, ±15 °C seasonal, 1.0 °C wobble | known to everyone |
| process noise | 10 per step | each particle's freedom to explore |
| steps | 24 hours = one day/night cycle | |

**Run it**

```bash
julia -t 10 glacier-code/temperature_advection_run/run_temperature_advection.jl
```

**Results:** `temperature_advection_run/results/temperature_advection/`
— `beta_evolution.gif`, `velocity_evolution.gif`, `rmse_beta.png`,
`diagnostics.png`, `ess_tracking.png`, `forcing_timeseries.png`, `tracking.h5`,
`summary.txt`

**Outcome:** β error 299 → 125 (**58% better**), ice speed error 2.75 → 0.96
m/yr, about 24 minutes.

---

### `double_bump_run/` — the original working setup

**The question:** can the filter recover a β field at all, using a clean
made-up pattern?

**The β pattern (the "double bump"):**

```
β = 2000 + 1500·sin(ωx)·sin(ωy)        ω = 2π/L
```

That tiles the domain with one 2×2 checkerboard of sticky and slippery patches,
β running from 500 to 3500.

**How β changes each hour** — advection, with a made-up velocity:

```
velocity = 1 + 0.0005·β        (sticky patches move faster)
```

**Parameters**

| | value |
|---|---|
| β centre / amplitude | 2000 / 1500 |
| advection coefficient `EPS_ADV` | 5e-4 |
| substeps per hour | 10 |
| process noise | 10 per step |
| steps | 20 hours |

**Run it**

```bash
julia -t 10 glacier-code/double_bump_run/run_double_bump_sidewalls.jl
```

**Results:** `double_bump_run/results/double_bump_sidewalls/`

**Outcome:** β error 299 → 113 (**62% better**), ice speed error 8.07 → 0.79
m/yr, about 35 minutes.

---

### `temp_run/` — forwards only, no filter

**The question:** if the temperature goes up and down over three days, what
happens to β and to the ice speed? No estimation here at all — just the physics,
running forwards.

**What it computes:**

```
temperature  →   β = 2000 − 40·(T + 10)   →   WAVI   →   ice speed
```

with a realistic wobble added to the temperature so it looks like a real record
rather than a clean sine wave.

**Parameters**

| | value |
|---|---|
| length | 73 steps = 3 days, hourly |
| temperature noise | σ = 2.0 °C, timescale 6 h, 15 km smooth |
| β map | `β = 2000 − 40·(T + 10)`, floor 10 |

**Run it**

```bash
julia -t 4 glacier-code/temp_run/run_temperature_noise.jl
```

**Results:** `temp_run/results/noise_temperature_forcing/` — about 85 seconds.

**What it shows:** β and speed are almost perfectly anti-correlated in time
(−0.99) — when the bed softens, the ice speeds up. The daily speed swing is 59%
of the mean. The added noise changes speed by only 1.4%, so the day/night cycle
dominates.

---

### `temp_pf_run/` — estimating the recipe instead of the field

**The question:** suppose we do **not** know how β responds to temperature. Can
the filter learn the two numbers `C` and `k` from the ice speed?

**What is unknown here:**

```
β = C − k·(T − T_ref) + δ
    └── 2 unknown numbers ──┘  └ 1600-cell leftover ┘
```

so the state has 1602 pieces instead of 1600.

**Parameters**

| | value |
|---|---|
| true values | `C` = 2000, `k` = 40 |
| starting guess | `C` ~ 2300 ± 300, `k` ~ 20 ± 12 (deliberately wrong — `k` is half the truth) |
| leftover field δ | starts at spread 50, grows 10 per step |
| steps | 24 |

**Run it**

```bash
julia -t 10 glacier-code/temp_pf_run/run_temperature_pf.jl
```

**Results:** `temp_pf_run/results/temperature_pf_24h/` — includes
`parameter_recovery.png` and `parameter_scatter.png`, which are the key figures.

**Outcome:** it finds both numbers almost exactly — `C` = 1979.5 (0.9% off) and
`k` = 40.24 (0.6% off). Total β error 439 → 67.

⚠️ **Read the split, not just the total.** The run prints the error separately
for the *recipe* and the *leftover field*: the recipe error falls 440 → 22, but
the leftover field error grows 2 → 63 and never recovers. 16 sensors can pin
down 2 numbers easily; they cannot pin down 1600.

---

### `temp_field_run/` — the control experiment

**The question:** what happens with temperature but **no** transport? This exists
to prove a point, not to succeed.

**How β changes each hour:**

```
β_next = β_now − k·(T_next − T_now) + noise
```

That is the temperature part only — no advection.

Everything else is identical to `temperature_advection_run`. It is the same
experiment with one piece removed, which is what makes it a fair control.

**Run it**

```bash
julia -t 10 glacier-code/temp_field_run/run_temperature_field_pf.jl
```

**Results:** `temp_field_run/results/temperature_field/`

**Outcome:** β error 299 → 264, only **12% better**. The run also measures *why*:
the error pattern stays 85% unchanged all day. Because the temperature term is
added to the truth and to every guess equally, it cancels out of the error, so
the error field never moves and the 16 sensors keep re-asking the same question.

---

### `temp_advect_run/` — first version of the transport idea

Same experiment as `temperature_advection_run`, written first. The physics,
parameters and results are identical — the newer folder is a tidied rewrite with
fuller comments.

**Keep it for:** the record of how the idea was tested. **Use instead:**
`temperature_advection_run/`.

**Results:** `temp_advect_run/results/temperature_advect/`

---

## Everything else

### `particleda/` — the big working folder

This is where most of the earlier work happened: **37 scripts** and **35 result
folders**. Nothing here is tidied up, and paths inside assume you run from this
folder. It is kept as the record of what was tried.

Two naming patterns:

- **`run02_` … `run17_`** — the step-by-step build-up. Prior width, number of
  sensors (16 vs 30), sensor placement, observations vs none, working in
  log-β space, advection, and finally tempering (`run17`).
- **`experiment_18_*`** — the WAVI-based experiments. `wavi_obs` is the
  pseudo-random β wave; `double_bump` and its variants (`_sidewalls`,
  `_center1000`, `_weertman_m`) are the bump pattern.

Each `run_*.jl` has a matching `plot_*.jl`. Also here:

| file | what it is |
|---|---|
| `ice_flow.jl` | shared WAVI setup used by several experiments |
| `glacier_model.jl` | glacier geometry helpers |
| `ice_experiment_dynamics.jl` | advection and noise routines |
| `temperature_driven_flow.jl`, `temperature_driven_flow_noisy.jl` | early temperature experiments, later tidied into `temp_run/` |

### `testing_stage/` — earliest practice

`particlefiltering4.jl`, `particlefilteringdiscrete.jl`,
`particlefilteringwithiceflow.jl` — learning the particle filter on small
problems, before WAVI was involved. Historical only.

### Loose scripts at the top level

| file | what it does | output |
|---|---|---|
| `main.jl` | draws the glacier geometry and a sample β field | `results/beta_field_expC.png`, `results/glacier_surface_2D.png` |
| `non_linear_advection.jl` | standalone nonlinear advection demo, no glacier | `results/quasilinear_advection.gif` |
| `particlefilteringnonlinear.jl` | particle filter using the `LowLevelParticleFilters` package | the `*non--linear*` files in `results/` |

> **Open questions carried over from the previous README:**
> `main.jl` — needs tidying.
> `non_linear_advection.jl` — check whether this is the version the professor
> supplied or the one we developed as the final nonlinear advection.

### `results/` (top level)

Output from the three loose scripts above — the early β wave GIFs, RMSE plots and
geometry figures. **Not** from any of the six run folders, which each keep their
results inside themselves.

### `refs/`

`snyder2008.pdf` — the paper on why particle filters struggle in high dimensions.
This is the reason tempering is needed everywhere in this project.

---

## Reading the output files

Every run folder produces the same set:

| file | what to look at |
|---|---|
| `summary.txt` | every number the run printed — read this first |
| `rmse_beta.png` | is the β error going down? |
| `beta_evolution.gif` | truth β \| our estimate \| the error, hour by hour |
| `velocity_evolution.gif` | the same for ice speed |
| `ess_tracking.png` | is the ensemble healthy? (should stay near the N/2 line) |
| `tracking.h5` | all the raw numbers, for your own plots |

**Two things to know when reading the pictures:**

- Colours are **fixed across all frames** of a GIF. If they rescaled each frame,
  a filter that was doing nothing would still look like it was converging.
- Error panels are **blue–white–red around zero**, so "getting better" reads as
  the picture fading to white.
