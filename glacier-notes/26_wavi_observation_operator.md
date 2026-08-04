# WAVI as the observation operator — Experiment 18

Purpose: document the first particle-filter run that uses the **real WAVI.jl
ice-sheet model** as the observation operator, applied per particle inside the
likelihood, instead of the algebraic surrogate `ux = 1000/β` (or its log) used
in every earlier run. This is the "base" setup we intend to build the main
runs on: upwind + nonlinear advection + pseudo-random-wave prior + tempering +
WAVI observations.

## 1. What changed vs everything before

Until now the observation operator was a cheap stand-in for ice velocity:

- runs 11–16: `h(β) = ux = 1000/β` (surrogate velocity), later `log(β)`.
- Exp 18: `h(β) = log|u_WAVI(β)|` at the sensor cells — the log of the true
  modelled surface speed from a full WAVI momentum solve.

β is still the state and still evolves by the glacier advection dynamics; only
the β → observation map changed, from one line of algebra to a PDE solve. WAVI
is wrapped in `glacier-code/particleda/ice_flow.jl` (`IceFlow.velocity_flat`),
which builds a `weertman_c = β` model on the 40×40 / 160 km grid and returns
`(u, v)`; we observe `log(sqrt(u² + v²))` at the 16 sensor cells.

Why **log** speed: WAVI surface speed spans ~1,500–15,000 m/yr across the
domain (two orders of magnitude), so a single additive-Gaussian σ is only
meaningful in log space (relative error). Same reasoning as the `log_beta`
work in `25_observation_error_analysis.md`.

## 2. The environment problem and the standalone-dynamics port

WAVI and ParticleDA **cannot co-load in one Julia environment**:

- Under `--project=test` (dev ParticleDA) WAVI fails to precompile — a broken
  `libnetcdf` artifact via NCDatasets.
- The default env has a working WAVI but no ParticleDA.

Earlier the two-stage Exp 09 sidestepped this by running WAVI *post-hoc* on the
truth and mean only. Exp 18 cannot: it needs WAVI **inside** the PF loop, once
per particle per step. So the β-dynamics were ported into a ParticleDA-free
module, `glacier-code/particleda/ice_experiment_dynamics.jl`, which runs under
the default env alongside WAVI.

The port is a verbatim copy of glacier_model.jl's dynamics (pseudo-random-wave
prior, squared-exponential smooth-noise Cholesky, upwind/LW advection, process
noise, RPF jitter) with the ParticleDA dispatch stripped. It was **validated
bit-for-bit** against glacier_model.jl:

| quantity | glacier_model.jl | standalone port |
|---|---|---|
| truth prior sum | 2965971.153246 | 2965971.153246 |
| sensor indices | identical | identical |
| noise factor L[1,1], diag-sum | 1.0, 3.269699 | 1.0, 3.269699 |
| upwind 1-step range | [1027.108868, 2436.968188] | [1027.108868, 2436.968188] |
| LW 1-step range | [1012.2316, 2447.1321] | [1012.2316, 2447.1321] |

Both advection branches match exactly, so the port is trustworthy.

## 3. Parameters (run: `experiment_18_wavi_obs.jl 500 20 10 0.5 5e-4 full`)

| Group | Name | Value | Notes |
|---|---|---:|---|
| Filter | N (particles) | 500 | verification scale, not main-run 1000 |
| Filter | T | 20 | model-hours (20 hourly steps) |
| Filter | K (tempering) | 10 | linear schedule φ_k = k/K |
| Filter | ESS threshold | 250 | resample when ESS < N/2 |
| Filter | SIGMA_JITTER | 20.0 | RPF jitter, β units |
| Filter | SEED_PF / SEED_OBS | 42 / 123 | |
| Obs | operator | `log|u_WAVI(β)|` | 16 sensors, `stations_grid_16.txt` |
| Obs | σ_obs | 0.5 | log-speed units (relative error) |
| Dynamics | advection | nonlinear upwind | v = 1 + ε·β |
| Dynamics | ε (advection_epsilon) | 5e-4 | established nonlinear value |
| Dynamics | n_integration_step | 10 | inner dt = 360 s |
| Dynamics | prior | pseudo_random_wave | center 2000, signal 300, background 300 |
| Dynamics | init / process std (β) | 200 / 10 | |
| Dynamics | noise ℓ | 15 km | squared-exponential GRF |

Note the advection is **upwind, not Lax–Wendroff**. run11–17 used
`lax_wendroff`; we reverted to upwind for the base setup because of the vague
RMSE behaviour LW gave (numerical dispersion; see `21_numerical_diffusion.md`).

σ_obs = 0.5 was chosen over a sharper 0.3: both keep ESS healthy under
tempering, but 0.3 raises the tempering cost (see §5). 0.5 gives the same
fractional-error model with lower overhead.

## 4. Results — the filter recovers β from the real ice-sheet model

| metric | value |
|---|---:|
| RMSE(β) initial → final | **299.2 → 143.9** (more than halved) |
| mean ESS | 292 (58% of N) |
| min ESS | **230 at t=1** (46% of N) — no first-step collapse |
| velocity RMSE at sensors (abs) | 1089 → 425 m/yr |
| velocity RMSE at sensors (rel) | 17.7% → 7.6% |
| wall time | 96 min |

Key points:

- **β is recovered.** Global RMSE(β) more than halves. The ensemble mean tracks
  the truth cross-section closely and the particle band brackets it, tightening
  at the sensor columns.
- **No first-step collapse.** min ESS is 230 (46% of N) at t=1 — the widest-
  ensemble first step that collapsed to single digits in the un-tempered runs
  (documented for log_beta in `25_observation_error_analysis.md`; the tempering
  fix was established in Exp 14–17) now holds. ESS oscillates around N/2 for the
  whole run.
- **Velocity improves too, and corroborates β.** Velocity RMSE at the 16
  sensors (computed from the stored log-speeds, `speed = exp(log-speed)`, no
  WAVI re-solve) drops from 17.7% to 7.6% relative. β RMSE and velocity RMSE
  move together — a consistency check that the *state* is genuinely improving,
  not just fitting observation noise. Both share a mid-run bump around
  t = 8–13 (the nonlinear-advection / prior-mismatch phase) before settling.

### Honest caveats

- The truth-vs-mean **velocity heatmaps look nearly identical**, which
  overstates the β recovery: WAVI surface speed is dominated by the fixed ice
  geometry (√-profile surface slope), with β only modulating it. The β field
  and RMSE are the honest recovery measures — good on large scales, smoother
  than truth in fine detail (expected with 16 sensors).
- **Single seed.** One (SEED_PF, SEED_OBS) realisation; no error bars.
- **Verification scale.** N = 500, T = 20 — enough to confirm the setup works,
  not the eventual main-run fidelity (N = 1000).

## 5. Compute cost and parallelism

One WAVI solve ≈ 1.2–1.4 s (full 40×40 momentum solve). The PF calls the
operator once per particle per step, so N=1000×T=100 would be ~33 h serial —
infeasible. Three levers brought Exp 18 to **96 min**:

1. **T = 20** instead of 50/100.
2. **σ = 0.5** instead of 0.3 → tempering overhead ×1.35 here (vs ×2.02 at
   σ=0.3; sharper likelihood → more intra-step resamples → more WAVI recomputes,
   since the likelihood must be recomputed at moved particle positions after
   each resample).
3. **Threading over particles.** The WAVI likelihood is embarrassingly parallel;
   we thread it with `Threads.@spawn` (run with `julia -t 10`), the same pattern
   ParticleDA uses in `src/filters.jl` (chunk particles, one spawned task per
   chunk, per-task scratch buffers via `task_index`). Concurrent WAVI was
   verified **bit-for-bit identical** to serial. Effective per-solve time fell
   1.4 s → **0.426 s (≈3.3×)**.

Ceiling: threading saturates at ~2.5–3.3× because WAVI allocates a fresh model
per call and Julia's GC pauses all threads (GC contention, not BLAS — forcing
`BLAS.set_num_threads(1)` did not change it). MPI (separate processes, separate
GCs), the way ParticleDA also parallelises, would scale further but is a larger
lift given the WAVI/ParticleDA env conflict.

Total for this run: 13,540 WAVI solves, 8 resamples, ×1.35 tempering overhead.

Thread-safety notes baked into the script: WAVI prints solver progress to
stdout, so `redirect_stdout` is applied **once around each WAVI region** (never
per-call inside threads — that races); the solve counter is a `Threads.Atomic`;
only the RNG-free likelihood loop is threaded (propagation, RNG and jitter stay
serial).

## 6. Files and outputs

- `glacier-code/particleda/experiment_18_wavi_obs.jl` — the run (ARGS:
  `N T K σ ε mode`; `mode=calib` reports timing only).
- `glacier-code/particleda/ice_experiment_dynamics.jl` — ParticleDA-free
  dynamics port (validated bit-for-bit).
- `glacier-code/particleda/ice_flow.jl` — WAVI wrapper (β → velocity).
- `glacier-code/particleda/stations_grid_16.txt` — 16-sensor 4×4 grid.
- `glacier-code/particleda/plot_exp18_velocity_rmse.jl` — velocity RMSE plots.
- Graphs in `results/experiment_18_wavi_obs/`: `rmse_beta.png`,
  `ess_tracking.png`, `crosssection_final.png`, `sensor_trail.png`,
  `beta_field_truth_vs_mean.png`, `velocity_field_truth_vs_mean.png`,
  `obs_fit.png`, `velocity_rmse_abs.png`, `velocity_rmse_rel.png`.
- Particle trajectory + checkpoint (135 MB each) on the external SSD at
  `/Volumes/ZX20/USRA 2026/experiment_18_wavi_obs/`.

## 7. How to reproduce

```bash
# default env (has WAVI); 10 threads
julia -t 10 glacier-code/particleda/experiment_18_wavi_obs.jl 500 20 10 0.5 5e-4 full
julia glacier-code/particleda/plot_exp18_velocity_rmse.jl
```

## 8. Next steps

- Repeat seeds for error bars (cheap now: ~1.6 h/run threaded).
- Push N back to 1000 and/or T to 50 for a main-run once the setup is trusted.
- Investigate MPI to beat the ~3× threading ceiling for the expensive main run.

## See also

- `23_ice_flow_integration.md` — the WAVI wrapper and ice-flow integration.
- `24_pseudorandom_wave_run.md` — the pseudo-random-wave prior this builds on.
- `25_observation_error_analysis.md` — the σ calibration and the t=1 ESS
  collapse (for the log_beta run) that this setup's tempering is designed to
  fix. Note: the tempering experiments themselves (Exp 14–17, which established
  K=10 and the peak-τ² reading of the t=1 collapse) are not yet written up in a
  note; the record is in `glacier-code/particleda/rmse_experiments/`.
- `21_numerical_diffusion.md` — why upwind over Lax–Wendroff.
