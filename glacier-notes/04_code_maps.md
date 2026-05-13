# Code Maps — ParticleDA + glacier-code

> Quick-reference diagrams so I don't re-read sources. Built 2026-05-13 from `graphify-out/GRAPH_REPORT.md` + targeted reads of `src/`, `test/models/llw2d.jl`, `glacier-code/main.jl`, `glacier-code/particlefilteringnonlinear.jl`. Re-run `graphify update .` if code shifts.

---

## 1. ParticleDA.jl — module architecture

```
src/ParticleDA.jl  (module root, re-exports)
│
├── params.jl       FilterParameters, get_params()
├── io.jl           HDF5 reading/writing, read_input_file(), YAML loader
├── models.jl       AbstractModel interface (signatures Models must implement)
├── statistics.jl   MeanSummaryStat, MeanAndVarSummaryStat (+ Naive variants)
├── filters.jl      ★ Core: BootstrapFilter, OptimalFilter, init_filter,
│                   sample_proposal_and_compute_log_weights!,
│                   update_states_given_observations!, run_particle_filter
├── utils.jl        helpers
├── testing.jl      run_unit_tests_for_generic_model_interface, etc.
└── kalman.jl       submodule Kalman: KalmanFilter, MatrixFreeKalmanFilter
                    (used for validation against linear-Gaussian baseline)
```

### Filter dispatch (filters.jl)

```
ParticleFilter (abstract)
   ├── BootstrapFilter  ← uses prior as proposal, weights = p(y|x)
   └── OptimalFilter    ← optimal proposal for conditionally linear-Gaussian
                          uses OfflineMatrices + OnlineMatrices
                          requires get_covariance_* methods on model
```

### Main loop (conceptual)

```
run_particle_filter(init_model, input_file, output_file, FilterType)
  │
  ├── init_model(...)               ← user provides; returns Model
  ├── init_filter(params, model, nprt/rank, FilterType, SummaryStat)
  ├── sample_initial_state!(model, states, rng)      (per particle)
  │
  └── for t in 1:n_time_step:
        update_state_deterministic!(model, state)    ← sub-stepped n_integration_step
        update_state_stochastic!(model, state, rng)  ← GRF noise
        sample_proposal_and_compute_log_weights!(...)
        normalise_weights → ESS → resample if needed
        write_snapshot to HDF5
```

### Model interface (what LLW2d implements — template for GlacierModel)

```
get_state_dimension / get_observation_dimension / *_eltype
get_initial_state_mean! / sample_initial_state!
update_state_deterministic!   ← physics (LLW2d: shallow water, glacier: ice flow)
update_state_stochastic!      ← GRF stochastic forcing
get_observation_mean_given_state!  ← H(x): map state → obs space
sample_observation_given_state!
get_log_density_observation_given_state
# Optional for OptimalFilter:
get_covariance_state_noise
get_covariance_observation_noise
get_covariance_observation_observation_given_previous_state
get_covariance_state_observation_given_previous_state
get_state_indices_correlated_to_observations
write_state / write_model_metadata
```

### LLW2d.jl key pieces (the template to clone)

```
LLW2dModelParameters           ← YAML-driven config
LLW2dModel{T,U,G}              ← holds grid, params, RandomField, station indices
RandomField{T,F}               ← Matérn GaussianRandomField wrapper

init(parameters_dict)          ← entry point passed to run_particle_filter
get_grid_axes / get_station_grid_indices (txt-file or auto-generated)
flat_state_to_fields           ← reshape vec ↔ (nx,ny,n_fields=3) [h,u,v]
update_state_deterministic!    ← n_integration_step sub-steps shallow-water
update_state_stochastic!       ← add Matérn GRF sample per field
```

---

## 2. glacier-code/main.jl — WAVI ice-sheet forward model (no DA yet)

**Purpose:** standalone WAVI.jl driver. Steady-state ice-flow solve at fixed β.
**Not yet wired to any PF.**

```
model_code()
  │
  ├── Define surface elevation z_s(x) = 1060·√(1 − x/L)     (parabolic dome)
  ├── Grid: 80×80, L = 160 km, dx = dy = 2 km
  ├── z_b = zeros (flat bed); h_init = z_s − z_b
  │
  ├── β field = 1000 + 1000·sin(ωx)·sin(ωy), ω = 2π/L      (sinusoidal drag)
  │
  ├── Build WAVI Model(grid, bed, IC, Params(beta=β))
  ├── update_state!(model)         ← solves momentum → u, v
  │
  └── Plots:
        glacier_surface_2D.png         (cross-section)
        beta_field_expC.png            (β heatmap)
        velocityheatmapexpC.png        (u heatmap)
        infintetifinite3.png           (u along flowline at x = L/4)
```

**Status:** physics generator only — produces a "truth" β + velocity. **No** particle filter, no observation sampling, no time stepping yet. This is the forward model that will plug into `update_state_deterministic!` in the eventual GlacierModel (Phase A).

---

## 3. glacier-code/particlefilteringnonlinear.jl — toy 2D advection + LowLevelPF

**Purpose:** prove a 2D-field bootstrap PF works in `LowLevelParticleFilters.jl` on a cheap toy. The basis for the Np=10000 ESS observations in [01_lowlevel_Np10000_observations.md](01_lowlevel_Np10000_observations.md).

```
main()
  │
  ├── Grid: 40×40 (state dim N = 1600), T = 200 steps, Np = 10000
  │
  ├── initial_field()  =  sin(2πx)·sin(2πy)              (smooth sinusoid)
  │
  ├── dynamics(x, ...)  =  non-linear upwind advection right
  │   velocity = 1 + ε·β,  ε = 0.001                      (state-dependent speed)
  │   dt = 0.2·dx/max_speed                               (CFL)
  │
  ├── measurement(x, ...) = x[sensor_indices]
  │   sensor_indices = 1:16:N    →  100 stations
  │
  ├── Noise:
  │     process     = MvNormal(0, 0.01²·I)
  │     measurement = MvNormal(0, 0.1²·I)
  │     init_state  = MvNormal(field, 0.8²·I)            ← WIDE σ_init = key
  │
  ├── pf = ParticleFilter(Np, dynamics, measurement, ...)
  │
  ├── Truth loop:    advance x_true, generate noisy obs
  ├── Filter loop:   pf(nothing, obs[t]); estimate = weighted_mean(pf)
  │                  log ESS, max/min weight, var, entropy
  │
  └── Plots:
        beta_wave_motion*.gif              (truth field animation)
        beta_estimate_wave_motion*.gif     (PF mean animation)
        beta_difference_wave_motion*.gif   (truth − estimate)
        rmse_particle_filter_*.png         (global + single-point RMSE)
        ess_evolution_non--linear.png      (★ the Np=10000 plot)
        weight_extremes / variance / entropy plots
        particle_ensemble_point.png        (histogram at (Nx/4, Ny/4))
        beta_point_obs_estimate.png        (truth vs est vs obs at one cell)
```

### Pressure budget for this toy

```
n_obs = 100,  σ_signal ≈ 1,  σ_obs = 0.1
pressure = 100 × (1/0.1)² = 10000   ← far above the collapse threshold
```

Yet ESS stays at full Np most of the time with ~10 sharp dips to ~10–15% of Np.
Why it works anyway: **σ_init = 0.8 vs σ_signal ≈ 1** gives a genuinely diverse
starting ensemble; adaptive resampling kicks in only on dips. See
[01_lowlevel_Np10000_observations.md](01_lowlevel_Np10000_observations.md) for
the full analysis.

---

## 4. How the three pieces will eventually connect (Phase A target)

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ glacier-code/main.jl        │         │ glacier-code/particlefilter… │
│ WAVI forward model          │         │ LowLevel PF on toy advection │
│ (β, h, u, v fields)         │         │ (sanity check only)          │
└──────────────┬──────────────┘         └──────────────────────────────┘
               │
               │ wrap as update_state_deterministic!
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NEW: src/glacier_model.jl  (clone of test/models/llw2d.jl)          │
│   - LLW2d → GlacierModel                                            │
│   - shallow-water update → WAVI ice-flow update                     │
│   - 3 fields (h,u,v) → β + maybe h                                  │
│   - keep: Matérn GRF, scattered stations, HDF5 IO, sub-stepping     │
└──────────────┬──────────────────────────────────────────────────────┘
               │ init_model(dict)
               ▼
        run_particle_filter(init, "glacier.yaml", "out.h5", BootstrapFilter)
                                                    │
                                                    └─ MPI when wall-clock demands
```

---

## 5. Pointers (don't re-read these files; check this map first)

| File | What's there |
|---|---|
| [src/ParticleDA.jl](../src/ParticleDA.jl) | module + run_particle_filter |
| [src/filters.jl](../src/filters.jl) | Bootstrap/Optimal + Offline/OnlineMatrices |
| [src/models.jl](../src/models.jl) | AbstractModel interface signatures |
| [test/models/llw2d.jl](../test/models/llw2d.jl) | template for GlacierModel |
| [glacier-code/main.jl](../glacier-code/main.jl) | WAVI standalone forward run |
| [glacier-code/particlefilteringnonlinear.jl](../glacier-code/particlefilteringnonlinear.jl) | LowLevel toy PF (sanity reference) |
| [bootstrap-pf-experiments/](../bootstrap-pf-experiments/) | tsunami sweep results + drivers |
| [graphify-out/GRAPH_REPORT.md](../graphify-out/GRAPH_REPORT.md) | 365 nodes, 28 communities, god nodes |
