# Glacier Bootstrap-PF on ParticleDA

First port of the glacier β-estimation problem from `LowLevelParticleFilters` to `ParticleDA`. State is θ = log β; forward map is the fast surrogate `ux = 1e3 / β` (no WAVI on this pass).

## Files

- `glacier_model.jl` — `module GlacierModel` implementing the ParticleDA model interface (state dim, init, deterministic + stochastic update, observation operator, log-density, HDF5 IO).
- `glacier.yaml` — config (grid, sensor stride, noises, particle count).
- `run_glacier_pda.jl` — driver: simulates synthetic truth + observations, then runs `BootstrapFilter`.
- `results/` — outputs land here (`glacier_obs.h5`, `particle_da.h5`).

## Run

```bash
julia --project=test glacier-code/particleda/run_glacier_pda.jl
```

Default config: N = 200 particles, T = 50 steps, 40×40 grid, 100 sensors (`sensor_stride = 16`), σ_obs = 0.05 on a surrogate ux ≈ O(1).

## Tweaking

Edit `glacier.yaml` only — no code change needed:

- `nprt` — particle count
- `obs_noise_std` — the dominant collapse lever (per Phase B, see `glacier-notes/00_action_plan.md`)
- `sensor_stride` — bigger → fewer sensors → lower pressure
- `init_std_theta`, `process_std_theta` — initial spread + per-step noise in log β
- `simulate_observations.n_time_step` — length of the synthetic truth

## Outputs

- `results/glacier_obs.h5` — synthetic truth states + observations
- `results/particle_da.h5` — filter snapshots: `weights/t####`, `state_avg/t####/{beta, log_beta}`, parameters, grid + station coordinates, `beta_prior`

## What this run is and isn't

It's: a parity check that the ParticleDA plumbing handles a `nx*ny` log-β state, a non-LLW2d physics block, and scattered scalar-velocity observations.

It isn't: a healthy filter. Same Bootstrap-PF degeneracy rules apply (`pressure = n_obs × (σ_signal/σ_obs)²` — see `glacier-notes/00_action_plan.md`). The next steps after this run are loosening σ_obs and then plugging in WAVI as the forward map.
