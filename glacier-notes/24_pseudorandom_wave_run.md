# Pseudo-random wave prior/background run — run11 reference

Purpose: document the Evensen-style pseudo-random `β` experiment we just
added to the glacier ParticleDA workflow, including the exact parameter
choices, what changed in code, and what outputs to look at.

## What we built

Three pieces:

- `glacier-code/particleda/glacier_model.jl`
  Adds a new `prior_mode = "pseudo_random_wave"` option alongside the
  legacy sinusoidal prior.
- `glacier-code/particleda/run_pseudorandom_wave.jl`
  Runs the particle filter on the pseudo-random truth/background setup and
  saves `tracking.h5`.
- `glacier-code/particleda/plot_pseudorandom_wave.jl`
  Reads that tracking file and writes the diagnostics and GIFs into
  `results/run11_pseudorandom_wave/`.

The idea is to replace the smooth double-bump prior with a low-wavenumber
random wave field closer to the MATLAB prototype:

```text
truth field      = centre + signal_scale * normalised_random_wave(seed_truth)
background field = truth   + background_std * normalised_random_wave(seed_background)
```

Truth and background are therefore **different fields**, but both are
smooth and live in the same low-frequency mode family.

## The pseudo-random wave formula

For `kx, ky = 0:Kmax`, we sum separable sine products with random
amplitudes and random phases:

```text
w(x, y) = Σ a(kx,ky) sin(2πkx i / nx + phx) sin(2πky j / ny + phy)
```

Then we normalise the whole field by its spatial standard deviation, so the
signal-scale parameter has a direct interpretation:

```text
w_norm = w / std(w)
truth  = β0 + σ_sig * w_norm
```

This matches the MATLAB construction structurally, just translated into the
Glacier/Julia state layout.

## Parameters used in run11

These are the exact settings from
`glacier-code/particleda/run_pseudorandom_wave.jl`.

| Group | Name | Value | Unit | Role |
|---|---|---:|---|---|
| Filter | `NPRT` | 1000 | — | particles |
| Filter | `T` | 100 | steps | 100 hourly updates |
| Filter | `K_TRACK` | 15 | — | tracked-particle subset for cross-sections |
| Filter | `SEED_PF` | 42 | — | particle-filter RNG |
| Filter | `SEED_OBS` | 123 | — | truth / observation RNG |
| Filter | `ESS_THRESHOLD` | 500 | — | resample when `ESS < 0.5*N` |
| Filter | `SIGMA_JITTER` | 20.0 | β units | post-resample RPF jitter |
| Grid | `nx`, `ny` | 40, 40 | — | state dimension = 1600 |
| Grid | `x_length`, `y_length` | 160 000, 160 000 | m | physical domain |
| Prior | `prior_mode` | `pseudo_random_wave` | — | switch away from sinusoidal prior |
| Prior | `prior_center_beta` | 2000.0 | β units | truth-field offset |
| Prior | `prior_signal_scale_beta` | 300.0 | β units | pseudo-random truth amplitude |
| Prior | `background_std_beta` | 300.0 | β units | truth-to-background offset scale |
| Prior | `prior_max_wavenumber` | 2 | — | low-frequency mode cutoff |
| Prior | `prior_truth_seed` | 11 | — | truth wave seed |
| Prior | `prior_background_seed` | 29 | — | background wave seed |
| Init spread | `init_std_beta` | 200.0 | β units | particle spread around the background |
| Process | `process_std_beta` | 10.0 | β units | smooth process noise per step |
| Process | `noise_length_scale` | 15 000 | m | GRF correlation length |
| Process | `min_beta` | 10.0 | β units | positivity floor |
| Obs | `obs_noise_std` | 0.10 | ux units | observation noise in `ux = 1000/β` |
| Obs | station file | `stations_crosssection.txt` | — | 10 sensors, 1 on the 76 km row |
| Adv | `advection_type` | `lax_wendroff` | — | second-order advection |
| Adv | `advection_epsilon` | 0.0 | — | linear advection (`v = 1`) |
| Adv | `n_integration_step` | 10 | — | substeps per filter step |
| Adv | `time_step` | 3600.0 | s | hourly cadence; inner `dt = 360 s` |

Why these values:

- `background_std_beta = σ_sig = 300` follows the "background variance
  comparable to signal variance" suggestion.
- `process_std_beta = 10` is about 3 % of `σ_sig`, which sits inside the
  suggested 1–5 % per-step range.
- `obs_noise_std = 0.10` is the same loose observation model we already use
  to keep bootstrap-PF ESS from collapsing.
- `SIGMA_JITTER = 20` is large enough to break duplicate particles after
  resampling, but smaller than both the background mismatch and the initial
  ensemble spread.

## What changed in the model

`glacier_model.jl` now supports two prior families:

- `double_bump` — the old sinusoidal `2000 + 2000 sin sin`
- `pseudo_random_wave` — the new MATLAB-style low-mode random wave

For the pseudo-random branch, the model stores **two** fields:

- `beta_prior_mean` — the background field used as the particle initial mean
- `truth_prior_mean` — the distinct truth field used by the run11 driver

That is the main conceptual difference from the old runs, where truth and
particle mean both came from the same stationary sinusoidal field.

## How run11 is executed

Command pair:

```bash
julia --project=test glacier-code/particleda/run_pseudorandom_wave.jl
julia --project=test glacier-code/particleda/plot_pseudorandom_wave.jl
```

The driver uses the same broad PF recipe as the LW reference run:

1. Start the truth from `truth_prior_mean`.
2. Start particles from `beta_prior_mean + smooth init noise`.
3. Advect each particle with Lax-Wendroff.
4. Add smooth process noise.
5. Weight by the sparse `ux` observations.
6. Resample only when `ESS < 0.5*N`.
7. Add a small smooth jitter after resampling.

In this workspace the external SSD path was not writable, so the run fell
back to the local results folder:

`glacier-code/particleda/results/run11_pseudorandom_wave/`

## Outputs produced

Main diagnostics in `results/run11_pseudorandom_wave/`:

- `prior_pair.png`
  Truth initial pseudo-random field beside the background field.
- `true_beta_obs_anim.gif`
  Heatmap of the true `β` field over time with observation markers.
- `est_beta_obs_anim.gif`
  Heatmap of the ensemble-mean estimated `β` field with the same markers.
- `crosssection_anim.gif`
  Row-wise `β(x)` animation on the sensor row.
- `crosssection_final.png`
  Final-time cross-section snapshot.
- `sensor_trail_anim.gif`
  `β(t)` at the sensor cell with all particles, truth, mean, and obs.
- `sensor_trail.png`
  Static version of the sensor trail.
- `ess_tracking.png`
  ESS history.
- `rmse_beta.png`
  Global RMSE of ensemble-mean `β` versus truth over time.
- `tracking.h5`
  Truth, ensemble mean, all particles, weights, ESS, and metadata.

## First-run behaviour

From the run we generated:

| Metric | Value |
|---|---:|
| initial β-RMSE | 299.1 |
| final β-RMSE | 242.2 |
| minimum β-RMSE during run | 234.5 |
| mean ESS | 472.0 |
| minimum ESS | 23.8 |
| maximum ESS | 901.0 |

Interpretation:

- The filter does improve the estimate overall: RMSE drops by about 19 %
  from start to finish.
- The RMSE curve is not monotone; it has a rough mid-run region before
  settling lower late in the run. That is normal for this tougher prior
  mismatch.
- ESS remains healthy on average, but there are still sharp low-ESS events,
  so the pseudo-random run is informative without being trivial.

## Why this run is useful

The old sinusoidal prior is convenient, but it is also extremely regular.
The pseudo-random wave run is a better stress test because:

- the truth is not just a translated copy of one clean mode,
- the particle background starts from a genuinely different field,
- the low-wavenumber structure is still smooth enough for the advection and
  sparse sensors to be visually interpretable.

So run11 is a better bridge between the toy double-bump case and the more
general "wrong but structured background" DA setting we care about.

## See also

- `22_lw_rpf_reference.md` — LW + ESS-gated resampling + RPF jitter
- `19_noise_pipeline_deepdive.md` — smooth Gaussian noise construction
- `20_particle_visualization.md` — how to read the trajectory GIFs and
  spaghetti-style plots
