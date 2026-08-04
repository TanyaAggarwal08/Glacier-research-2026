# Graph Report - ParticleDA.jl  (2026-08-03)

## Corpus Check
- 128 files · ~5,989,121 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1121 nodes · 1420 edges · 81 communities (65 shown, 16 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `348ba17b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]

## God Nodes (most connected - your core abstractions)
1. `HDF5` - 46 edges
2. `Statistics` - 46 edges
3. `Plots` - 46 edges
4. `Printf` - 43 edges
5. `Random` - 41 edges
6. `ParticleDA` - 37 edges
7. `LinearAlgebra` - 32 edges
8. `PDMats` - 25 edges
9. `IceExpDyn` - 19 edges
10. `LLW2d` - 18 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (81 total, 16 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (10): HDF5, CaseResult, load_case(), draw_section(), row_slice(), load_seed(), SeedResult, Plots (+2 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (36): code:block1 (add ParticleDA.jl), code:@docs (ParticleDA.get_observation_mean_given_state!), code:@docs (ParticleDA.run_unit_tests_for_generic_model_interface), code:@docs (ParticleDA.get_state_indices_correlated_to_observations), code:yaml (filter:), code:@docs (ParticleDA.FilterParameters), code:julia (# Load ParticleDA), code:@docs (LLW2d.LLW2dModelParameters) (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (16): Aqua, BenchmarkTools, ChunkSplitters, Documenter, Logging, MPI, ParticleDA, Serialization (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (34): 1. Was LowLevel wrong?, 2.1 The big picture, 2.2 Inside `update_state_deterministic!`, 2.3 Why dt matters — what would happen at other values, 2.4 The CFL condition in plain words, 2. What does `dt = 400 s` actually do in the code?, 3.1 Full list of our current canonical config, 3.2 LowLevel reference (`particlefilteringwithiceflow.jl`) (+26 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (33): 1. Setup recap (so the rest makes sense), 2.1 What the code does, 2.2 What this means in numbers, 2.3 What happens after t = 0, 2.4 Same story for tsunami, 2. Q1 — Do particles start at the truth, or are they scattered?, 3.1 Noise is added in ux units, not β units, 3.2 The β plot exaggerates the noise (+25 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (34): 1. Parameters, 2.1 Jitter for numerical positive-definiteness, 2. The spatial correlation kernel, 3.1 Why this lets us draw a smooth field, 3.2 Empirical check, 3. Cholesky factor — the heart of the smooth sampler, 4.1 Where this lands the initial particles, 4. Stage 1: initial-state perturbation `δβ₀` (+26 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (6): LinearGaussian, init(), Lorenz63, OrdinaryDiffEqTsit5, PDMats, Random

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (31): 10. Where this shows up, 11. Glossary additions, 1. The puzzle, 2. The discrete scheme, 3. Modified-equation analysis — what PDE does (2) actually solve?, 4. Why amplitude decays exponentially, 5. Plug in the numbers, 6. Why the periodic BC is innocent (+23 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (31): 1. (Confounder) Different problems, 2. (Confounder) 50× particle count, 3. (Confounder) Resampling cadence and what ESS measures, 4. (Confounder) σ_init in observation-space units, 5. (Real degeneracy) Posterior variance collapses to 1.0 in two steps, 6. (Real degeneracy) Pressure budget, A. Filter-library sanity check on the LowLevel toy, Aside — why we didn't just port the LowLevel toy's parameters (+23 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (25): 1. Inheritance from LLW2d isn't optional — it's the killer feature, 2. The forward-model code is the same in both libraries, 3. Ice-flow forward model = expensive per particle = MPI matters, Action items, Caveats that don't go away, code:bash (cd ~/Desktop/particleDA/ParticleDA.jl), code:bash (cp -r ParticleDA.jl/test/models/llw2d.jl YourProject.jl/src/), code:bash (julia --project=test your_glacier_benchmark.jl) (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (25): 1. File map — what lives where, 2. What to run, depending on what you changed, 3. The two-step pattern, always, 4. Naming convention for results, 5. Common gotchas, 6. Quick reference — the three commands you'll use 90% of the time, A. You edited `glacier.yaml` (parameters only — no code change), B. You edited a non-default YAML (e.g. `glacier_random.yaml` for scattered sensors) (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (24): 1.1 The workflow in plain English, 1.2 The exact tsunami numbers, 1.3 *Why* tsunami needs 10 sub-steps, 1. What the tsunami model actually does (verified from source), 2.1 The workflow, 2.2 The exact glacier numbers, 2.3 Why we only need 1 sub-step, 2. What our glacier model does (verified from source) (+16 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (22): 1. How the ablation works, 2. Headline numbers, 3. What the plots actually show, 4.1 The numerical diffusion budget, 4.2 Why this kills the truth's initial perturbation, 4.3 What this means for the with-obs run's RMSE, 4.4 So which is "right"?, 4. Why no-obs RMSE drops — the numerical-diffusion explanation (+14 more)

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (21): 1. Change parameters and re-run on your own, 2. Two commands to run a tracking experiment, 3. Other drivers (YAML-based, simpler workflow), 4. Common edits and what they do, 5. Outputs you can look at, 6. Troubleshooting, 7. Where the science context lives, A. Filter / physics parameters (+13 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (11): DelimitedFiles, GaussianRandomFields, get_float_eltype(), get_grid_axes(), get_station_grid_indices(), init(), LLW2d, _build_sensor_indices() (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (21): B.1 — Inflate σ_obs, B.2 — Widen σ_init (if B.1 didn't fully fix it), B.3 — Reduce n_obs (last resort before going to bigger N), B.4 — Only after B.1–B.3: try larger N, code:block1 (pressure  =  n_obs × (σ_signal / σ_obs)²       per timestep), Context, Decisions to make first, Files / location (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.1
Nodes (20): 1. Model time covered ≠ compute time used, 2.1 Cost per internal physics step, 2.2 Cost per stochastic step, 2.3 Total cost ratio (theory), 2. The factor-by-factor breakdown, 3.1 Slower waves → larger CFL ceiling → larger dt → fewer sub-steps, 3.2 Fewer state variables → smaller inner-loop body, 3.3 Simpler noise model (+12 more)

### Community 17 - "Community 17"
Cohesion: 0.1
Nodes (20): 1. ParticleDA.jl — module architecture, 2. glacier-code/main.jl — WAVI ice-sheet forward model (no DA yet), 3. glacier-code/particlefilteringnonlinear.jl — toy 2D advection + LowLevelPF, 4. How the three pieces will eventually connect (Phase A target), 5. Pointers (don't re-read these files; check this map first), Code Maps — ParticleDA + glacier-code, code:block1 (src/ParticleDA.jl  (module root, re-exports)), code:block2 (ParticleFilter (abstract)) (+12 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (7): Distributions, FillArrays, LinearAlgebra, LowLevelParticleFilters, Pkg, run_pf(), systematic_resample()

### Community 19 - "Community 19"
Cohesion: 0.1
Nodes (19): 1. The error term added to particles after resampling (RPF jitter), 2. Lax–Wendroff advection, 3. Resampling, 4. Parameters in use (run10_wendroff_amplitude), code:block1 (z  ~ 𝒩(0, Iₙ)              # nx·ny iid standard normals), code:block2 (K(x₁, x₂) = exp(−|x₁ − x₂|² / (2 ℓ²))   + 1e-8 · I  (PD jitt), code:julia (if SIGMA_JITTER > 0), code:block4 (β_new[j, i] = β[j, i]) (+11 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (18): 1. What actually happens in one 400-second step, 2. The three things to verify, 3. The counterintuitive bit — why more sub-steps doesn't automatically help, 4. So when *would* you want more sub-steps?, 5. What this means for our results so far, 6. The honest trade-off, 7. Direct answer to "how is the PDE updating between intervals?", 8. Empirical verification we could run (+10 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (16): 1. The custom tracking driver, 2.1 Cross-section animation — `crosssection_anim.gif`, 2.2 Spaghetti per probe cell — `spaghetti_probe_cells.png`, 2.3 Per-particle snapshot collage — `particle_snapshots.png`, 2.4 Weight evolution — `weight_evolution.png`, 2. The four visualisations, 3. How to re-run, 4. What this lets you see that the standard plots can't (+8 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (14): 1. The exact noise model in the code, 2. Why σ_obs = 0.10, 3.1 The math, 3.2 Concrete numbers at β ≈ 1000, 3.3 What about other β values?, 3.4 Statistical sanity check, 3. Why the crosses look "very far apart" — the noise amplification, 4. What the filter actually sees (+6 more)

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (14): apply_noise!(), _build_noise_factor(), _build_prior_fields(), _build_sensor_indices(), _double_bump_field(), _double_bump_truth_background(), IceExpDyn, init() (+6 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (15): 1. What changed in the code, 2. The stations file format, 3. How this run's sensors are placed, 4. Headline result — same physics, same filter health, 5. Where there *is* a difference, 6.1 To try a different layout, 6.2 Suggestions for layouts worth trying, 6. Practical workflow notes (+7 more)

### Community 25 - "Community 25"
Cohesion: 0.12
Nodes (16): 3.10 Does the filter ever see the truth?, 3.11 Where does our implementation match the paper, and where does it deviate?, 3.1 Initialisation, 3.2 Forecast / prediction step, 3.3 Observation operator, 3.4 Likelihood computation, 3.5 Weight normalisation and ESS, 3.6 Resampling (+8 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (15): code:bash (julia --project=test -e 'using MPI; MPI.Init(); println("ran), code:bash (# 4 MPI ranks on this machine), code:yaml (filter:), How to actually invoke MPI, How to make single-rank ParticleDA faster (if you ever need to), How we verified, Implication for the glacier project, One-line summary for the notebook (+7 more)

### Community 27 - "Community 27"
Cohesion: 0.12
Nodes (15): Bootstrap Particle Filter on LLW2d Tsunami — Findings & Plan, code:block1 (pressure = n_obs × (signal_std / σ_obs)²        per timestep), Files produced by the sweep (still in repo), Finding 1 — Number of particles N: **DOESN'T HELP**, Finding 2 — Observation noise σ_obs: **THIS IS THE LEVER**, Finding 3 — Number of stations n_stations: helps, but only at very low counts, Finding 4 — Counter-intuitive: looser observations gave BETTER accuracy, Finding 5 — Empirical rule of thumb (write this down) (+7 more)

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (14): 1.1 What changed in code, 1.2 Why drop the log transform?, 1.3 Risk: β going negative, 1.4 Behaviour stayed essentially identical, 1. State switched to β directly, 2.1 Why the corner cell looks flat, 2.2 Direct evidence: PF mean at non-degenerate cells, 2.3 The visual proof (+6 more)

### Community 29 - "Community 29"
Cohesion: 0.13
Nodes (14): code:text (truth field      = centre + signal_scale * normalised_random), code:text (w(x, y) = Σ a(kx,ky) sin(2πkx i / nx + phx) sin(2πky j / ny ), code:text (w_norm = w / std(w)), code:bash (julia --project=test glacier-code/particleda/run_pseudorando), First-run behaviour, How run11 is executed, Outputs produced, Parameters used in run11 (+6 more)

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (15): 2.1 The continuous PDE, 2.2 The discrete update equation, 2.3 Parameter-by-parameter, 2.4 Diffusion terms — there are none *explicitly*, 2.5 Stochastic update, 2.6 Deterministic or stochastic?, 2.7 CFL constraint, 2.8 Does the implementation match the intended physics? (+7 more)

### Community 31 - "Community 31"
Cohesion: 0.25
Nodes (12): construct_dense_covariance_matrix(), get_covariance_initial_state(), get_covariance_observation_noise(), get_covariance_observation_observation_given_previous_state(), get_covariance_state_noise(), get_covariance_state_observation_given_previous_state(), get_initial_state_mean(), get_state_indices_correlated_to_observations() (+4 more)

### Community 32 - "Community 32"
Cohesion: 0.15
Nodes (12): code:julia (using Pkg, ParticleDA), code:julia (using PkgBenchmark, ParticleDA), code:block3 (benchmarkpkg(ParticleDA, BenchmarkConfig(; env = Dict("JULIA), code:julia (Pkg.activate()), code:julia (]activate), code:julia (using Pkg, ParticleDA), code:julia (cd(pkgdir(ParticleDA))), code:julia (using PkgBenchmark, BenchmarkCI) (+4 more)

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (12): Caveat, code:block1 (pressure = n_obs × (σ_signal / σ_obs)²), Concrete next step for the glacier project, Good news, LowLevelParticleFilters — Np=10000 ESS Observations, Practical recommendations, TL;DR for the notebook, What I'm logging into the action-plan file (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.15
Nodes (12): 1.1 Smooth noise (squared-exponential covariance), 1.2 Hourly cadence, 1.3 Sensor / unsensored RMSE split, 1. What changed, 2. The result, 3. Why smooth noise fixes the audit's central puzzle, 4. Files touched, 5. Glossary additions (+4 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (12): 1. What changed vs everything before, 2. The environment problem and the standalone-dynamics port, 3. Parameters (run: `experiment_18_wavi_obs.jl 500 20 10 0.5 5e-4 full`), 4. Results — the filter recovers β from the real ice-sheet model, 5. Compute cost and parallelism, 6. Files and outputs, 7. How to reproduce, 8. Next steps (+4 more)

### Community 36 - "Community 36"
Cohesion: 0.15
Nodes (12): 1. The motivation in everyday terms, 2. The hidden bug we found while setting this up, 3. The actual experiment, 4.1 Headline numbers, 4.2 What the plots show, 4.3 The CFL warning, 4. What we found, 5. Why fewer observations gave *better* ESS (+4 more)

### Community 37 - "Community 37"
Cohesion: 0.18
Nodes (10): 1. Why redo it, 2. What changed in the config, 3. Side-by-side numbers, 4. What the envelope plots show, 5. What this means, 6. Quick answer to "what `n_integration_step` are we using?", 7. Glossary (carried over from [07_observation_cadence.md](07_observation_cadence.md)), Per-seed table (v2) (+2 more)

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (10): 1. The operator being tested, 2. Observation-space spread, s_obs, 3. σ_log versus s_obs, 4. ESS trace — the mean hides an early collapse, 5. run12 vs run14 — what is and isn't confounded, 6. The tension that has to be resolved before retuning, 7. Status, code:bash (julia --project=test glacier-code/particleda/rmse_experiment) (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (11): 1. RMSE starts near zero because the experiment is rigged that way, 2. The truth itself is a random process — it's not a fixed target, 3. Process noise accumulates roughly like √t, 4. The collapsed runs (almost all of them) are one random walk vs another, 5. Even the healthy run (sig5.0) can't drive RMSE down, 6. So would RMSE ever decrease? Yes, in two regimes, 7. Bullet form for the write-down, 8. Implication for the glacier model (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.18
Nodes (10): `benchmark_plot.jl` (~30 s), `benchmark_run.jl` (~5 min single-rank), `benchmark_sweep.jl` (~2–3 hours), Bootstrap-PF Experiments on LLW2d, code:block1 (bootstrap-pf-experiments/), code:bash (# 1. Reproduce the paper's wave-propagation showcase figure ), Folder layout, How to run (+2 more)

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (9): AbstractKalmanFilter, Kalman, KalmanFilter, lmult_by_observation_matrix!(), lmult_by_state_transition_matrix!(), MatrixFreeKalmanFilter, pre_and_postmultiply_by_observation_matrix!(), pre_and_postmultiply_by_state_transition_matrix!() (+1 more)

### Community 42 - "Community 42"
Cohesion: 0.2
Nodes (9): 1. Default canonical config (grid-aligned 16 sensors), 2. Scattered-sensor variant, 3. Just re-plot an existing archived run (no filter rerun — fastest iteration), code:block1 (add ParticleDA), code:block2 (dev ParticleDA), Documentation, Installation, License (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.2
Nodes (10): code:block7 (N = 500, σ_obs = 0.1, n_stations = 15  →  mean ESS = 2.12  →), code:yaml (filter:), ⚠ Corrections — what I got wrong vs the actual paper (Giles et al. 2024 §6.1), Does this invalidate the sweep findings?, Effects of dt on the sweep, in principle, Mismatch 1: time_step, Mismatch 2: observation noise σ_obs, One-line summary (+2 more)

### Community 44 - "Community 44"
Cohesion: 0.2
Nodes (10): code:block4 (internal_dt ≤ dx / wave_speed), code:julia (dt = model.parameters.time_step / model.parameters.n_integra), code:block6 (total_compute ≈ N_particles × n_time_step × n_integration_st), FAQ — What is `n_integration_step` and why does it matter for runtime?, How does it scale runtime?, Implications for the glacier model, Quick mental model, The short version (+2 more)

### Community 45 - "Community 45"
Cohesion: 0.39
Nodes (8): BootstrapFilter, init_filter(), init_offline_matrices(), init_online_matrices(), OptimalFilter, ParticleFilter, sample_proposal_and_compute_log_weights!(), update_states_given_observations!()

### Community 46 - "Community 46"
Cohesion: 0.22
Nodes (8): Interpretation, Multi-seed robustness check (5 seeds at run-03 config), Per-seed summary, Plots, Spread between seeds is small enough to do science with, What's robust (i.e. not a fluke), What still happens but is not a problem, What this clears us to do next

### Community 47 - "Community 47"
Cohesion: 0.22
Nodes (8): First-run numbers (linear advection, run10-ish defaults, T=100h), Ice-flow model integration — β → velocity RMSE pipeline, Interpretation, Key API adaptation, See also, Two things worth remembering, What this unlocks, What was built

### Community 48 - "Community 48"
Cohesion: 0.22
Nodes (8): 1. What changed vs Exp 18 (seed 1), 2. Results — the recovery reproduces, 3. Figures, 4. Files and reproduction, 5. Next steps, code:bash (# default env (has WAVI); 10 threads. ARGS: N T K σ ε mode S), See also, WAVI observation operator — second seed (replicate of Exp 18)

### Community 49 - "Community 49"
Cohesion: 0.22
Nodes (9): 1.1 What the model state actually contains, 1.2 How β is represented, 1.3 The prior equation for β, 1.4 Why the truth doesn't look smooth even though the prior does, 1.5 How the truth trajectory is generated, 1. State representation, code:block1 (β_prior(x, y) = 1000 + 500 · sin(ω · x) · sin(ω · y),     ω ), code:block2 (β_truth(x, y, t) = β_prior(x, y) + initial_noise(x, y) + adv) (+1 more)

### Community 50 - "Community 50"
Cohesion: 0.29
Nodes (7): animate_data(), plot_data(), InteractiveUtils, Markdown, PlutoUI, Unitful, UnitfulRecipes

### Community 51 - "Community 51"
Cohesion: 0.29
Nodes (4): IceFlow, velocity(), velocity_flat(), WAVI

### Community 53 - "Community 53"
Cohesion: 0.25
Nodes (8): 4.1 How synthetic observations are generated, 4.2 Sensor locations, 4.3 Observation cadence, 4.4 Observation noise, 4.5 Sparse vs dense observations, 4.6 What "no observations" actually means in our ablation, 4. Observation system, code:block14 (y_t = h(x_t) + v_t,   v_t ~ N(0, σ_obs² · I_16))

### Community 54 - "Community 54"
Cohesion: 0.43
Nodes (4): logdens!(), logspeed_at_sensors(), quiet(), wavi_logspeed()

### Community 56 - "Community 56"
Cohesion: 0.43
Nodes (4): build_model_dict(), run_one(), simulate_truth_and_obs(), tkey()

### Community 57 - "Community 57"
Cohesion: 0.43
Nodes (4): check_covariance_function(), check_cross_covariance_function(), check_mean_function(), run_tests_for_optimal_proposal_model_interface()

### Community 59 - "Community 59"
Cohesion: 0.29
Nodes (6): 7.1 Where we differ from "best practice", 7.2 Where we *deviate* in a way that might look like a flaw but isn't, 7.3 Conceptual flaws or simplifications worth flagging, 7. Comparison against ParticleDA paper / standard PF theory, Files referenced, Full audit — glacier particle filter end-to-end

### Community 63 - "Community 63"
Cohesion: 0.33
Nodes (6): 5.1 What's compared, where, when, 5.2 Timing, 5.3 Are we comparing the right variables?, 5.4 Possible RMSE bugs to rule out, 5. RMSE calculation, code:block15 (rmse_beta[t] = sqrt(mean((beta_est[t] .- beta_true[t]).^2)))

### Community 64 - "Community 64"
Cohesion: 0.33
Nodes (6): 6.1 Diagnostic 1 — Truth-vs-prior std over time, 6.2 Diagnostic 2 — RMSE split: sensor cells vs unsensored cells, 6.3 Why the with-obs unsensored RMSE is *higher* than no-obs, 6.4 The no-obs RMSE *also* drops (from 300 to 84) — why?, 6.5 Checklist for everything I ruled out, 6. Critical consistency checks — diagnosing the RMSE puzzle

### Community 65 - "Community 65"
Cohesion: 0.33
Nodes (6): 8.1 Is the current behaviour physically and statistically reasonable?, 8.2 Most likely root cause of the unexpected RMSE trend, 8.3 Prioritised debugging checklist, 8.4 What this audit confirms about the code, 8.5 What it confirms about our understanding so far, 8. Final diagnostic assessment

### Community 67 - "Community 67"
Cohesion: 0.6
Nodes (4): _ensure_log_initialised(), log_experiment!(), run_pf_rmse(), _systematic_resample()

### Community 72 - "Community 72"
Cohesion: 0.5
Nodes (3): Alex May21, Files, List of tracked input files

### Community 74 - "Community 74"
Cohesion: 0.67
Nodes (3): How the sweep was implemented (for later reference), Numerical results table, Sweep grid (final, 11 runs)

## Knowledge Gaps
- **539 isolated node(s):** `BenchmarkTools`, `StableRNGs`, `Aqua`, `GaussianRandomFields`, `OrdinaryDiffEqTsit5` (+534 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Plots` connect `Community 0` to `Community 66`, `Community 68`, `Community 69`, `Community 6`, `Community 71`, `Community 70`, `Community 73`, `Community 50`, `Community 51`, `Community 18`, `Community 54`, `Community 55`, `Community 56`, `Community 60`, `Community 61`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `HDF5` connect `Community 0` to `Community 2`, `Community 68`, `Community 6`, `Community 71`, `Community 70`, `Community 14`, `Community 18`, `Community 50`, `Community 54`, `Community 55`, `Community 56`, `Community 60`, `Community 61`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `IceExpDyn` connect `Community 23` to `Community 0`, `Community 18`, `Community 6`, `Community 14`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `BenchmarkTools`, `StableRNGs`, `Aqua` to the rest of the system?**
  _539 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._