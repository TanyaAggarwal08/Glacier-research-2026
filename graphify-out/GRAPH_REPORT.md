# Graph Report - ParticleDA.jl  (2026-05-13)

## Corpus Check
- 45 files · ~1,424,824 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 365 nodes · 473 edges · 28 communities (22 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f056a8ce`
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
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]

## God Nodes (most connected - your core abstractions)
1. `LLW2d` - 18 edges
2. `Bootstrap Particle Filter on LLW2d Tsunami — Findings & Plan` - 18 edges
3. `ParticleDA` - 15 edges
4. `Random` - 15 edges
5. `ParticleDA` - 14 edges
6. `HDF5` - 12 edges
7. `Kalman` - 12 edges
8. `Why was ParticleDA so slow? Did we actually use MPI?` - 12 edges
9. `Statistics` - 11 edges
10. `LinearAlgebra` - 10 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (28 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (23): Aqua, BenchmarkTools, ChunkSplitters, Documenter, HDF5, Logging, LinearGaussian, init() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (36): code:block1 (add ParticleDA.jl), code:@docs (ParticleDA.get_observation_mean_given_state!), code:@docs (ParticleDA.run_unit_tests_for_generic_model_interface), code:@docs (ParticleDA.get_state_indices_correlated_to_observations), code:yaml (filter:), code:@docs (ParticleDA.FilterParameters), code:julia (# Load ParticleDA), code:@docs (LLW2d.LLW2dModelParameters) (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (17): Distributions, animate_data(), plot_data(), FillArrays, run_simple_nonlinear_advection(), wrap(), InteractiveUtils, LinearAlgebra (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (28): Bootstrap Particle Filter on LLW2d Tsunami — Findings & Plan, code:block1 (pressure = n_obs × (signal_std / σ_obs)²        per timestep), code:block4 (internal_dt ≤ dx / wave_speed), code:julia (dt = model.parameters.time_step / model.parameters.n_integra), code:block6 (total_compute ≈ N_particles × n_time_step × n_integration_st), FAQ — What is `n_integration_step` and why does it matter for runtime?, Files produced by the sweep (still in repo), Finding 1 — Number of particles N: **DOESN'T HELP** (+20 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (25): 1. Inheritance from LLW2d isn't optional — it's the killer feature, 2. The forward-model code is the same in both libraries, 3. Ice-flow forward model = expensive per particle = MPI matters, Action items, Caveats that don't go away, code:bash (cd ~/Desktop/particleDA/ParticleDA.jl), code:bash (cp -r ParticleDA.jl/test/models/llw2d.jl YourProject.jl/src/), code:bash (julia --project=test your_glacier_benchmark.jl) (+17 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (21): B.1 — Inflate σ_obs, B.2 — Widen σ_init (if B.1 didn't fully fix it), B.3 — Reduce n_obs (last resort before going to bigger N), B.4 — Only after B.1–B.3: try larger N, code:block1 (pressure  =  n_obs × (σ_signal / σ_obs)²       per timestep), Context, Decisions to make first, Files / location (+13 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (15): code:bash (julia --project=test -e 'using MPI; MPI.Init(); println("ran), code:bash (# 4 MPI ranks on this machine), code:yaml (filter:), How to actually invoke MPI, How to make single-rank ParticleDA faster (if you ever need to), How we verified, Implication for the glacier project, One-line summary for the notebook (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.25
Nodes (12): construct_dense_covariance_matrix(), get_covariance_initial_state(), get_covariance_observation_noise(), get_covariance_observation_observation_given_previous_state(), get_covariance_state_noise(), get_covariance_state_observation_given_previous_state(), get_initial_state_mean(), get_state_indices_correlated_to_observations() (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (7): DelimitedFiles, GaussianRandomFields, get_float_eltype(), get_grid_axes(), get_station_grid_indices(), init(), LLW2d

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (12): code:julia (using Pkg, ParticleDA), code:julia (using PkgBenchmark, ParticleDA), code:block3 (benchmarkpkg(ParticleDA, BenchmarkConfig(; env = Dict("JULIA), code:julia (Pkg.activate()), code:julia (]activate), code:julia (using Pkg, ParticleDA), code:julia (cd(pkgdir(ParticleDA))), code:julia (using PkgBenchmark, BenchmarkCI) (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (12): Caveat, code:block1 (pressure = n_obs × (σ_signal / σ_obs)²), Concrete next step for the glacier project, Good news, LowLevelParticleFilters — Np=10000 ESS Observations, Practical recommendations, TL;DR for the notebook, What I'm logging into the action-plan file (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (11): 1. RMSE starts near zero because the experiment is rigged that way, 2. The truth itself is a random process — it's not a fixed target, 3. Process noise accumulates roughly like √t, 4. The collapsed runs (almost all of them) are one random walk vs another, 5. Even the healthy run (sig5.0) can't drive RMSE down, 6. So would RMSE ever decrease? Yes, in two regimes, 7. Bullet form for the write-down, 8. Implication for the glacier model (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.18
Nodes (10): `benchmark_plot.jl` (~30 s), `benchmark_run.jl` (~5 min single-rank), `benchmark_sweep.jl` (~2–3 hours), Bootstrap-PF Experiments on LLW2d, code:block1 (bootstrap-pf-experiments/), code:bash (# 1. Reproduce the paper's wave-propagation showcase figure ), Folder layout, How to run (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.4
Nodes (9): AbstractKalmanFilter, Kalman, KalmanFilter, lmult_by_observation_matrix!(), lmult_by_state_transition_matrix!(), MatrixFreeKalmanFilter, pre_and_postmultiply_by_observation_matrix!(), pre_and_postmultiply_by_state_transition_matrix!() (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.2
Nodes (10): code:block7 (N = 500, σ_obs = 0.1, n_stations = 15  →  mean ESS = 2.12  →), code:yaml (filter:), ⚠ Corrections — what I got wrong vs the actual paper (Giles et al. 2024 §6.1), Does this invalidate the sweep findings?, Effects of dt on the sweep, in principle, Mismatch 1: time_step, Mismatch 2: observation noise σ_obs, One-line summary (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.39
Nodes (8): BootstrapFilter, init_filter(), init_offline_matrices(), init_online_matrices(), OptimalFilter, ParticleFilter, sample_proposal_and_compute_log_weights!(), update_states_given_observations!()

### Community 16 - "Community 16"
Cohesion: 0.36
Nodes (5): build_model_dict(), run_one(), simulate_truth_and_obs(), tkey(), Printf

### Community 18 - "Community 18"
Cohesion: 0.43
Nodes (4): check_covariance_function(), check_cross_covariance_function(), check_mean_function(), run_tests_for_optimal_proposal_model_interface()

### Community 20 - "Community 20"
Cohesion: 0.29
Nodes (6): code:block1 (add ParticleDA), code:block2 (dev ParticleDA), Documentation, Installation, License, ParticleDA

### Community 22 - "Community 22"
Cohesion: 0.5
Nodes (3): Alex May21, Files, List of tracked input files

## Knowledge Gaps
- **154 isolated node(s):** `BenchmarkTools`, `StableRNGs`, `Aqua`, `GaussianRandomFields`, `DelimitedFiles` (+149 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LLW2d` connect `Community 8` to `Community 0`, `Community 2`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `ParticleDA` connect `Community 0` to `Community 8`, `Community 16`, `Community 2`, `Community 13`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `Bootstrap Particle Filter on LLW2d Tsunami — Findings & Plan` connect `Community 3` to `Community 11`, `Community 14`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **What connects `BenchmarkTools`, `StableRNGs`, `Aqua` to the rest of the system?**
  _154 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._