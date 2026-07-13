# Experiment 09 (PF stage) — dump truth + ensemble-mean β trajectories.
#
# Two-stage pipeline (see design notes in Exp 09 conversation):
#   STAGE 1 (this file, run under --project=test):
#     Run bootstrap PF on the linear advection model. Save the full truth-β
#     trajectory and the full ensemble-mean-β trajectory to HDF5. Also save
#     β-space RMSE per time step and metadata.
#   STAGE 2 (experiment_09_velocity_rmse_post.jl, run under default env
#            because it needs WAVI):
#     Load HDF5, call IceFlow.velocity per timestep on truth and mean β,
#     compute velocity RMSE, plot β-RMSE (continuous) and velocity-RMSE
#     (discrete points) on a single figure.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/experiment_09_velocity_rmse_pf.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

include(joinpath(@__DIR__, "_helpers.jl"))
using HDF5, Printf

const OUT_H5 = joinpath(RMSE_OUT, "exp09_beta_trajectories.h5")
const NOTES  = "Two-stage velocity-RMSE pipeline; stage 1 = β trajectories dump."

base_params = Dict{String, Any}(
    "nx" => 40, "ny" => 40,
    "x_length" => 160_000.0, "y_length" => 160_000.0,
    "station_filename" => "glacier-code/particleda/stations_crosssection.txt",
    "init_std_beta" => 600.0,
    "process_std_beta" => 28.0,
    "obs_noise_std" => 0.10,
    "advection_epsilon" => 5e-4,
    "n_integration_step" => 10,
    "time_step" => 3600.0,
    "min_beta" => 10.0,
    "noise_length_scale" => 15_000.0,
    "advection_type" => "linear",
)
const NPRT = 1000
const T    = 100

println("=== Experiment 09 stage 1: PF trajectory dump ===")
t0 = time()
res = run_pf_rmse(base_params; NPRT=NPRT, T=T, return_trajectories=true)
elapsed = time() - t0
@printf("  PF done in %.1fs. Final β-RMSE=%.1f  mean ESS=%.1f\n",
        elapsed, res.rmse_global[end], mean(res.ess))

# ── Save trajectories + metadata ─────────────────────────────────────────
isfile(OUT_H5) && rm(OUT_H5)
h5open(OUT_H5, "w") do f
    f["truth_beta"]     = res.truth_states           # (nx*ny, T+1)
    f["mean_beta"]      = res.ensemble_mean          # (nx*ny, T+1)
    f["model_time_h"]   = res.model_time             # (T+1,)
    f["rmse_beta"]      = res.rmse_global            # (T+1,)
    f["ess"]            = res.ess                    # (T,)
    f["sensor_idx"]     = res.sensor_idx
    g = create_group(f, "parameters")
    attributes(g)["NPRT"]      = NPRT
    attributes(g)["T"]         = T
    attributes(g)["nx"]        = base_params["nx"]
    attributes(g)["ny"]        = base_params["ny"]
    attributes(g)["x_length"]  = base_params["x_length"]
    attributes(g)["y_length"]  = base_params["y_length"]
    attributes(g)["time_step"] = base_params["time_step"]
    attributes(g)["advection_type"]      = base_params["advection_type"]
    attributes(g)["init_std_beta"]       = base_params["init_std_beta"]
    attributes(g)["process_std_beta"]    = base_params["process_std_beta"]
    attributes(g)["obs_noise_std"]       = base_params["obs_noise_std"]
    attributes(g)["noise_length_scale"]  = base_params["noise_length_scale"]
end
println("Saved trajectories → $OUT_H5")

# ── Log a row so it appears in the experiment log ───────────────────────
runtime_note = @sprintf("%s (wall %.1fs)", NOTES, elapsed)
log_experiment!(;
    figure          = "exp09_velocity_rmse.png",       # produced in stage 2
    model_type      = "Linear (velocity-RMSE pipeline)",
    sigma_init      = base_params["init_std_beta"],
    sigma_proc      = base_params["process_std_beta"],
    length_scale_km = base_params["noise_length_scale"] / 1000,
    particles       = NPRT,
    n_obs           = length(res.sensor_idx),
    obs_interval_s  = base_params["time_step"],
    sigma_obs       = base_params["obs_noise_std"],
    T_steps         = T,
    notes           = runtime_note,
)
println("Logged → $(LOG_PATH)")
