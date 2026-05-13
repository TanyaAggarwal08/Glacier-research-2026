# Run from anywhere:
#   julia --project=test bootstrap-pf-experiments/code/benchmark_sweep.jl
ENV["GKSwstype"] = "100"   # headless GR backend for PNG output

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
const SWEEP_DIR = joinpath(REPO_ROOT, "bootstrap-pf-experiments", "sweep")
mkpath(SWEEP_DIR)
cd(REPO_ROOT)   # so the YAML's "inputs/stationsW1.txt" resolves correctly when used elsewhere
@info "Working directory: $(pwd())  |  outputs → $SWEEP_DIR"

using ParticleDA
using ParticleDA: BootstrapFilter, MeanAndVarSummaryStat, FilterParameters
include(joinpath(REPO_ROOT, "test", "models", "llw2d.jl"))
using .LLW2d
using HDF5, Statistics, Random, Plots, Printf

const RESULTS_CSV = joinpath(SWEEP_DIR, "benchmark_sweep_results.csv")
const N_TIME_STEP = 100

const BASE_LLW2D = Dict{String,Any}(
    "x_length" => 200.0e3, "y_length" => 200.0e3,
    "nx" => 51, "ny" => 51,
    "station_filename" => joinpath(REPO_ROOT, "inputs", "stationsW1.txt"),
    "obs_noise_std" => [0.01],
    "nu" => 2.5, "lambda" => 5.0e3, "sigma" => [0.1, 10.0, 10.0],
    "nu_initial_state" => 2.5, "lambda_initial_state" => 5.0e3,
    "sigma_initial_state" => 0.001,
    "n_integration_step" => 10, "time_step" => 5.0,
    "peak_height" => 30.0, "peak_position" => [1.0e4, 1.0e4],
    "use_peak_initial_state_mean" => true,
)

tkey(n::Integer) = "t" * lpad(string(n), 4, '0')

# ── Build a per-run llw2d Dict with overrides ─────────────────────────────────
function build_model_dict(sigma_obs::Float64, n_stations::Int)
    llw = deepcopy(BASE_LLW2D)
    llw["obs_noise_std"] = [sigma_obs]
    if n_stations != 15
        delete!(llw, "station_filename")
        if n_stations == 5
            llw["n_stations_x"] = 5
            llw["n_stations_y"] = 1
            llw["station_distance_x"] = 40.0e3
            llw["station_boundary_x"] = 20.0e3
            llw["station_distance_y"] = 0.0
            llw["station_boundary_y"] = 100.0e3
        elseif n_stations == 1
            llw["n_stations_x"] = 1
            llw["n_stations_y"] = 1
            llw["station_boundary_x"] = 100.0e3
            llw["station_boundary_y"] = 100.0e3
        end
    end
    return Dict{String,Any}("llw2d" => llw)
end

# ── Simulate the truth trajectory + observations from a model_dict ────────────
function simulate_truth_and_obs(model_dict::Dict, n_steps::Int)
    rng = Random.MersenneTwister(123)
    model = LLW2d.init(model_dict, 1)
    state = Vector{Float64}(undef, ParticleDA.get_state_dimension(model))
    obs_dim = ParticleDA.get_observation_dimension(model)
    obs_seq = Matrix{Float64}(undef, obs_dim, n_steps)
    nx = model_dict["llw2d"]["nx"]
    ny = model_dict["llw2d"]["ny"]
    n_grid = nx * ny
    truth_heights = Vector{Matrix{Float64}}(undef, n_steps)

    ParticleDA.sample_initial_state!(state, model, rng)
    for t in 1:n_steps
        ParticleDA.update_state_deterministic!(state, model, t)
        ParticleDA.update_state_stochastic!(state, model, rng)
        ParticleDA.sample_observation_given_state!(view(obs_seq, :, t), state, model, rng)
        truth_heights[t] = reshape(state[1:n_grid], nx, ny)
    end
    return obs_seq, truth_heights
end

# ── Run one filter configuration, return metrics ──────────────────────────────
function run_one(label::String, nprt::Int, sigma_obs::Float64, n_stations::Int)
    println("\n=== Run $label  (N=$nprt, σ_obs=$sigma_obs, n_stations=$n_stations) ===")
    model_dict = build_model_dict(sigma_obs, n_stations)

    obs_seq, truth_heights = simulate_truth_and_obs(model_dict, N_TIME_STEP)

    output_file = joinpath(SWEEP_DIR, "sweep_$label.h5")
    isfile(output_file) && rm(output_file)
    fp = FilterParameters(; nprt=nprt, verbose=true, seed=42, output_filename=output_file)

    t0 = time()
    ParticleDA.run_particle_filter(LLW2d.init, fp, model_dict, obs_seq,
                                   BootstrapFilter, MeanAndVarSummaryStat)
    wallclock = time() - t0

    ess_hist = zeros(N_TIME_STEP)
    rmse_hist = zeros(N_TIME_STEP)
    max_w_final = 0.0
    h5open(output_file, "r") do f
        for t in 1:N_TIME_STEP
            w = read(f["weights"][tkey(t)])
            w_n = w ./ sum(w)
            ess_hist[t] = 1.0 / sum(w_n .^ 2)
            mean_h = read(f["state_avg"][tkey(t)]["height"])
            rmse_hist[t] = sqrt(mean((mean_h .- truth_heights[t]) .^ 2))
        end
        w = read(f["weights"][tkey(N_TIME_STEP)])
        w_n = w ./ sum(w)
        max_w_final = maximum(w_n)
    end

    r = (label=label, nprt=nprt, sigma_obs=sigma_obs, n_stations=n_stations,
         ess_hist=ess_hist, rmse_hist=rmse_hist,
         min_ess=minimum(ess_hist), mean_ess=mean(ess_hist), final_ess=ess_hist[end],
         max_w_final=max_w_final,
         mean_rmse=mean(rmse_hist), final_rmse=rmse_hist[end],
         wallclock=wallclock)

    @printf("  → min_ess=%.2f  mean_ess=%.2f  max_w=%.4f  mean_rmse=%.4f  time=%.1fs\n",
            r.min_ess, r.mean_ess, r.max_w_final, r.mean_rmse, r.wallclock)
    return r
end

# ── Plot all three sweep axes (works on partial results) ─────────────────────
function plot_sweep(results)
    isempty(results) && return

    n_runs   = sort(filter(r -> r.sigma_obs == 0.01 && r.n_stations == 15, results), by = r -> r.nprt)
    sig_runs = sort(filter(r -> r.nprt == 500 && r.n_stations == 15, results), by = r -> r.sigma_obs)
    s_runs   = sort(filter(r -> r.nprt == 500 && r.sigma_obs == 0.01, results), by = r -> r.n_stations)

    # ── ESS curves
    p_n = plot(; title="N-sweep  (σ_obs=0.01, S=15)", xlabel="Timestep", ylabel="ESS", legend=:right)
    for r in n_runs;   plot!(p_n,   1:length(r.ess_hist), r.ess_hist; label="N=$(r.nprt)", linewidth=2) end
    p_sig = plot(; title="σ_obs-sweep  (N=500, S=15)",  xlabel="Timestep", ylabel="ESS", legend=:right)
    for r in sig_runs; plot!(p_sig, 1:length(r.ess_hist), r.ess_hist; label="σ=$(r.sigma_obs)", linewidth=2) end
    p_s = plot(; title="n_stations-sweep  (N=500, σ_obs=0.01)", xlabel="Timestep", ylabel="ESS", legend=:right)
    for r in s_runs;   plot!(p_s,   1:length(r.ess_hist), r.ess_hist; label="S=$(r.n_stations)", linewidth=2) end
    fig_ess = plot(p_n, p_sig, p_s; layout=(3,1), size=(1000, 1100),
                   plot_title="Bootstrap PF — ESS across sweep axes")
    savefig(fig_ess, joinpath(SWEEP_DIR, "benchmark_sweep_ess.png"))

    # ── RMSE curves
    p_n_r = plot(; title="N-sweep  RMSE", xlabel="Timestep", ylabel="RMSE (m)", legend=:right)
    for r in n_runs;   plot!(p_n_r,   1:length(r.rmse_hist), r.rmse_hist; label="N=$(r.nprt)", linewidth=2) end
    p_sig_r = plot(; title="σ_obs-sweep  RMSE", xlabel="Timestep", ylabel="RMSE (m)", legend=:right)
    for r in sig_runs; plot!(p_sig_r, 1:length(r.rmse_hist), r.rmse_hist; label="σ=$(r.sigma_obs)", linewidth=2) end
    p_s_r = plot(; title="n_stations-sweep  RMSE", xlabel="Timestep", ylabel="RMSE (m)", legend=:right)
    for r in s_runs;   plot!(p_s_r,   1:length(r.rmse_hist), r.rmse_hist; label="S=$(r.n_stations)", linewidth=2) end
    fig_rmse = plot(p_n_r, p_sig_r, p_s_r; layout=(3,1), size=(1000, 1100),
                    plot_title="Bootstrap PF — RMSE across sweep axes")
    savefig(fig_rmse, joinpath(SWEEP_DIR, "benchmark_sweep_rmse.png"))

    # ── Summary bar chart
    labels = [r.label for r in results]
    min_pct  = [r.min_ess  / r.nprt * 100 for r in results]
    mean_pct = [r.mean_ess / r.nprt * 100 for r in results]
    p_bar = bar(labels, mean_pct; label="mean ESS  (% of N)",
                ylabel="ESS as % of N", rotation=45, legend=:topright,
                size=(1100, 600), bar_width=0.7, color=:lightblue)
    bar!(p_bar, labels, min_pct; label="min ESS  (% of N)", bar_width=0.5, color=:steelblue)
    hline!(p_bar, [10.0]; linestyle=:dash, color=:red, label="10% threshold")
    savefig(p_bar, joinpath(SWEEP_DIR, "benchmark_sweep_summary.png"))
end

function write_csv_row(io, r)
    @printf(io, "%s,%d,%.4f,%d,%.4f,%.4f,%.4f,%.6f,%.4f,%.4f,%.1f\n",
            r.label, r.nprt, r.sigma_obs, r.n_stations,
            r.min_ess, r.mean_ess, r.final_ess,
            r.max_w_final, r.mean_rmse, r.final_rmse, r.wallclock)
    flush(io)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
csv_io = open(RESULTS_CSV, "w")
println(csv_io, "label,nprt,sigma_obs,n_stations,min_ess,mean_ess,final_ess,max_w_final,mean_rmse,final_rmse,wallclock_s")
flush(csv_io)

# ordered fast→slow so partial results are durable
configs = [
    (label="N50",    nprt=50,    sigma_obs=0.01, n_stations=15),
    (label="N200",   nprt=200,   sigma_obs=0.01, n_stations=15),
    (label="sig0.1", nprt=500,   sigma_obs=0.1,  n_stations=15),
    (label="sig1.0", nprt=500,   sigma_obs=1.0,  n_stations=15),
    (label="sig5.0", nprt=500,   sigma_obs=5.0,  n_stations=15),
    (label="S5",     nprt=500,   sigma_obs=0.01, n_stations=5),
    (label="S1",     nprt=500,   sigma_obs=0.01, n_stations=1),
    (label="N500",   nprt=500,   sigma_obs=0.01, n_stations=15),
    (label="N1000",  nprt=1000,  sigma_obs=0.01, n_stations=15),
    (label="N5000",  nprt=5000,  sigma_obs=0.01, n_stations=15),
    (label="N10000", nprt=10000, sigma_obs=0.01, n_stations=15),
]

results = NamedTuple[]
for (i, cfg) in enumerate(configs)
    println("\n[$i / $(length(configs))]")
    try
        r = run_one(cfg.label, cfg.nprt, cfg.sigma_obs, cfg.n_stations)
        push!(results, r)
        write_csv_row(csv_io, r)
        try
            plot_sweep(results)
            println("  plots refreshed.")
        catch e
            @warn "plot_sweep failed: $e"
        end
    catch e
        @warn "run $(cfg.label) failed: $e"
    end
end
close(csv_io)

# Final summary
println("\n========== FINAL SUMMARY ==========")
@printf("%-8s %6s %8s %3s | %8s %8s %8s | %10s %10s %10s\n",
        "label","N","σ_obs","S","min_ess","mean_ess","max_w",
        "mean_rmse","final_rmse","wallclock")
for r in results
    @printf("%-8s %6d %8.3f %3d | %8.2f %8.2f %8.4f | %10.4f %10.4f %10.1f\n",
            r.label, r.nprt, r.sigma_obs, r.n_stations,
            r.min_ess, r.mean_ess, r.max_w_final,
            r.mean_rmse, r.final_rmse, r.wallclock)
end

if !isempty(results)
    best_abs   = results[argmax([r.min_ess        for r in results])]
    best_ratio = results[argmax([r.min_ess/r.nprt for r in results])]
    println("\n→ Highest absolute min_ess:  $(best_abs.label) → min_ess=$(round(best_abs.min_ess, digits=2))")
    println("→ Highest min_ess / N:       $(best_ratio.label) → $(round(100*best_ratio.min_ess/best_ratio.nprt, digits=2))%")
end
