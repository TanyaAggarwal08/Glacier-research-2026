# Diagnostic — is σ_log = 0.20 well calibrated for the run14 log(β) operator?
#
# Measurement only: nothing here retunes or reruns the filter. Everything is
# read out of the run14 tracking.h5 that already exists.
#
# The question is whether the assumed observation noise is small relative to
# how much the *predicted observations* actually disagree across the ensemble.
# If σ_log ≳ s_obs the likelihood is nearly flat and the filter learns little;
# if σ_log ≪ s_obs one particle takes all the weight and ESS collapses.
#
# Note on which ensemble is measured: run14 stores
# `all_particles[:, :, t+1] = particles` AFTER propagation but BEFORE
# resampling, so slice t+1 is exactly the forecast ensemble the likelihood
# saw at step t. That is the right ensemble for this comparison.
#
# Run: julia --project=test glacier-code/particleda/rmse_experiments/diagnose_14_log_obs_calibration.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", "..", ".."))
cd(REPO_ROOT)

using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const RUN14 = joinpath("glacier-code", "particleda", "results",
                       "run14_pseudorandom_wave_30obs_logbeta", "tracking.h5")
const RUN12 = joinpath("glacier-code", "particleda", "results",
                       "run12_pseudorandom_wave_30obs", "tracking.h5")
const OUT_DIR = joinpath("glacier-code", "particleda", "results", "rmse_analysis")
mkpath(OUT_DIR)

const SIGMA_LOG = 0.20
const MIN_BETA  = 10.0

f14 = h5open(RUN14, "r")
sensor_idx = read(f14["sensor_indices"])
ess        = read(f14["weights"]["ess"])
nx = read(attributes(f14["parameters"])["nx"])
ny = read(attributes(f14["parameters"])["ny"])
NPRT = read(attributes(f14["parameters"])["nprt"])
T    = read(attributes(f14["parameters"])["T"])

n_obs = length(sensor_idx)
println("run14: nx=$nx ny=$ny N=$NPRT T=$T  n_obs=$n_obs  σ_log=$SIGMA_LOG\n")

# ── 1. Observation-space spread s_obs ───────────────────────────────────
# Read one (ny, nx, NPRT) forecast slice at a time — the full array is 1.3 GB.
function s_obs_at(t_model::Int)
    slice = f14["particles_all"]["beta"][:, :, :, t_model + 1]   # (ny, nx, NPRT)
    flat  = reshape(slice, ny * nx, NPRT)                        # column-major flat
    h     = log.(max.(flat[sensor_idx, :], MIN_BETA))            # H(β) per sensor,particle
    return vec(std(h; dims=2))                                   # std across particles
end

const TIMES = [0, 50, T]
spreads = Dict{Int, Vector{Float64}}()
for t in TIMES
    spreads[t] = s_obs_at(t)
end

println("=== 1. Per-sensor s_obs = std over particles of log(β) at sensor ===\n")
@printf("%-8s %12s %12s %12s\n", "sensor", "t=0", "t=50", "t=$T")
for k in 1:n_obs
    @printf("%-8d %12.4f %12.4f %12.4f\n", k,
            spreads[0][k], spreads[50][k], spreads[T][k])
end
println()
for t in TIMES
    s = spreads[t]
    @printf("t=%-4d  mean s_obs = %.4f   median = %.4f   min = %.4f   max = %.4f\n",
            t, mean(s), median(s), minimum(s), maximum(s))
end

# ── 2. σ_log vs s_obs ───────────────────────────────────────────────────
println("\n=== 2. Ratio σ_log / s_obs   (target band: 0.1 – 0.3) ===\n")
@printf("%-8s %12s %12s %12s %12s\n", "t", "mean s_obs", "ratio(mean)", "ratio(min s)", "ratio(max s)")
for t in TIMES
    s = spreads[t]
    @printf("%-8d %12.4f %12.2f %12.2f %12.2f\n", t, mean(s),
            SIGMA_LOG / mean(s), SIGMA_LOG / maximum(s), SIGMA_LOG / minimum(s))
end
for t in TIMES
    r = SIGMA_LOG / mean(spreads[t])
    verdict = r > 1.0   ? "TOO LOOSE — noise exceeds ensemble spread, obs near-uninformative" :
              r > 0.3   ? "loose — above the 0.1–0.3 band" :
              r < 0.1   ? "TOO TIGHT — noise far below spread, ESS-collapse risk" :
                          "IN BAND (0.1–0.3)"
    @printf("  t=%-4d ratio = %.2f  → %s\n", t, r, verdict)
end

# ── 3. Full ESS trace ───────────────────────────────────────────────────
println("\n=== 3. ESS trace ===\n")
ess_min, ess_argmin = findmin(ess)
n_below_half = count(<(NPRT / 2), ess)
@printf("  mean ESS            = %.1f\n", mean(ess))
@printf("  min  ESS            = %.1f  at timestep t = %d\n", ess_min, ess_argmin)
@printf("  median ESS          = %.1f\n", median(ess))
@printf("  steps with ESS<N/2  = %d / %d  (%.0f %%)\n",
        n_below_half, length(ess), 100 * n_below_half / length(ess))
@printf("  steps with ESS<N/10 = %d / %d\n", count(<(NPRT / 10), ess), length(ess))
@printf("  first 10 steps      = %s\n", join(round.(ess[1:10]; digits=1), ", "))

p_ess = plot(1:length(ess), ess;
             xlabel="timestep", ylabel="ESS",
             title="run14 (log β obs, σ_log = $SIGMA_LOG) — ESS trace",
             lw=2, color=:purple, label="ESS", ylim=(0, NPRT))
hline!(p_ess, [NPRT / 2]; linestyle=:dash, color=:gray, label="N/2 (resample trigger)")
hline!(p_ess, [mean(ess)]; linestyle=:dot, color=:darkgreen, label=@sprintf("mean = %.1f", mean(ess)))
scatter!(p_ess, [ess_argmin], [ess_min]; color=:red, ms=6,
         label=@sprintf("min = %.1f @ t=%d", ess_min, ess_argmin))
savefig(p_ess, joinpath(OUT_DIR, "ess_trace_run14_logbeta.png"))
println("\n  Saved → $(joinpath(OUT_DIR, "ess_trace_run14_logbeta.png"))")

# s_obs evolution plot
p_s = plot(; xlabel="timestep", ylabel="s_obs (std of log β across particles)",
           title="run14 — observation-space spread vs σ_log", legend=:topright)
for t in TIMES
    scatter!(p_s, fill(t, n_obs), spreads[t]; label=false, color=:steelblue,
             alpha=0.5, ms=4)
end
plot!(p_s, TIMES, [mean(spreads[t]) for t in TIMES];
      label="mean s_obs across sensors", lw=2.5, color=:black, marker=:square)
hline!(p_s, [SIGMA_LOG]; linestyle=:dash, color=:firebrick, lw=2.5,
       label="σ_log = $SIGMA_LOG")
savefig(p_s, joinpath(OUT_DIR, "s_obs_vs_sigma_run14.png"))
println("  Saved → $(joinpath(OUT_DIR, "s_obs_vs_sigma_run14.png"))")

# ── 4. Confound check: is run12's truth identical to run14's? ───────────
println("\n=== 4. run12 vs run14 confound check ===\n")
f12 = h5open(RUN12, "r")
truth12 = read(f12["truth"]["beta"])
truth14 = read(f14["truth"]["beta"])
@printf("  truth fields identical      : %s  (max |Δβ| = %.3e)\n",
        truth12 == truth14, maximum(abs.(truth12 .- truth14)))
s12 = read(f12["sensor_indices"]); s14 = read(f14["sensor_indices"])
@printf("  sensor layout identical     : %s  (n_obs %d vs %d)\n",
        s12 == s14, length(s12), length(s14))
bg12 = read(f12["beta_background_prior"]); bg14 = read(f14["beta_background_prior"])
tp12 = read(f12["beta_truth_prior"]);      tp14 = read(f14["beta_truth_prior"])
@printf("  background prior identical  : %s\n", bg12 == bg14)
@printf("  truth prior identical       : %s\n", tp12 == tp14)
@printf("  N particles                 : %d vs %d\n",
        read(attributes(f12["parameters"])["nprt"]), NPRT)
@printf("  T steps                     : %d vs %d\n",
        read(attributes(f12["parameters"])["T"]), T)

# Initial ensembles are drawn from rng_pf before any obs are used, so they
# should match bit-for-bit if nothing else changed.
init12 = f12["particles_all"]["beta"][:, :, :, 1]
init14 = f14["particles_all"]["beta"][:, :, :, 1]
@printf("  initial ensemble identical  : %s  (max |Δβ| = %.3e)\n",
        init12 == init14, maximum(abs.(init12 .- init14)))

mean12 = read(f12["ensemble_mean"]["beta"])
mean14 = read(f14["ensemble_mean"]["beta"])
rmse12 = [sqrt(mean((mean12[:, :, t] .- truth12[:, :, t]).^2)) for t in 1:size(truth12, 3)]
rmse14 = [sqrt(mean((mean14[:, :, t] .- truth14[:, :, t]).^2)) for t in 1:size(truth14, 3)]
@printf("\n  RMSE(β) final : run12 = %.1f   run14 = %.1f\n", rmse12[end], rmse14[end])
@printf("  RMSE(β) mean  : run12 = %.1f   run14 = %.1f\n", mean(rmse12), mean(rmse14))
ess12 = read(f12["weights"]["ess"])
@printf("  mean ESS      : run12 = %.1f   run14 = %.1f\n", mean(ess12), mean(ess))
@printf("  min  ESS      : run12 = %.1f   run14 = %.1f\n", minimum(ess12), ess_min)

close(f12); close(f14)
println("\nDone — measurement only, nothing retuned.")
