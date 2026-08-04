# Exp 18 — regenerate the field-comparison figures as 3-panel "before → after"
# plots that add the INITIAL GUESS alongside the final truth and final mean.
#
#   beta_field_truth_vs_mean.png     : initial guess β | truth β (final) | mean β (final)
#   velocity_field_truth_vs_mean.png : initial guess speed | truth speed (final) | mean speed (final)
#
# The β panels come straight from the saved trajectory in tracking.h5 (the
# initial guess is the t=0 ensemble mean = the prior mean before any
# assimilation). The velocity fields were NOT saved (only the 16-sensor
# log-speeds were), so this script re-runs WAVI on exactly three β states:
# the initial-guess mean, the final truth, and the final mean — 3 solves, ~4 s.
#
# RUN UNDER THE DEFAULT ENV (has WAVI), no threads needed:
#   julia glacier-code/particleda/plot_exp18_field_evolution.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Statistics, LinearAlgebra, Printf, HDF5
ENV["GKSwstype"] = "100"
using Plots, Plots.PlotMeasures
include(joinpath(@__DIR__, "ice_flow.jl")); using .IceFlow

const RUN_TAG = length(ARGS) >= 1 ? ARGS[1] : "experiment_18_wavi_obs"
const OUT = joinpath("glacier-code", "particleda", "results", RUN_TAG)
mkpath(OUT)

# ── locate the saved trajectory (SSD preferred, checkpoint fallback) ──────
const SSD = joinpath("/Volumes/ZX20/USRA 2026", RUN_TAG)
h5path = let cands = [joinpath(SSD, "tracking.h5"), joinpath(SSD, "checkpoint.h5")]
    i = findfirst(isfile, cands)
    i === nothing && error("No tracking.h5/checkpoint.h5 found under $SSD — is the SSD mounted?")
    cands[i]
end
println("Reading $h5path")

truth_beta, mean_beta, sensors, nx, ny = h5open(h5path, "r") do f
    tb = read(f["truth/beta"])          # (ny, nx, T+1)  Julia column-major layout
    mb = read(f["ensemble_mean/beta"])  # (ny, nx, T+1)
    si = Int.(read(f["sensor_indices"]))
    ny_, nx_ = size(tb, 1), size(tb, 2)
    (tb, mb, si, nx_, ny_)
end
T = size(truth_beta, 3) - 1
@printf("grid %d×%d, T=%d, %d sensors\n", nx, ny, T, length(sensors))

# ── the three β fields (initial guess → final truth → final mean) ─────────
init_beta  = mean_beta[:, :, 1]     # t=0 ensemble mean = initial guessed field
truth_bF   = truth_beta[:, :, end]  # final truth
mean_bF    = mean_beta[:, :, end]   # final mean

# ── WAVI velocity for exactly those three states (3 solves) ───────────────
quiet(f) = redirect_stdout(f, devnull)
function wavi_speed(beta_field)                 # beta_field is (ny, nx)
    uv = quiet(() -> IceFlow.velocity_flat(vec(beta_field)))
    reshape(sqrt.(uv.u .^ 2 .+ uv.v .^ 2), ny, nx)
end
println("Running WAVI on initial-guess / truth / mean β (3 solves) ...")
init_spd  = wavi_speed(init_beta)
truth_spd = wavi_speed(truth_bF)
mean_spd  = wavi_speed(mean_bF)

# ── geometry for axes + sensor markers (matches experiment_18) ────────────
x_length = 160_000.0; y_length = 160_000.0
dx = x_length / nx; dy = y_length / ny
gx = collect(0:nx-1) .* (dx / 1000); gy = collect(0:ny-1) .* (dy / 1000)
sx = Float64[]; sy = Float64[]
for idx in sensors
    j = ((idx - 1) % ny) + 1; i = ((idx - 1) ÷ ny) + 1
    push!(sx, (i - 1) * dx / 1000); push!(sy, (j - 1) * dy / 1000)
end

# shared colour scale across all three panels so the "bad → good" is honest
shared_clims(fields...) = begin
    v = vcat(vec.(fields)...)
    (quantile(v, 0.02), quantile(v, 0.98))
end

panel(field, ttl, clims, cmap, mcolor; cbtitle="") =
    let h = heatmap(gx, gy, field; clims=clims, c=cmap, aspect_ratio=1,
                    title=ttl, xlabel="x (km)", ylabel="y (km)",
                    colorbar_title=cbtitle)
        scatter!(h, sx, sy; mc=mcolor, ms=3, msw=0, label=false); h
    end

# ── β field: initial guess | truth (final) | mean (final) ─────────────────
cb = shared_clims(init_beta, truth_bF, mean_bF)
b1 = panel(init_beta, "initial guess β (mean, t=0)", cb, :viridis, :red)
b2 = panel(truth_bF,  "truth β (final)",             cb, :viridis, :red)
b3 = panel(mean_bF,   "mean β (final)",              cb, :viridis, :red)
savefig(plot(b1, b2, b3; layout=(1, 3), size=(1750, 460),
             left_margin=8mm, right_margin=8mm, bottom_margin=6mm, top_margin=3mm),
        joinpath(OUT, "beta_field_truth_vs_mean.png"))

# ── WAVI speed field: initial guess | truth (final) | mean (final) ────────
cv = shared_clims(init_spd, truth_spd, mean_spd)
v1 = panel(init_spd,  "initial guess WAVI speed (t=0)", cv, :thermal, :cyan; cbtitle="m/yr")
v2 = panel(truth_spd, "truth WAVI speed (final)",       cv, :thermal, :cyan; cbtitle="m/yr")
v3 = panel(mean_spd,  "mean WAVI speed (final)",        cv, :thermal, :cyan; cbtitle="m/yr")
savefig(plot(v1, v2, v3; layout=(1, 3), size=(1750, 460),
             left_margin=8mm, right_margin=8mm, bottom_margin=6mm, top_margin=3mm),
        joinpath(OUT, "velocity_field_truth_vs_mean.png"))

# ── quick numeric summary of the improvement ──────────────────────────────
rmse(a, b) = sqrt(mean((a .- b) .^ 2))
@printf("\nβ RMSE vs final truth:  initial guess=%.1f  final mean=%.1f\n",
        rmse(init_beta, truth_bF), rmse(mean_bF, truth_bF))
@printf("speed RMSE vs final truth: initial guess=%.1f  final mean=%.1f m/yr\n",
        rmse(init_spd, truth_spd), rmse(mean_spd, truth_spd))
println("\nSaved:")
println("  ", joinpath(OUT, "beta_field_truth_vs_mean.png"))
println("  ", joinpath(OUT, "velocity_field_truth_vs_mean.png"))
