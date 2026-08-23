# Exp 18 — animated VELOCITY evolution: truth | ensemble mean | error, per step.
#
# Companion to plot_exp18_beta_gif.jl. The velocity fields were never written to
# tracking.h5 (only β and the 16-sensor log-speeds), but that does NOT require
# re-running the experiment: β is the complete state, so the velocity at every
# frame is recovered by re-solving WAVI on the stored β. That is 2·(T+1) = 42
# solves, ~10 s, versus ~25 min to redo the assimilation.
#
#   velocity_evolution.gif : truth speed | mean speed | error (mean − truth)
#
# IMPORTANT — boundary conditions must match the run being visualised. ice_flow.jl
# now defaults to free-slip sidewalls (v_iszero = south,east,west). Runs made
# before that change (experiment_18_wavi_obs{,_seed2,_weertman_m},
# experiment_18_double_bump{,_weertman_m,_center1000}) used open edges and must
# be re-solved the same way or the velocities will not match what the filter saw:
#
#   WAVI_V_ISZERO=south julia .../plot_exp18_velocity_gif.jl experiment_18_double_bump
#
# The script prints the active BC so the figure can always be traced back.
#
# As in the β gif, colour limits are FIXED across all frames (per-frame
# autoscaling would fake convergence), truth and mean share one scale, and the
# error panel is symmetric-diverging so convergence reads as fading to white.
#
# RUN UNDER THE DEFAULT ENV (needs WAVI):
#   julia glacier-code/particleda/plot_exp18_velocity_gif.jl [run_tag] [fps]

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Statistics, Printf, HDF5
ENV["GKSwstype"] = "100"
include(joinpath(@__DIR__, "ice_flow.jl")); using .IceFlow
using Plots, Plots.PlotMeasures

const RUN_TAG = length(ARGS) >= 1 ? ARGS[1] : "experiment_18_double_bump_sidewalls"
const FPS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
# Colour-limit mode:
#   "clip" (default) → 2nd–98th percentile, matching plot_exp18_field_evolution.jl
#                      → velocity_evolution.gif
#   "full"           → absolute min/max, nothing saturated, so extreme values are
#                      visible where they actually occur
#                      → velocity_evolution_full.gif
const CLIM_MODE = length(ARGS) >= 3 ? ARGS[3] : "clip"
CLIM_MODE in ("clip", "full") || error("clim mode must be \"clip\" or \"full\", got $CLIM_MODE")
const OUT = joinpath("glacier-code", "particleda", "results", RUN_TAG)
mkpath(OUT)

println("run_tag = $RUN_TAG")
println("BC: u_iszero=", IceFlow.U_ISZERO, "  v_iszero=", IceFlow.V_ISZERO)

const SSD = joinpath("/Volumes/ZX20/USRA 2026", RUN_TAG)
h5path = let cands = [joinpath(SSD, "tracking.h5"), joinpath(SSD, "checkpoint.h5"),
                      joinpath(OUT, "tracking.h5")]
    i = findfirst(isfile, cands)
    i === nothing && error("No tracking.h5/checkpoint.h5 under $SSD — is the SSD mounted?")
    cands[i]
end
println("Reading $h5path")

truth_b, mean_b, sensors, nx, ny = h5open(h5path, "r") do f
    (read(f["truth/beta"]), read(f["ensemble_mean/beta"]),
     Int.(read(f["sensor_indices"])), size(f["truth/beta"])[2], size(f["truth/beta"])[1])
end
T = size(truth_b, 3) - 1

speed(bfield) = let f = redirect_stdout(devnull) do
                        IceFlow.velocity_flat(vec(bfield))
                    end
    reshape(sqrt.(f.u .^ 2 .+ f.v .^ 2), ny, nx)
end

@printf("Re-solving WAVI on stored β: %d frames × 2 = %d solves ...\n", T + 1, 2 * (T + 1))
Strue = [speed(truth_b[:, :, t]) for t in 1:T+1]
Smean = [speed(mean_b[:, :, t])  for t in 1:T+1]
err   = [Smean[t] .- Strue[t] for t in 1:T+1]

# Axes + sensor markers: same convention as the other exp-18 figures.
dx = 160_000.0 / nx; dy = 160_000.0 / ny
xs = collect(0:nx-1) .* (dx / 1000); ys = collect(0:ny-1) .* (dy / 1000)
sx = [(((s - 1) ÷ ny) + 1 - 1) * dx / 1000 for s in sensors]
sy = [(((s - 1) % ny) + 1 - 1) * dy / 1000 for s in sensors]

# Colour limits: 2nd–98th percentile over ALL frames, matching the convention in
# plot_exp18_field_evolution.jl. Straight min/max is wrong here — the t=0
# ensemble mean carries a single hot spot (~103 m/yr from the bad initial guess)
# while 98% of all data sits below ~37, so one frame's outlier would compress
# every other frame into the bottom third of the colourbar. Still ONE scale for
# all frames (never per-frame autoscaling, which would fake convergence);
# out-of-range values saturate, exactly as in the static figures.
allv = vcat(vec.(Strue)..., vec.(Smean)...)
alle = abs.(vcat(vec.(err)...))
slim, elim = if CLIM_MODE == "clip"
    (quantile(allv, 0.02), quantile(allv, 0.98)), (-quantile(alle, 0.98), quantile(alle, 0.98))
else
    (minimum(allv), maximum(allv)), (-maximum(alle), maximum(alle))
end
@printf("clim mode = %s  → speed scale %.1f – %.1f m/yr, error scale ±%.1f\n",
        CLIM_MODE, slim..., elim[2])

# Report which frames the "clip" mode actually saturates, so the two versions can
# be compared knowing exactly what differs between them.
let hi = quantile(allv, 0.98)
    @printf("cells above the 98%% limit (%.1f m/yr), per frame — these saturate in \"clip\" mode:\n", hi)
    for t in 1:T+1
        cm = count(>(hi), Smean[t]); ct = count(>(hi), Strue[t])
        (cm + ct) > 0 && @printf("   step %2d: mean %3d/%d cells (max %.1f), truth %3d/%d (max %.1f)\n",
                                 t - 1, cm, length(Smean[t]), maximum(Smean[t]),
                                 ct, length(Strue[t]), maximum(Strue[t]))
    end
end
rmse = [sqrt(mean(e .^ 2)) for e in err]
@printf("speed %.1f – %.1f m/yr | max |error| %.1f | speed RMSE %.2f → %.2f m/yr\n",
        slim..., elim[2], rmse[1], rmse[end])

panel(f, ttl, lm, cm) =
    heatmap(xs, ys, f, aspect_ratio = 1, color = cm, clims = lm,
            colorbar_title = "m/yr", title = ttl, titlefontsize = 9,
            xlabel = "x (km)", ylabel = "y (km)")

anim = @animate for t in 1:T+1
    step = t - 1
    p1 = panel(Strue[t], "truth speed", slim, :thermal)
    p2 = panel(Smean[t], "ensemble mean speed", slim, :thermal)
    p3 = panel(err[t], "error (mean − truth)", elim, :balance)
    for p in (p1, p2); scatter!(p, sx, sy, ms = 2.5, color = :cyan, label = ""); end
    scatter!(p3, sx, sy, ms = 2.5, color = :black, label = "")
    scale = CLIM_MODE == "clip" ? "2–98% scale" : "full min–max scale"
    ttl = step == 0 ?
        @sprintf("%s — velocity, t=0 (prior)   speed RMSE=%.2f m/yr   [%s]",
                 RUN_TAG, rmse[t], scale) :
        @sprintf("%s — velocity, step %d/%d   speed RMSE=%.2f m/yr   [%s]",
                 RUN_TAG, step, T, rmse[t], scale)
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 480),
         plot_title = ttl, plot_titlefontsize = 11,
         left_margin = 9mm, bottom_margin = 6mm, right_margin = 3mm, top_margin = 2mm)
end

out = joinpath(OUT, CLIM_MODE == "clip" ? "velocity_evolution.gif" :
                                          "velocity_evolution_full.gif")
gif(anim, out, fps = FPS)
println("Saved → $out")
