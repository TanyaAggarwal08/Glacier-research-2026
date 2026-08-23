# Exp 18 — animated β evolution: truth | ensemble mean | error, frame per step.
#
# The 3-panel static figures only show t=0 and t=T, which hides how the filter
# actually converges. Everything needed for the animation is already in
# tracking.h5 (truth/beta and ensemble_mean/beta are both (ny,nx,T+1)) — no
# WAVI solves and no re-running the experiment.
#
#   beta_evolution.gif : truth β | mean β | error (mean − truth), 21 frames
#
# Colour limits are FIXED across all frames (computed once over the whole
# trajectory) so brightness changes mean real changes, not rescaling. The error
# panel uses a symmetric diverging scale centred on zero: blue = filter
# underestimates β, red = overestimates.
#
# RUN UNDER THE DEFAULT ENV (no WAVI needed, no threads):
#   julia glacier-code/particleda/plot_exp18_beta_gif.jl [run_tag] [fps]

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Statistics, Printf, HDF5
ENV["GKSwstype"] = "100"
using Plots, Plots.PlotMeasures

const RUN_TAG = length(ARGS) >= 1 ? ARGS[1] : "experiment_18_wavi_obs"
const FPS     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const OUT = joinpath("glacier-code", "particleda", "results", RUN_TAG)
mkpath(OUT)

# ── locate the saved trajectory (SSD preferred, checkpoint fallback) ──────
const SSD = joinpath("/Volumes/ZX20/USRA 2026", RUN_TAG)
h5path = let cands = [joinpath(SSD, "tracking.h5"), joinpath(SSD, "checkpoint.h5"),
                      joinpath(OUT, "tracking.h5")]
    i = findfirst(isfile, cands)
    i === nothing && error("No tracking.h5/checkpoint.h5 found under $SSD — is the SSD mounted?")
    cands[i]
end
println("Reading $h5path")

truth, mean_beta, ess, sensors, nx, ny = h5open(h5path, "r") do f
    tb = read(f["truth/beta"])            # (ny, nx, T+1)
    mb = read(f["ensemble_mean/beta"])    # (ny, nx, T+1)
    es = haskey(f, "weights/ess") ? read(f["weights/ess"]) : Float64[]
    si = Int.(read(f["sensor_indices"]))
    (tb, mb, es, si, size(tb, 2), size(tb, 1))
end

T = size(truth, 3) - 1
@printf("grid %d×%d, T=%d (%d frames), %d sensors\n", nx, ny, T, T + 1, length(sensors))

# Axes + sensor markers — SAME convention as plot_exp18_field_evolution.jl
# (grid nodes at (i−1)·dx, flat index → (j,i) column-major) so markers land in
# the same place as in every other figure for this experiment.
dx = 160_000.0 / nx; dy = 160_000.0 / ny
xs = collect(0:nx-1) .* (dx / 1000)
ys = collect(0:ny-1) .* (dy / 1000)
sx = [(((s - 1) ÷ ny) + 1 - 1) * dx / 1000 for s in sensors]
sy = [(((s - 1) % ny) + 1 - 1) * dy / 1000 for s in sensors]

err = mean_beta .- truth

# Fixed scales across the whole animation.
blim = (min(minimum(truth), minimum(mean_beta)), max(maximum(truth), maximum(mean_beta)))
elim = maximum(abs.(err));  elim = (-elim, elim)
rmse = [sqrt(mean((mean_beta[:, :, t] .- truth[:, :, t]) .^ 2)) for t in 1:T+1]
@printf("β range %.0f – %.0f | max |error| %.0f | RMSE %.1f → %.1f\n",
        blim..., elim[2], rmse[1], rmse[end])

panel(f, ttl, lims, cmap, cbt) =
    heatmap(xs, ys, f, aspect_ratio = 1, color = cmap, clims = lims,
            colorbar_title = cbt, title = ttl, titlefontsize = 9,
            xlabel = "x (km)", ylabel = "y (km)")

anim = @animate for t in 1:T+1
    step = t - 1
    p1 = panel(truth[:, :, t], "truth β", blim, :viridis, "Pa·s/m")
    p2 = panel(mean_beta[:, :, t], "ensemble mean β", blim, :viridis, "Pa·s/m")
    p3 = panel(err[:, :, t], "error (mean − truth)", elim, :balance, "Pa·s/m")
    for p in (p1, p2)
        scatter!(p, sx, sy, ms = 2.5, color = :red, label = "")
    end
    scatter!(p3, sx, sy, ms = 2.5, color = :black, label = "")
    ttl = step == 0 ?
        @sprintf("%s — t=0 (prior, before any assimilation)   RMSE=%.1f", RUN_TAG, rmse[t]) :
        @sprintf("%s — step %d/%d (model hour %d)   RMSE=%.1f%s",
                 RUN_TAG, step, T, step, rmse[t],
                 isempty(ess) ? "" : @sprintf("   ESS=%.0f", ess[step]))
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 480),
         plot_title = ttl, plot_titlefontsize = 11,
         left_margin = 9mm, bottom_margin = 6mm, right_margin = 3mm, top_margin = 2mm)
end

out = joinpath(OUT, "beta_evolution.gif")
gif(anim, out, fps = FPS)
println("Saved → $out")
