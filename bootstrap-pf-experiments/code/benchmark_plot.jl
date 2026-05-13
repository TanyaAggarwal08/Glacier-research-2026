# Run from anywhere:
#   julia --project=test bootstrap-pf-experiments/code/benchmark_plot.jl
ENV["GKSwstype"] = "100"   # headless GR backend for PNG output

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
const RESULTS_DIR = joinpath(REPO_ROOT, "bootstrap-pf-experiments", "results")
const OBS_PATH = joinpath(RESULTS_DIR, "benchmark_obs.h5")
const DA_PATH  = joinpath(RESULTS_DIR, "particle_da.h5")

using HDF5, Statistics, Plots

# Paper snapshot times (s) → with time_step = 5.0 these are integer step counts
const SNAPSHOT_TIMES  = [0, 320, 740, 960, 1280]
const SNAPSHOT_STEPS  = [0,  64, 148, 192,  256]
const N_PARTICLES     = 200

tkey(n) = "t" * lpad(string(n), 4, '0')

# ── Read data ─────────────────────────────────────────────────────────────────
x_km = h5open(f -> read(f["grid_coordinates"]["x"]), DA_PATH, "r") ./ 1e3
y_km = h5open(f -> read(f["grid_coordinates"]["y"]), DA_PATH, "r") ./ 1e3

sx_km, sy_km, truth_snaps, mean_snaps, all_weights = h5open(DA_PATH, "r") do fda
    h5open(OBS_PATH, "r") do fobs
        sx = read(fda["station_coordinates"]["x"]) ./ 1e3
        sy = read(fda["station_coordinates"]["y"]) ./ 1e3
        tr = [read(fobs["state"][tkey(s)]["height"]) for s in SNAPSHOT_STEPS]
        mn = [read(fda["state_avg"][tkey(s)]["height"]) for s in SNAPSHOT_STEPS]
        # weights/t0000 are uninitialised — start at t=1
        n_steps = length(keys(fda["weights"])) - 1
        ww = [read(fda["weights"][tkey(t)]) for t in 1:n_steps]
        sx, sy, tr, mn, ww
    end
end

# ── Figure 1: 5 × 2 truth-vs-assimilated grid ────────────────────────────────
panels = Vector{Any}(undef, 10)
for (row, (t_s, truth, mn)) in enumerate(zip(SNAPSHOT_TIMES, truth_snaps, mean_snaps))
    # Shared colour scale per row (per the paper figure)
    lo, hi = extrema([truth; mn])
    clims = (lo, hi)

    show_xlabel = (row == length(SNAPSHOT_TIMES))
    xlab = show_xlabel ? "x (km)" : ""

    left = heatmap(x_km, y_km, truth';
                   title = (row == 1 ? "Surface Elevation Truth" : ""),
                   xlabel = xlab, ylabel = "y (km)",
                   color = :viridis, clims = clims,
                   aspect_ratio = 1, xlims = (0, 200), ylims = (0, 200),
                   colorbar_title = "m")
    scatter!(left, sx_km, sy_km;
             marker = :star5, color = :red, markersize = 4,
             label = (row == 1 ? "Observation Locations" : ""),
             legend = (row == 1 ? :bottomright : false))

    right = heatmap(x_km, y_km, mn';
                    title = (row == 1 ? "Surface Elevation Assimilated" : ""),
                    xlabel = xlab, ylabel = "",
                    color = :viridis, clims = clims,
                    aspect_ratio = 1, xlims = (0, 200), ylims = (0, 200),
                    colorbar_title = "m",
                    right_margin = 8 * Plots.mm)
    scatter!(right, sx_km, sy_km;
             marker = :star5, color = :red, markersize = 4,
             label = (row == 1 ? "Observation Locations" : ""),
             legend = (row == 1 ? :bottomright : false))

    # Annotate row time in the right panel (top-right corner outside plot area)
    annotate!(right, 215, 100, text("T = $(t_s)s", :black, 10, :left))

    panels[2 * row - 1] = left
    panels[2 * row]     = right
end

fig1 = plot(panels...; layout = (5, 2), size = (1200, 2400),
            plot_title = "LLW2d Bootstrap PF  (N=$N_PARTICLES)")
savefig(fig1, joinpath(RESULTS_DIR, "benchmark_fields.png"))
println("Saved benchmark_fields.png")

# ── Figure 2: ESS over time ──────────────────────────────────────────────────
ess = map(all_weights) do w
    w_n = w ./ sum(w)
    1.0 / sum(w_n .^ 2)
end

n_steps = length(ess)
fig2 = plot(1:n_steps, ess;
            xlabel = "Timestep", ylabel = "ESS",
            title = "Effective Sample Size (Bootstrap, N=$N_PARTICLES)",
            label = "ESS", linewidth = 2, legend = :topright,
            ylims = (0, N_PARTICLES + 5))
hline!(fig2, [N_PARTICLES * 0.1];
       linestyle = :dash, color = :red,
       label = "10% of N (ESS = $(Int(N_PARTICLES * 0.1)))")
# Mark the snapshot timesteps
vline!(fig2, SNAPSHOT_STEPS[2:end];
       linestyle = :dot, color = :gray, alpha = 0.6,
       label = "snapshot times")
savefig(fig2, joinpath(RESULTS_DIR, "benchmark_ess.png"))
println("Saved benchmark_ess.png")

# ── Figure 3: sorted normalised weights at last snapshot ──────────────────────
final_step = SNAPSHOT_STEPS[end]
w_final = all_weights[final_step]
w_final_norm = sort(w_final ./ sum(w_final))
fig3 = bar(1:length(w_final_norm), w_final_norm;
           xlabel = "Particle rank (sorted)", ylabel = "Normalised weight",
           title = "Sorted normalised weights at t=$(final_step)  (T=$(SNAPSHOT_TIMES[end])s)",
           legend = false)
savefig(fig3, joinpath(RESULTS_DIR, "benchmark_weights.png"))
println("Saved benchmark_weights.png")

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n── Summary ──────────────────────────────────────────────")
println("  Snapshots:  steps $(SNAPSHOT_STEPS)  →  T = $(SNAPSHOT_TIMES) s")
println("  ESS range over t=1..$n_steps: $(round(minimum(ess),digits=2)) – $(round(maximum(ess),digits=2))")
println("  Max weight at final snapshot: $(round(maximum(w_final_norm), digits=4))")
for (t_s, tr, mn) in zip(SNAPSHOT_TIMES, truth_snaps, mean_snaps)
    println("  T=$(t_s)s  truth range $(round.(extrema(tr), digits=2))   mean range $(round.(extrema(mn), digits=2))")
end
