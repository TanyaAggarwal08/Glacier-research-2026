# run17 — β-vs-x cross-section drawn as individual particle paths (spaghetti).
#
# Same section as crosssection_final_tempered.png (row with the most sensors),
# but every line is one particle instead of a percentile band, so the diversity
# tempering maintains is visible particle-by-particle. Reads the full particle
# dump back from the external SSD; nothing is re-simulated.
#
# Run: julia --project=test glacier-code/particleda/plot_run17_particle_paths.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)
using HDF5, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots

const RUN_NAME = "run17_pseudorandom_wave_16obs_tempered_logbeta"
const OUT = joinpath("glacier-code", "particleda", "results", RUN_NAME)
mkpath(OUT)
const TRACK = let
    local_h5 = joinpath(OUT, "tracking.h5")
    ssd_h5   = joinpath("/Volumes/ZX20/USRA 2026", RUN_NAME, "tracking.h5")
    isfile(local_h5) ? local_h5 : ssd_h5
end
@assert isfile(TRACK) "tracking.h5 not found (checked local and SSD)"
println("Reading $TRACK")

const N_SHOW = 150      # individual particle lines to draw (subset of NPRT)

f = h5open(TRACK, "r")
truth_grid = read(f["truth"]["beta"])            # (ny, nx, T+1)
mean_grid  = read(f["ensemble_mean"]["beta"])    # (ny, nx, T+1)
sensor_idx = read(f["sensor_indices"])
nx = read(attributes(f["parameters"])["nx"])
ny = read(attributes(f["parameters"])["ny"])
NPRT = read(attributes(f["parameters"])["nprt"])
T    = read(attributes(f["parameters"])["T"])
dt   = read(attributes(f["parameters"])["time_step"])
x_length = 160_000.0; y_length = 160_000.0

# Row with the most sensors (matches the run17 cross-section choice).
sensor_rows = [((idx - 1) % ny) + 1 for idx in sensor_idx]
row_counts  = Dict{Int,Int}()
for r in sensor_rows; row_counts[r] = get(row_counts, r, 0) + 1; end
row_mid = argmax(row_counts)
row_y_km = (row_mid - 1) * (y_length / ny) / 1000
xs_km = collect(0:nx-1) .* (x_length / nx) / 1000
sensors_on_row_x = [ (((idx-1) ÷ ny))*(x_length/nx)/1000
                     for idx in sensor_idx if (((idx-1) % ny)+1) == row_mid ]
println("Cross-section row=$row_mid (y≈$(round(row_y_km,digits=1)) km), $(length(sensors_on_row_x)) sensors on it")

# Evenly-spaced subset of particle columns for the section at row_mid.
show_idx = round.(Int, range(1, NPRT; length=N_SHOW))
# Read only the needed row slice per timestep: (nx, NPRT).
row_slice(t) = f["particles_all"]["beta"][row_mid, :, :, t]

clim = (quantile(vec(truth_grid), 0.02), quantile(vec(truth_grid), 0.98))

function draw_section(t)
    P = row_slice(t)                              # (nx, NPRT)
    p = plot(; xlabel="x (km)", ylabel="β (Pa·s/m)",
             title=@sprintf("run17 tempered — β cross-section (particle paths)  y=%.0f km  t=%.0f h",
                            row_y_km, (t-1)*dt/3600),
             ylim=(minimum(clim), maximum(clim)), legend=:topright)
    for (n, ip) in enumerate(show_idx)
        plot!(p, xs_km, P[:, ip]; lw=0.5, color=:steelblue, alpha=0.10,
              label=(n==1 ? "particles ($N_SHOW shown)" : false))
    end
    plot!(p, xs_km, mean_grid[row_mid, :, t]; lw=2, ls=:dash, color=:navy,
          label="ensemble mean")
    plot!(p, xs_km, truth_grid[row_mid, :, t]; lw=3, color=:black, label="truth")
    for (i_s, sx) in enumerate(sensors_on_row_x)
        vline!(p, [sx]; color=:red, ls=:dot, lw=1.2, label=(i_s==1 ? "sensor x" : false))
    end
    return p
end

savefig(draw_section(T+1), joinpath(OUT, "crosssection_final_particles.png"))
println("Saved → $(joinpath(OUT, "crosssection_final_particles.png"))")

anim = @animate for t in 1:T+1
    draw_section(t)
end
gif(anim, joinpath(OUT, "crosssection_anim_particles.gif"), fps=6)
println("Saved → $(joinpath(OUT, "crosssection_anim_particles.gif"))")

close(f)
println("Done.")
