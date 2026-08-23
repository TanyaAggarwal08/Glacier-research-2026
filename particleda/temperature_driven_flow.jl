# Temperature-driven β → ice velocity, forward run (NO advection, NO particle filter).
#
# Replaces the pseudo-random-wave + upwind-advection β dynamics used in the
# Exp-18 line with a β field driven entirely by the surface temperature forcing:
#
#   T(x,t) = Tmean + Aann·cos(2π(t−dpeak)/365)    seasonal
#                  + Adiur·cos(2π(t−φ))           diurnal
#                  − Γ·(z_s(x) − z_ref)           elevation lapse
#   β(x,t) = β_center − k·(T − T_center)          linear, inverse (warm ⇒ slippery)
#   u(x,t) = WAVI(β)                              full momentum solve per step
#
# Both the seasonal and diurnal terms are active (seasonal is kept even over a
# short window where it barely moves, as requested — it shifts the mean level
# the diurnal cycle oscillates about). All definitions come straight from
# temperature_graph.jl; nothing is duplicated here.
#
# This is a FORWARD run: β is prescribed from temperature at every step, so
# there is no advection, no process noise, and no state to estimate. It answers
# "what does temperature-driven β do to the velocity field", which is the
# prerequisite for later using it as PF dynamics.
#
# RUN UNDER THE DEFAULT ENV (needs WAVI), no threads required:
#   julia glacier-code/particleda/temperature_driven_flow.jl [n_days] [start_day] [steps_per_day]
# defaults: 3 days, starting day 200 (annual peak), 24 steps/day (hourly)
#
# Outputs to results/temperature_forcing/ :
#   tempflow_evolution.gif   — T | β | WAVI speed, one frame per step
#   tempflow_timeseries.png  — domain-mean T, β and speed vs time (+ sensor speeds)
#   tempflow_response.png    — β vs speed scatter, coloured by time

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Statistics, Printf
include(joinpath(@__DIR__, "temperature_graph.jl"))   # T(x,t), β(T); defines NX,NY,L,XS,YS
include(joinpath(@__DIR__, "ice_flow.jl")); using .IceFlow
using Plots, Plots.PlotMeasures

const NDAYS  = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 3.0
const DAY0   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 200.0
const PERDAY = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 24
const OUT = joinpath("glacier-code", "particleda", "results", "temperature_forcing")
mkpath(OUT)

ts = collect(DAY0:(1 / PERDAY):(DAY0 + NDAYS))       # days
nt = length(ts)
@printf("Temperature-driven flow: %d steps over %.1f days from day %.1f (%d/day)\n",
        nt, NDAYS, DAY0, PERDAY)
@printf("weertman_m = %.1f (from ice_flow.jl)\n", IceFlow.WEERTMAN_M)

# Sensors: reuse the 16-station 4×4 grid so results line up with the PF runs.
using DelimitedFiles
coords = readdlm(joinpath(@__DIR__, "stations_grid_16.txt"), ',', Float64, '\n';
                 comments = true, comment_char = '#')
dx = L / NX
sens = [(clamp(round(Int, coords[r, 2] / dx) + 1, 1, NY),      # (j = y, i = x)
         clamp(round(Int, coords[r, 1] / dx) + 1, 1, NX)) for r in axes(coords, 1)]

# Interior mask drops a 3-cell border: the north/south boundary conditions
# (u_iszero/v_iszero) create fast edge jets that swamp the β-driven signal.
const IN = 4:37

Tf = [Array{Float64}(undef, NY, NX) for _ in 1:nt]
Bf = [Array{Float64}(undef, NY, NX) for _ in 1:nt]
Sf = [Array{Float64}(undef, NY, NX) for _ in 1:nt]

print("Running WAVI: ")
for (n, t) in enumerate(ts)
    Tf[n] .= temperature_field(t)
    Bf[n] .= beta_field_from_temperature(t)
    f = redirect_stdout(devnull) do            # silence WAVI's per-solve chatter
        IceFlow.velocity_flat(vec(Bf[n]))
    end
    Sf[n] .= reshape(sqrt.(f.u .^ 2 .+ f.v .^ 2), NY, NX)
    n % 12 == 0 && print("$n ")
end
println("done")

# ── summary series ───────────────────────────────────────────────────────
Tmean_t = [mean(T[IN, IN]) for T in Tf]
Bmean_t = [mean(B[IN, IN]) for B in Bf]
Smean_t = [mean(S[IN, IN]) for S in Sf]
Ssens_t = [mean([S[j, i] for (j, i) in sens]) for S in Sf]

@printf("T  (interior mean): %.2f – %.2f °C\n", minimum(Tmean_t), maximum(Tmean_t))
@printf("β  (interior mean): %.1f – %.1f Pa·s/m\n", minimum(Bmean_t), maximum(Bmean_t))
@printf("speed (interior):   %.2f – %.2f m/yr  (swing %.1f%% of mean)\n",
        minimum(Smean_t), maximum(Smean_t),
        100 * (maximum(Smean_t) - minimum(Smean_t)) / mean(Smean_t))
@printf("corr(mean β, mean speed) = %+.4f\n", cor(Bmean_t, Smean_t))
@printf("corr(mean T, mean speed) = %+.4f\n", cor(Tmean_t, Smean_t))

# ── animation: T | β | speed, fixed colour limits across all frames ──────
xs = collect(0:NX-1) .* (dx / 1000); ys = collect(0:NY-1) .* (dx / 1000)
sxs = [(i - 1) * dx / 1000 for (_, i) in sens]
sys = [(j - 1) * dx / 1000 for (j, _) in sens]
lims(v) = (minimum(minimum.(v)), maximum(maximum.(v)))
Tlim, Blim = lims(Tf), lims(Bf)
# Speed is scaled to the INTERIOR range only. The north/south boundary jets run
# to ~1500 m/yr while the interior sits near 20, so a full-field scale renders
# the interior uniformly black and hides the entire diurnal signal. Edge cells
# saturate at the top of the colourbar as a result — that is intended.
Slim = (minimum(minimum(S[IN, IN]) for S in Sf), maximum(maximum(S[IN, IN]) for S in Sf))

pan(f, ttl, lm, cm, cb) =
    heatmap(xs, ys, f, aspect_ratio = 1, color = cm, clims = lm,
            colorbar_title = cb, title = ttl, titlefontsize = 9,
            xlabel = "x (km)", ylabel = "y (km)")

anim = @animate for n in 1:nt
    day = ts[n]
    p1 = pan(Tf[n], "surface T", Tlim, :thermal, "°C")
    p2 = pan(Bf[n], "β from T", Blim, :viridis, "Pa·s/m")
    p3 = pan(Sf[n], "WAVI speed (interior scale)", Slim, :thermal, "m/yr")
    scatter!(p3, sxs, sys, ms = 2.5, color = :cyan, label = "")
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 480),
         plot_title = @sprintf("day %.2f (hour %d of run)   T=%.1f°C  β=%.0f  speed=%.1f m/yr",
                               day, n - 1, Tmean_t[n], Bmean_t[n], Smean_t[n]),
         plot_titlefontsize = 11,
         left_margin = 9mm, bottom_margin = 6mm, right_margin = 3mm, top_margin = 2mm)
end
gif(anim, joinpath(OUT, "tempflow_evolution.gif"), fps = 6)

# ── time series: the diurnal cycle in T, β and speed ─────────────────────
hrs = (ts .- DAY0) .* 24
p1 = plot(hrs, Tmean_t, lw = 2, color = :orangered, legend = false,
          ylabel = "T (°C)", title = "Interior-mean surface temperature")
p2 = plot(hrs, Bmean_t, lw = 2, color = :purple, legend = false,
          ylabel = "β (Pa·s/m)", title = "β from T — inverted (warm ⇒ low β)")
p3 = plot(hrs, Smean_t, lw = 2, color = :steelblue, label = "interior mean",
          ylabel = "speed (m/yr)", xlabel = "hours from day $(Int(DAY0))",
          title = "WAVI speed — follows T, opposes β", legend = :best)
plot!(p3, hrs, Ssens_t, lw = 2, ls = :dash, color = :seagreen, label = "16-sensor mean")
plot(p1, p2, p3, layout = (3, 1), size = (950, 850),
     left_margin = 10mm, bottom_margin = 5mm)
savefig(joinpath(OUT, "tempflow_timeseries.png"))

# ── response curve: β vs speed, coloured by time ─────────────────────────
scatter(Bmean_t, Smean_t, zcolor = hrs, color = :viridis, ms = 5, msw = 0,
        colorbar_title = "hours", xlabel = "interior-mean β (Pa·s/m)",
        ylabel = "interior-mean speed (m/yr)", legend = false,
        title = @sprintf("β–speed response over %.1f days (corr = %+.3f)",
                         NDAYS, cor(Bmean_t, Smean_t)),
        size = (760, 600), right_margin = 8mm, left_margin = 6mm)
savefig(joinpath(OUT, "tempflow_response.png"))

println("Saved → $OUT/{tempflow_evolution.gif, tempflow_timeseries.png, tempflow_response.png}")
