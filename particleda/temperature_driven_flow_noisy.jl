# Temperature-driven β → ice velocity WITH smooth process noise on the temperature.
#
# Extends temperature_driven_flow.jl by perturbing the temperature field before
# it is mapped to β:
#
#   T*(x,t) = T(x,t) + σ_T · N(x,t)          noisy temperature
#   N(·,t)  = ρ·N(·,t−1) + √(1−ρ²)·L·z_t     AR(1) in time, smooth in space
#   β(x,t)  = β_center − k·(T*(x,t) − T_c)   same linear map as before
#   u(x,t)  = WAVI(β)                        full momentum solve per step
#
# The noise is SMOOTH in both senses:
#   • space — innovations are L·z where L is the Cholesky factor of a squared-
#     exponential kernel, exp(−r²/2ℓ²) with ℓ = 15 km. This is the SAME generator
#     the particle filter uses for its process noise (IceExpDyn._build_noise_factor),
#     so the perturbations look like the ones the filter is built to handle rather
#     than white pixel noise.
#   • time — an AR(1) chain with decorrelation timescale τ, so the field drifts
#     rather than flickering independently every step. ρ = exp(−Δt/τ).
# Because K has unit diagonal, L·z has unit variance per cell, so σ_T is directly
# the standard deviation of the temperature perturbation in °C.
#
# The CLEAN run is computed alongside the noisy one (same steps, same WAVI) so
# every plot compares like with like — that costs 2×(nt) solves, still ~1 min.
#
# RUN UNDER THE DEFAULT ENV (needs WAVI):
#   julia .../temperature_driven_flow_noisy.jl [n_days] [start_day] [per_day] [sigma_T] [tau_hours] [seed]
# defaults: 3 days, day 200, 24/day (hourly), σ_T = 2.0 °C, τ = 6 h, seed 42
#
# Outputs to results/noise_temperature_forcing/ :
#   tempflow_noisy_evolution.gif  — clean T | noisy T | β | WAVI speed, per step
#   tempflow_noisy_timeseries.png — clean vs noisy T, β, speed vs time
#   tempflow_noisy_response.png   — β–speed response, clean vs noisy
#   tempflow_noise_field.png      — the noise itself: snapshots + autocorrelation

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Statistics, Printf, Random, LinearAlgebra, DelimitedFiles
include(joinpath(@__DIR__, "temperature_graph.jl"))   # T(x,t), β(T); NX,NY,L
include(joinpath(@__DIR__, "ice_experiment_dynamics.jl")); using .IceExpDyn
include(joinpath(@__DIR__, "ice_flow.jl")); using .IceFlow
using Plots, Plots.PlotMeasures

const NDAYS  = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 3.0
const DAY0   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 200.0
const PERDAY = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 24
const SIGMA_T = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0    # °C
const TAU_H   = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 6.0    # hours
const SEED    = length(ARGS) >= 6 ? parse(Int,     ARGS[6]) : 42
const OUT = joinpath("glacier-code", "particleda", "results", "noise_temperature_forcing")
mkpath(OUT)

ts = collect(DAY0:(1 / PERDAY):(DAY0 + NDAYS))
nt = length(ts)
dt_h = 24 / PERDAY                      # step length in hours
rho  = exp(-dt_h / TAU_H)               # AR(1) memory

@printf("Noisy temperature-driven flow: %d steps over %.1f days from day %.1f\n", nt, NDAYS, DAY0)
@printf("noise: sigma_T=%.2f °C, tau=%.1f h (rho=%.3f per %.1f h step), ell=15 km, seed=%d\n",
        SIGMA_T, TAU_H, rho, dt_h, SEED)
@printf("weertman_m=%.1f  v_iszero=%s\n", IceFlow.WEERTMAN_M, IceFlow.V_ISZERO)

# Smooth-in-space generator: reuse the PF's own noise factor (ℓ = 15 km).
pnoise = IceExpDyn.Params(nx = NX, ny = NY, x_length = L, y_length = L,
                          noise_length_scale = 15_000.0,
                          station_filename = joinpath(@__DIR__, "stations_grid_16.txt"))
Lfac = IceExpDyn._build_noise_factor(pnoise)
rng  = MersenneTwister(SEED)
smooth_draw() = reshape(Lfac * randn(rng, NX * NY), NY, NX)

# Sensors (same 16-station grid as the PF runs).
coords = readdlm(joinpath(@__DIR__, "stations_grid_16.txt"), ',', Float64, '\n';
                 comments = true, comment_char = '#')
dx = L / NX
sens = [(clamp(round(Int, coords[r, 2] / dx) + 1, 1, NY),
         clamp(round(Int, coords[r, 1] / dx) + 1, 1, NX)) for r in axes(coords, 1)]
const IN = 4:37       # interior mask (drop 3-cell border)

speed_of(bfield) = reshape(
    (redirect_stdout(devnull) do; f = IceFlow.velocity_flat(vec(bfield)); sqrt.(f.u.^2 .+ f.v.^2) end),
    NY, NX)

Tc = [Array{Float64}(undef, NY, NX) for _ in 1:nt]   # clean T
Tn = [Array{Float64}(undef, NY, NX) for _ in 1:nt]   # noisy T
Nf = [Array{Float64}(undef, NY, NX) for _ in 1:nt]   # the noise field itself
Bc = [Array{Float64}(undef, NY, NX) for _ in 1:nt]
Bn = [Array{Float64}(undef, NY, NX) for _ in 1:nt]
Sc = [Array{Float64}(undef, NY, NX) for _ in 1:nt]
Sn = [Array{Float64}(undef, NY, NX) for _ in 1:nt]

Nprev = smooth_draw()                 # stationary start: unit variance already
print("Running WAVI (clean + noisy): ")
for (n, t) in enumerate(ts)
    Nf[n] .= n == 1 ? Nprev : (rho .* Nprev .+ sqrt(1 - rho^2) .* smooth_draw())
    global Nprev = Nf[n]
    Tc[n] .= temperature_field(t)
    Tn[n] .= Tc[n] .+ SIGMA_T .* Nf[n]
    Bc[n] .= beta_from_temperature(Tc[n])
    Bn[n] .= beta_from_temperature(Tn[n])
    Sc[n] .= speed_of(Bc[n])
    Sn[n] .= speed_of(Bn[n])
    n % 12 == 0 && print("$n ")
end
println("done")

im(v) = [mean(x[IN, IN]) for x in v]
Tcm, Tnm, Bcm, Bnm, Scm, Snm = im(Tc), im(Tn), im(Bc), im(Bn), im(Sc), im(Sn)
Snsens = [mean([S[j, i] for (j, i) in sens]) for S in Sn]

@printf("\nnoise realised: sd=%.2f °C (target %.2f), lag-1 autocorr=%.3f (target rho=%.3f)\n",
        std(vcat(vec.(Nf)...)) * SIGMA_T, SIGMA_T,
        cor(vcat(vec.(Nf[1:end-1])...), vcat(vec.(Nf[2:end])...)), rho)
@printf("T  clean %.2f–%.2f  | noisy %.2f–%.2f °C\n",
        minimum(Tcm), maximum(Tcm), minimum(Tnm), maximum(Tnm))
@printf("β  clean %.0f–%.0f  | noisy %.0f–%.0f Pa·s/m\n",
        minimum(Bcm), maximum(Bcm), minimum(Bnm), maximum(Bnm))
@printf("speed clean %.2f–%.2f | noisy %.2f–%.2f m/yr\n",
        minimum(Scm), maximum(Scm), minimum(Snm), maximum(Snm))
@printf("corr(β,speed): clean %+.4f | noisy %+.4f   (interior means over time)\n",
        cor(Bcm, Scm), cor(Bnm, Snm))
@printf("corr(β,speed) pooled over ALL cells+times: clean %+.4f | noisy %+.4f\n",
        cor(vcat(vec.(Bc)...), vcat(vec.(Sc)...)), cor(vcat(vec.(Bn)...), vcat(vec.(Sn)...)))
@printf("speed departure from clean: rms %.2f m/yr (%.1f%% of clean mean)\n",
        sqrt(mean((Snm .- Scm) .^ 2)), 100 * sqrt(mean((Snm .- Scm) .^ 2)) / mean(Scm))

# ── animation ────────────────────────────────────────────────────────────
xs = collect(0:NX-1) .* (dx / 1000); ys = collect(0:NY-1) .* (dx / 1000)
sxs = [(i - 1) * dx / 1000 for (_, i) in sens]; sys = [(j - 1) * dx / 1000 for (j, _) in sens]
q(v, a) = quantile(vcat(vec.(v)...), a)
Tlim = (min(q(Tc, 0.01), q(Tn, 0.01)), max(q(Tc, 0.99), q(Tn, 0.99)))
Blim = (min(q(Bc, 0.01), q(Bn, 0.01)), max(q(Bc, 0.99), q(Bn, 0.99)))
# speed: interior-only limits — the boundary jets are gone under the sidewall BC,
# but the x=0 divide edge is still fast enough to skew a raw min/max.
Slim = (quantile(vcat([vec(S[IN,IN]) for S in Sn]...), 0.01),
        quantile(vcat([vec(S[IN,IN]) for S in Sn]...), 0.99))

pan(f, ttl, lm, cm, cb) =
    heatmap(xs, ys, f, aspect_ratio = 1, color = cm, clims = lm, colorbar_title = cb,
            title = ttl, titlefontsize = 9, xlabel = "x (km)", ylabel = "y (km)")

anim = @animate for n in 1:nt
    p1 = pan(Tc[n], "clean T", Tlim, :thermal, "°C")
    p2 = pan(Tn[n], "noisy T (σ=$(SIGMA_T)°C, τ=$(TAU_H)h)", Tlim, :thermal, "°C")
    p3 = pan(Bn[n], "β from noisy T", Blim, :viridis, "Pa·s/m")
    p4 = pan(Sn[n], "WAVI speed", Slim, :thermal, "m/yr")
    scatter!(p4, sxs, sys, ms = 2.0, color = :cyan, label = "")
    plot(p1, p2, p3, p4, layout = (1, 4), size = (1900, 470),
         plot_title = @sprintf("day %.2f (hour %d)   T̄ %.1f→%.1f°C   β̄ %.0f   speed %.1f m/yr",
                               ts[n], n - 1, Tcm[n], Tnm[n], Bnm[n], Snm[n]),
         plot_titlefontsize = 11,
         left_margin = 9mm, bottom_margin = 6mm, right_margin = 3mm, top_margin = 2mm)
end
gif(anim, joinpath(OUT, "tempflow_noisy_evolution.gif"), fps = 6)

# ── time series: clean vs noisy ──────────────────────────────────────────
hrs = (ts .- DAY0) .* 24
p1 = plot(hrs, Tcm, lw = 2, color = :gray40, ls = :dash, label = "clean",
          ylabel = "T (°C)", title = "Interior-mean surface temperature", legend = :best)
plot!(p1, hrs, Tnm, lw = 2, color = :orangered, label = "noisy")
p2 = plot(hrs, Bcm, lw = 2, color = :gray40, ls = :dash, label = "clean",
          ylabel = "β (Pa·s/m)", title = "β from T", legend = false)
plot!(p2, hrs, Bnm, lw = 2, color = :purple, label = "noisy")
p3 = plot(hrs, Scm, lw = 2, color = :gray40, ls = :dash, label = "clean",
          ylabel = "speed (m/yr)", xlabel = "hours from day $(Int(DAY0))",
          title = "WAVI speed", legend = :best)
plot!(p3, hrs, Snm, lw = 2, color = :steelblue, label = "noisy (interior)")
plot!(p3, hrs, Snsens, lw = 1.5, ls = :dot, color = :seagreen, label = "noisy (16 sensors)")
plot(p1, p2, p3, layout = (3, 1), size = (950, 850), left_margin = 10mm, bottom_margin = 5mm)
savefig(joinpath(OUT, "tempflow_noisy_timeseries.png"))

# ── response curve ───────────────────────────────────────────────────────
scatter(Bcm, Scm, ms = 5, msw = 0, color = :gray60, label = @sprintf("clean (r=%+.3f)", cor(Bcm, Scm)),
        xlabel = "interior-mean β (Pa·s/m)", ylabel = "interior-mean speed (m/yr)",
        title = "β–speed response: does noise break the relationship?",
        size = (820, 620), legend = :topright, left_margin = 6mm)
scatter!(Bnm, Snm, ms = 5, msw = 0, color = :crimson, label = @sprintf("noisy (r=%+.3f)", cor(Bnm, Snm)))
savefig(joinpath(OUT, "tempflow_noisy_response.png"))

# ── the noise field itself ───────────────────────────────────────────────
k = max(1, nt ÷ 4)
np = [heatmap(xs, ys, SIGMA_T .* Nf[i], aspect_ratio = 1, color = :balance,
              clims = (-3 * SIGMA_T, 3 * SIGMA_T), title = @sprintf("hour %d", i - 1),
              titlefontsize = 9, colorbar_title = "°C") for i in (1, k, 2k, 3k)]
lags = 0:min(24, nt - 1)
ac = [cor(vcat(vec.(Nf[1:end-l])...), vcat(vec.(Nf[1+l:end])...)) for l in lags]
pac = plot(lags .* dt_h, ac, lw = 2, marker = :circle, ms = 3, color = :darkgreen, legend = false,
           xlabel = "lag (hours)", ylabel = "autocorrelation",
           title = @sprintf("temporal autocorrelation (τ = %.1f h)", TAU_H))
plot!(pac, lags .* dt_h, exp.(-(lags .* dt_h) ./ TAU_H), lw = 1.5, ls = :dash, color = :black)
plot(np..., pac, layout = @layout([grid(1, 4); a]), size = (1600, 760),
     left_margin = 8mm, bottom_margin = 6mm)
savefig(joinpath(OUT, "tempflow_noise_field.png"))

println("Saved → $OUT/")
