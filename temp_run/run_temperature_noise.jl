#!/usr/bin/env julia
# =============================================================================
#  TEMPERATURE-DRIVEN BASAL DRAG → ICE VELOCITY, WITH SMOOTH PROCESS NOISE
# =============================================================================
#
#  ONE FILE, ONE COMMAND, ALL RESULTS:
#
#      julia glacier-code/temp_run/run_temperature_noise.jl
#
#  Everything lands in  glacier-code/temp_run/results/noise_temperature_forcing/
#
#  This script is SELF-CONTAINED: it defines the temperature model, the β(T)
#  map, the smooth-noise generator, the WAVI wrapper and every plot inline. It
#  does not `include` anything from glacier-code/particleda/. The exploratory
#  versions of this work live there and are kept as the testing record; this is
#  the clean re-implementation meant to be run and cited.
#
# -----------------------------------------------------------------------------
#  WHAT IT COMPUTES
# -----------------------------------------------------------------------------
#  A forward experiment (no data assimilation, nothing estimated) in four steps,
#  repeated at every time step:
#
#    1. TEMPERATURE      T(x,t) = Tmean
#                               + Aann ·cos(2π(t − dpeak)/365)      seasonal
#                               + Adiur·cos(2π(t − φ))              diurnal
#                               − Γ·(z_s(x) − z_ref)                elevation lapse
#
#    2. SMOOTH NOISE     T*(x,t) = T(x,t) + σ_T·N(x,t)
#                        N(·,t)  = ρ·N(·,t−1) + √(1−ρ²)·L·z_t
#                        • smooth in SPACE: L is the Cholesky factor of the
#                          squared-exponential kernel exp(−r²/2ℓ²), ℓ = 15 km —
#                          the SAME generator the particle filter uses for its
#                          process noise, so perturbations look like the ones
#                          the filter is designed to cope with, not white noise.
#                        • smooth in TIME: AR(1) with decorrelation time τ, so
#                          the field drifts instead of flickering every step.
#                          ρ = exp(−Δt/τ), and because the kernel has unit
#                          diagonal, σ_T is directly the perturbation sd in °C.
#
#    3. BASAL DRAG       β(x,t) = β_center − k·(T* − T_center),  floored at min_β
#                        A deliberately simple INVERSE linear caricature: warm
#                        ⇒ low β ⇒ slippery bed. Not a physical melt model.
#
#    4. ICE VELOCITY     u(x,t) = WAVI(β)   — full momentum solve, every step.
#
#  The CLEAN (noise-free) case is computed alongside the noisy one, step for
#  step, so every figure compares like with like. That doubles the solve count
#  and is still about a minute.
#
# -----------------------------------------------------------------------------
#  BOUNDARY CONDITIONS  (the single most confusing part of WAVI)
# -----------------------------------------------------------------------------
#  WAVI's orientation names refer to ARRAY INDICES, not compass directions.
#  In Grid.jl/orientations2bc the arrays are (x,y), so:
#
#      "north" → A[1,:]   = x = 0        "south" → A[end,:] = x = L
#      "west"  → A[:,1]   = y = 0        "east"  → A[:,end] = y = L
#
#  i.e. north/south are the X edges and east/west are the Y edges.
#
#  For any edge, the component that CROSSES it is the one to zero:
#      x-edges are crossed by u   →  u_iszero
#      y-edges are crossed by v   →  v_iszero
#
#  This script uses:
#      u_iszero = ["north"]                  → wall at x = 0 (ice divide)
#      v_iszero = ["south","east","west"]    → free-slip sidewalls at y = 0, y = L
#
#  Leaving the y-edges unconstrained (the original setup) makes them behave as
#  calving fronts: nothing resists the ice, so it ran at ~690 m/yr against an
#  interior of ~20 — 32× — which hijacked every colour scale and buried the β
#  signal. Adding the sidewalls drops that ratio to ~1.0 and was verified not to
#  change the physics: interior speed 17.8 → 17.3 m/yr, and in a full particle-
#  filter rerun final RMSE moved 113.4 → 112.6.
#
# -----------------------------------------------------------------------------
#  OUTPUTS  →  results/noise_temperature_forcing/
# -----------------------------------------------------------------------------
#    tempflow_noisy_evolution.gif   clean T | noisy T | β | speed, one frame/step
#    tempflow_noisy_timeseries.png  clean vs noisy T, β and speed against time
#    tempflow_noisy_response.png    β–speed scatter, clean vs noisy
#    tempflow_noise_field.png       the noise itself + its autocorrelation
#    summary.txt                    every number printed below, for the record
#
# -----------------------------------------------------------------------------
#  USAGE
# -----------------------------------------------------------------------------
#    julia run_temperature_noise.jl [n_days] [start_day] [per_day] [σ_T] [τ_h] [seed]
#
#    n_days     length of the run in days                    (default 3)
#    start_day  day-of-year to start from; 200 ≈ annual peak (default 200)
#    per_day    steps per day; 24 = hourly                   (default 24)
#    σ_T        temperature noise sd, °C; 0 disables noise   (default 2.0)
#    τ_h        noise decorrelation time, hours              (default 6.0)
#    seed       RNG seed                                     (default 42)
#
#  Examples
#    julia run_temperature_noise.jl                    # 3 days, hourly, σ=2 °C
#    julia run_temperature_noise.jl 3 200 24 8.0 6 42  # stronger noise
#    julia run_temperature_noise.jl 365 0 1 2.0 24 42  # a full year, daily
#
#  Runs in the DEFAULT Julia environment (needs WAVI, Plots, HDF5-free).
#  No threads required. Roughly one minute for the default settings.
# =============================================================================

using Statistics, Printf, Random, LinearAlgebra
ENV["GKSwstype"] = "100"                    # headless plotting
using WAVI
using Plots, Plots.PlotMeasures

# ── command line ─────────────────────────────────────────────────────────
_arg(i, d) = length(ARGS) >= i ? ARGS[i] : d
const NDAYS   = parse(Float64, _arg(1, "3.0"))
const DAY0    = parse(Float64, _arg(2, "200.0"))
const PERDAY  = parse(Int,     _arg(3, "24"))
const SIGMA_T = parse(Float64, _arg(4, "2.0"))
const TAU_H   = parse(Float64, _arg(5, "6.0"))
const SEED    = parse(Int,     _arg(6, "42"))

const OUTDIR = joinpath(@__DIR__, "results", "noise_temperature_forcing")
mkpath(OUTDIR)

# Everything printed also goes to summary.txt. Call as say(@sprintf(...)) —
# @sprintf needs a literal format string, so it cannot be wrapped in a helper.
const LOG = IOBuffer()
say(s) = (println(s); println(LOG, s))

# =============================================================================
#  1. DOMAIN
# =============================================================================
const NX, NY = 40, 40
const DOMAIN = 160_000.0                 # 160 km square
const DX     = DOMAIN / NX               # 4 km cells

# Surface elevation: a dome sloping only in x, highest at x=0 (the divide).
# clamp keeps the square root defined at/after x = L.
z_s(x) = 1060.0 * sqrt(clamp(1.0 - x / DOMAIN, 0.0, 1.0))

const XS    = [(i - 0.5) * DX for i in 1:NX]     # cell-centre x, metres
const Z_S_X = z_s.(XS)                            # surface elevation along x
const Z_REF = mean(Z_S_X)                         # reference elevation ≈ 707 m

# Interior mask: drop a 3-cell border. The x=0 divide holds ice back (~6 m/yr
# against ~21 interior) over roughly three columns, so domain-wide means are
# reported over the interior to keep boundary effects out of the statistics.
const IN = 4:37

# 16 sensors on a 4×4 grid at x,y ∈ {20, 50, 110, 140} km — the same locations
# the particle-filter experiments use, embedded here so this file needs no
# external data. Stored as (j,i) = (row=y, col=x) index pairs.
const SENSOR_XY_KM = [(x, y) for y in (20, 50, 110, 140) for x in (20, 50, 110, 140)]
const SENSORS = [(clamp(round(Int, y * 1000 / DX) + 1, 1, NY),
                  clamp(round(Int, x * 1000 / DX) + 1, 1, NX)) for (x, y) in SENSOR_XY_KM]

# =============================================================================
#  2. TEMPERATURE FORCING
# =============================================================================
# Seasonal + diurnal + elevation lapse. With Tmean=-10, Aann=15 and Adiur=10 the
# envelope at the reference elevation is +15 / −35 °C, which is a wide swing —
# these amplitudes are inherited from the Flowers-2008-style caricature and are
# not tuned to a specific glacier.
Base.@kwdef struct TempParams
    Tmean::Float64 = -10.0      # annual mean at z_ref, °C
    Aann::Float64  =  15.0      # seasonal amplitude, °C
    Adiur::Float64 =  10.0      # diurnal amplitude, °C
    Gamma::Float64 =   0.0059   # lapse rate, °C/m (5.9 °C/km)
    dpeak::Float64 = 200.0      # day-of-year of the annual peak
    phi::Float64   =   0.6      # diurnal peak phase, as a fraction of a day
end
const TP = TempParams()

"Temperature at the reference elevation (no lapse term), °C. `t` in days."
base_temperature(t; p::TempParams = TP) =
    p.Tmean + p.Aann * cos(2π * (t - p.dpeak) / 365) + p.Adiur * cos(2π * (t - p.phi))

"Elevation correction −Γ·(z_s − z_ref) along x, °C. Colder uphill. Length NX."
const LAPSE = @. -TP.Gamma * (Z_S_X - Z_REF)

"""
    temperature_field(t) -> (NY, NX) matrix, °C

2D temperature at time `t` (days), in the (y, x) layout used throughout so it
can sit alongside a β field without transposing. Uniform in y — the only
spatial structure comes from the elevation lapse, which varies with x alone.
"""
temperature_field(t::Real) =
    repeat(reshape(LAPSE .+ base_temperature(t), 1, NX), NY, 1)

# =============================================================================
#  3. TEMPERATURE → BASAL DRAG
# =============================================================================
# Inverse linear map: warm ⇒ low β ⇒ slippery. β_center is chosen to match the
# centre used by the particle-filter priors, so the two lines of work produce
# comparable β magnitudes (roughly 900–3100 here vs [500, 3500] for double-bump).
Base.@kwdef struct BetaTempParams
    beta_center::Float64 = 2000.0   # β at T = T_center, Pa·s/m
    T_center::Float64    = -10.0    # reference temperature, °C
    k::Float64           =   40.0   # sensitivity, Pa·s/m per °C (positive ⇒ inverse)
    min_beta::Float64    =   10.0   # floor
end
const BTP = BetaTempParams()

beta_from_temperature(T; q::BetaTempParams = BTP) =
    @. max(q.beta_center - q.k * (T - q.T_center), q.min_beta)

# =============================================================================
#  4. SMOOTH NOISE GENERATOR
# =============================================================================
"""
    build_noise_factor(ℓ) -> L

Cholesky factor of a squared-exponential covariance, K[i,j] = exp(−r²/2ℓ²), on
the NX×NY grid flattened in the (y, x) column-major order used everywhere here.
`L*z` with z ~ N(0,I) is then a smooth field with unit variance per cell, so a
noise amplitude in °C can be applied by simple multiplication.

A 1e-8 ridge is added to the diagonal to keep the factorisation well posed.
This is a 1600×1600 dense factorisation — about a second, done once.
"""
function build_noise_factor(len_scale::Float64)
    n = NX * NY
    ℓ2 = len_scale^2
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i2 in 1:NX, j2 in 1:NY
        idx2 = (i2 - 1) * NY + j2
        x2 = (i2 - 1) * DX; y2 = (j2 - 1) * DX
        for i1 in 1:NX, j1 in 1:NY
            idx1 = (i1 - 1) * NY + j1
            x1 = (i1 - 1) * DX; y1 = (j1 - 1) * DX
            K[idx1, idx2] = exp(-((x1 - x2)^2 + (y1 - y2)^2) / (2 * ℓ2))
        end
    end
    @inbounds for i in 1:n; K[i, i] += 1e-8; end
    return Matrix(cholesky(Symmetric(K)).L)
end

# =============================================================================
#  5. WAVI WRAPPER
# =============================================================================
# The grid, bed and initial thickness are built once and reused; only β changes
# between solves. WAVI embeds β in its Params, so a fresh Model is constructed
# per solve — that is WAVI's design, not an inefficiency we introduce.
const U_ISZERO = ["north"]                     # u = 0 at x = 0      → divide wall
const V_ISZERO = ["south", "east", "west"]     # v = 0 at y = 0, y = L → sidewalls
                                               # ("south" is inert here: v ≡ 0 anyway)
const GRID = Grid(nx = NX, ny = NY, dx = DX, dy = DX, x0 = 0.0, y0 = 0.0,
                  u_iszero = U_ISZERO, v_iszero = V_ISZERO)

# Sliding law exponent. WAVI's Weertman law is τ_b = C·|u|^(1/m−1)·u, and it
# internally forms gh.β = weertman_c·|u_bed|^(1/m−1). With m = 1 that exponent
# is zero, the velocity-dependent factor collapses to 1, and gh.β becomes
# IDENTICALLY the field passed in (verified: max|gh.β − weertman_c| = 0). The
# sliding law is then linear/viscous, τ_b = β·u, so speed ≈ driving stress / β.
const WEERTMAN_M = 1.0

const Z_B    = zeros(NX, NY)
const H_INIT = max.(z_s.(GRID.xxh) .- Z_B, 0.0)
const IC     = InitialConditions(initial_thickness = H_INIT)

"""
    wavi_speed(beta_yx) -> (NY, NX) speed in m/yr

Run WAVI on a β field given in the (y, x) layout used by this script, and
return the speed magnitude in the same layout. The transposes convert to and
from WAVI's own (x, y) convention. WAVI's solver chatter is suppressed.
"""
function wavi_speed(beta_yx::AbstractMatrix{<:Real})
    beta_xy = Matrix(transpose(beta_yx))                # (y,x) → (x,y)
    model = Model(grid = GRID, bed_elevation = Z_B, initial_conditions = IC,
                  params = Params(weertman_c = beta_xy, weertman_m = WEERTMAN_M))
    redirect_stdout(devnull) do
        update_state!(model)
    end
    u = transpose(model.fields.gh.u)                    # back to (y,x)
    v = transpose(model.fields.gh.v)
    return sqrt.(u .^ 2 .+ v .^ 2)
end

# =============================================================================
#  6. RUN
# =============================================================================
ts   = collect(DAY0:(1 / PERDAY):(DAY0 + NDAYS))
nt   = length(ts)
dt_h = 24 / PERDAY                       # step length, hours
rho  = exp(-dt_h / TAU_H)                # AR(1) memory per step

say("="^78)
say(" TEMPERATURE-DRIVEN FLOW WITH SMOOTH PROCESS NOISE")
say("="^78)
say(@sprintf(" run        : %d steps over %.1f days from day %.1f (%d steps/day, Δt = %.2f h)",
             nt, NDAYS, DAY0, PERDAY, dt_h))
say(@sprintf(" noise      : σ_T = %.2f °C, τ = %.1f h  ⇒  ρ = %.3f per step, ℓ = 15 km, seed = %d",
             SIGMA_T, TAU_H, rho, SEED))
say(@sprintf(" sliding    : weertman_m = %.1f (linear, τ_b = β·u)", WEERTMAN_M))
say(" boundaries : u_iszero = $U_ISZERO   v_iszero = $V_ISZERO")
say(@sprintf(" β map      : β = %.0f − %.0f·(T − %.0f), floor %.0f Pa·s/m",
             BTP.beta_center, BTP.k, BTP.T_center, BTP.min_beta))
say("")

Lfac = build_noise_factor(15_000.0)
rng  = MersenneTwister(SEED)
smooth_draw() = reshape(Lfac * randn(rng, NX * NY), NY, NX)

Tc = Vector{Matrix{Float64}}(undef, nt)   # clean temperature
Tn = Vector{Matrix{Float64}}(undef, nt)   # noisy temperature
Nf = Vector{Matrix{Float64}}(undef, nt)   # the noise field (unit variance)
Bc = Vector{Matrix{Float64}}(undef, nt)   # β from clean T
Bn = Vector{Matrix{Float64}}(undef, nt)   # β from noisy T
Sc = Vector{Matrix{Float64}}(undef, nt)   # speed from clean β
Sn = Vector{Matrix{Float64}}(undef, nt)   # speed from noisy β

print(" solving  : ")
Nprev = smooth_draw()                      # stationary start (unit variance)
t_start = time()
for n in 1:nt
    # AR(1) update. √(1−ρ²) keeps the stationary variance at exactly 1 so σ_T
    # remains the true sd of the perturbation regardless of τ.
    Nf[n] = n == 1 ? Nprev : (rho .* Nprev .+ sqrt(1 - rho^2) .* smooth_draw())
    global Nprev = Nf[n]

    Tc[n] = temperature_field(ts[n])
    Tn[n] = Tc[n] .+ SIGMA_T .* Nf[n]
    Bc[n] = beta_from_temperature(Tc[n])
    Bn[n] = beta_from_temperature(Tn[n])
    Sc[n] = wavi_speed(Bc[n])
    Sn[n] = wavi_speed(Bn[n])
    n % 12 == 0 && print("$n ")
end
elapsed = time() - t_start
println()
say(@sprintf(" solved   : %d WAVI solves in %.0f s (%.3f s/solve)", 2nt, elapsed, elapsed / 2nt))
say("")

# ── summary statistics (interior means, boundary excluded) ───────────────
im(v) = [mean(x[IN, IN]) for x in v]
Tcm, Tnm = im(Tc), im(Tn)
Bcm, Bnm = im(Bc), im(Bn)
Scm, Snm = im(Sc), im(Sn)
Ssens = [mean([S[j, i] for (j, i) in SENSORS]) for S in Sn]

noise_sd   = std(vcat(vec.(Nf)...)) * SIGMA_T
noise_ac1  = nt > 1 ? cor(vcat(vec.(Nf[1:end-1])...), vcat(vec.(Nf[2:end])...)) : NaN
dep_rms    = sqrt(mean((Snm .- Scm) .^ 2))

say("-"^78)
say(" RESULTS")
say("-"^78)
say(@sprintf(" noise check      : realised sd %.2f °C (target %.2f), lag-1 autocorr %.3f (target %.3f)",
             noise_sd, SIGMA_T, noise_ac1, rho))
say(@sprintf(" temperature      : clean %6.2f … %6.2f °C  |  noisy %6.2f … %6.2f °C",
             minimum(Tcm), maximum(Tcm), minimum(Tnm), maximum(Tnm)))
say(@sprintf(" basal drag β     : clean %6.0f … %6.0f     |  noisy %6.0f … %6.0f Pa·s/m",
             minimum(Bcm), maximum(Bcm), minimum(Bnm), maximum(Bnm)))
say(@sprintf(" ice speed        : clean %6.2f … %6.2f     |  noisy %6.2f … %6.2f m/yr",
             minimum(Scm), maximum(Scm), minimum(Snm), maximum(Snm)))
say(@sprintf(" speed swing      : %.1f%% of mean (diurnal cycle amplitude)",
             100 * (maximum(Scm) - minimum(Scm)) / mean(Scm)))
say("")
say(@sprintf(" corr(β, speed) over TIME (interior means) : clean %+.4f | noisy %+.4f",
             cor(Bcm, Scm), cor(Bnm, Snm)))
say(@sprintf(" corr(β, speed) POOLED over cells & times  : clean %+.4f | noisy %+.4f",
             cor(vcat(vec.(Bc)...), vcat(vec.(Sc)...)), cor(vcat(vec.(Bn)...), vcat(vec.(Sn)...))))
say("   (the pooled value is weaker for BOTH runs — that gap is the spatial")
say("    lapse tilt in β which velocity does not follow linearly, not the noise)")
say("")
say(@sprintf(" noise impact on speed : %.2f m/yr rms = %.1f%% of the clean mean",
             dep_rms, 100 * dep_rms / mean(Scm)))
say("-"^78)

# =============================================================================
#  7. FIGURES
# =============================================================================
# Shared plotting conventions:
#   • colour limits are FIXED across all frames (per-frame autoscaling would
#     fake convergence and make brightness changes meaningless);
#   • 1st–99th percentile rather than raw min/max, so a single outlier cell
#     cannot compress every other frame;
#   • speed uses INTERIOR percentiles — the x=0 divide is slow enough to skew
#     the low end otherwise.
gx = collect(0:NX-1) .* (DX / 1000)
gy = collect(0:NY-1) .* (DX / 1000)
sxs = [(i - 1) * DX / 1000 for (_, i) in SENSORS]
sys = [(j - 1) * DX / 1000 for (j, _) in SENSORS]

qlim(v, lo, hi) = (quantile(vcat(vec.(v)...), lo), quantile(vcat(vec.(v)...), hi))
Tlim = (min(qlim(Tc, 0.01, 0.99)[1], qlim(Tn, 0.01, 0.99)[1]),
        max(qlim(Tc, 0.01, 0.99)[2], qlim(Tn, 0.01, 0.99)[2]))
Blim = (min(qlim(Bc, 0.01, 0.99)[1], qlim(Bn, 0.01, 0.99)[1]),
        max(qlim(Bc, 0.01, 0.99)[2], qlim(Bn, 0.01, 0.99)[2]))
Sint = vcat([vec(S[IN, IN]) for S in Sn]...)
Slim = (quantile(Sint, 0.01), quantile(Sint, 0.99))

pan(f, ttl, lm, cm, cb) =
    heatmap(gx, gy, f, aspect_ratio = 1, color = cm, clims = lm, colorbar_title = cb,
            title = ttl, titlefontsize = 9, xlabel = "x (km)", ylabel = "y (km)")

print(" plotting : evolution gif ")
anim = @animate for n in 1:nt
    p1 = pan(Tc[n], "clean T",                                Tlim, :thermal, "°C")
    p2 = pan(Tn[n], @sprintf("noisy T (σ=%.1f°C, τ=%.0fh)", SIGMA_T, TAU_H),
                                                              Tlim, :thermal, "°C")
    p3 = pan(Bn[n], "β from noisy T",                         Blim, :viridis, "Pa·s/m")
    p4 = pan(Sn[n], "WAVI speed (interior scale)",            Slim, :thermal, "m/yr")
    scatter!(p4, sxs, sys, ms = 2.0, color = :cyan, label = "")
    plot(p1, p2, p3, p4, layout = (1, 4), size = (1900, 470),
         plot_title = @sprintf("day %.2f (hour %d)   T̄ %.1f→%.1f °C   β̄ %.0f   speed %.1f m/yr",
                               ts[n], n - 1, Tcm[n], Tnm[n], Bnm[n], Snm[n]),
         plot_titlefontsize = 11,
         left_margin = 9mm, bottom_margin = 6mm, right_margin = 3mm, top_margin = 2mm)
end
gif(anim, joinpath(OUTDIR, "tempflow_noisy_evolution.gif"), fps = 6)

print("| timeseries ")
hrs = (ts .- DAY0) .* 24
p1 = plot(hrs, Tcm, lw = 2, color = :gray50, ls = :dash, label = "clean",
          ylabel = "T (°C)", title = "Interior-mean surface temperature", legend = :best)
plot!(p1, hrs, Tnm, lw = 2, color = :orangered, label = "noisy")
p2 = plot(hrs, Bcm, lw = 2, color = :gray50, ls = :dash, label = "clean",
          ylabel = "β (Pa·s/m)", title = "β from T  (inverted: warm ⇒ low β)", legend = false)
plot!(p2, hrs, Bnm, lw = 2, color = :purple, label = "noisy")
p3 = plot(hrs, Scm, lw = 2, color = :gray50, ls = :dash, label = "clean",
          ylabel = "speed (m/yr)", xlabel = @sprintf("hours from day %d", Int(DAY0)),
          title = "WAVI speed  (follows T, opposes β)", legend = :best)
plot!(p3, hrs, Snm, lw = 2, color = :steelblue, label = "noisy (interior)")
plot!(p3, hrs, Ssens, lw = 1.5, ls = :dot, color = :seagreen, label = "noisy (16 sensors)")
plot(p1, p2, p3, layout = (3, 1), size = (950, 850), left_margin = 10mm, bottom_margin = 5mm)
savefig(joinpath(OUTDIR, "tempflow_noisy_timeseries.png"))

print("| response ")
scatter(Bcm, Scm, ms = 5, msw = 0, color = :gray60,
        label = @sprintf("clean (r = %+.3f)", cor(Bcm, Scm)),
        xlabel = "interior-mean β (Pa·s/m)", ylabel = "interior-mean speed (m/yr)",
        title = "β–speed response: does the noise break the relationship?",
        size = (820, 620), legend = :topright, left_margin = 6mm, bottom_margin = 5mm)
scatter!(Bnm, Snm, ms = 5, msw = 0, color = :crimson,
         label = @sprintf("noisy (r = %+.3f)", cor(Bnm, Snm)))
savefig(joinpath(OUTDIR, "tempflow_noisy_response.png"))

print("| noise field ")
k = max(1, nt ÷ 4)
snaps = [heatmap(gx, gy, SIGMA_T .* Nf[i], aspect_ratio = 1, color = :balance,
                 clims = (-3 * max(SIGMA_T, eps()), 3 * max(SIGMA_T, eps())),
                 title = @sprintf("hour %d", i - 1), titlefontsize = 9,
                 colorbar_title = "°C") for i in unique(clamp.((1, k, 2k, 3k), 1, nt))]
lags = 0:min(24, nt - 1)
ac = [cor(vcat(vec.(Nf[1:end-l])...), vcat(vec.(Nf[1+l:end])...)) for l in lags]
pac = plot(collect(lags) .* dt_h, ac, lw = 2, marker = :circle, ms = 3,
           color = :darkgreen, label = "measured",
           xlabel = "lag (hours)", ylabel = "autocorrelation",
           title = @sprintf("temporal autocorrelation of the noise (τ = %.1f h)", TAU_H))
plot!(pac, collect(lags) .* dt_h, exp.(-(collect(lags) .* dt_h) ./ TAU_H),
      lw = 1.5, ls = :dash, color = :black, label = "exp(−lag/τ)")
plot(snaps..., pac, layout = @layout([grid(1, length(snaps)); a]),
     size = (1600, 760), left_margin = 8mm, bottom_margin = 6mm)
savefig(joinpath(OUTDIR, "tempflow_noise_field.png"))
println("| done")

open(joinpath(OUTDIR, "summary.txt"), "w") do f
    write(f, String(take!(LOG)))
end

say("")
say(" all outputs → $OUTDIR")
for fn in sort(readdir(OUTDIR)); println("   $fn"); end
