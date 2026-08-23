# Temperature forcing over a full season — SETUP / validation
# Julia port of the MATLAB setup script (Flowers 2008, Fig 2b).
#
# Builds a surface-air-temperature field T(x, y, t) on the SAME glacier grid the
# particle-filter experiments use (40×40 over 160 km, surface z_s(x)=1060√(1−x/L)),
# combining three pieces:
#
#   T(x, t) = Tmean + Aann·cos(2π(t−dpeak)/365)   annual (seasonal) cycle, period 365 d
#                   + Adiur·cos(2π(t−φ))          diurnal cycle,          period 1 d
#                   − Γ·(z_s(x) − z_ref)          elevation lapse-rate correction
#
# The field is x-banded and uniform in y, because the surface elevation profile
# z_s depends only on x — so temperature inherits that symmetry. Higher ground
# (small x) is COLDER: the lapse term is negative where z_s > z_ref.
#
# Time `t` is in DAYS (hourly steps, 1/24 d) so the diurnal cycle is resolved.
# Note this is a *different* clock from the particle filter's `time_step=3600 s`
# model hour — nothing here is wired into the PF yet; this script is standalone
# setup/validation, exactly like the MATLAB it replaces.
#
# RUN (no WAVI, no ParticleDA needed — plain default env, no threads):
#   julia glacier-code/particleda/temperature_graph.jl
#
# Outputs to glacier-code/particleda/results/temperature_forcing/ :
#   temperature_mean_elevation.png  — base series at mean elevation (Fig 2b) + 5-day zoom
#   temperature_snapshot.png        — 2D T(x,y) map at a warm afternoon (day 200.6)
#   temperature_hovmoller.png       — x-vs-time section showing season + lapse together
#   beta_temperature_relation.png   — the linear β(T) map + mirrored annual series
#   beta_snapshot.png               — T and implied β at the warmest / coldest instants
#   beta_hovmoller.png              — x-vs-time β field implied by the temperature forcing
#
# The functions are also `include`-able: the plotting block at the bottom only
# runs when this file is executed directly, so other scripts can do
#   include("temperature_graph.jl"); T = temperature_field(200.6)
# and get the (ny, nx) field in the Glacier reshape convention.

using Statistics, Printf
ENV["GKSwstype"] = "100"                            # headless GR
using Plots, Plots.PlotMeasures                     # top-level: `@layout` must be
                                                    # resolvable when the plotting
                                                    # block below is macroexpanded

# ── Grid (matches ice_flow.jl / the PF setup) ────────────────────────────
# Cell CENTRES, not MATLAB's linspace(0,L,N) endpoints: the WAVI/Glacier grid
# is a finite-volume grid with dx = L/nx, so centres sit at (i−½)·dx. This
# shifts x by 2 km relative to the MATLAB script and avoids the x=L cell where
# z_s collapses to exactly 0.
const NX = 40
const NY = 40
const L  = 160_000.0
const DX = L / NX
const XS = [(i - 0.5) * DX for i in 1:NX]          # length NX, metres
const YS = [(j - 0.5) * DX for j in 1:NY]          # length NY, metres (dy = dx)

# Surface elevation, m. Same profile as ice_flow.jl; clamped so the √ stays real.
z_s(x) = 1060.0 * sqrt(clamp(1.0 - x / L, 0.0, 1.0))

const Z_S   = z_s.(XS)                              # length NX
const Z_REF = mean(Z_S)                             # ≈ 707 m (continuous mean = 1060·2/3)

# ── Forcing parameters ───────────────────────────────────────────────────
# NOTE (carried over from the MATLAB "<-- verify sign" comment): with
# Tmean=-10 and Aann=15 the annual maximum at the reference elevation reaches
# +5 °C and the minimum -25 °C; adding the diurnal ±10 gives a +15/-35 °C
# envelope. That is a wide swing — check Aann/Adiur against Flowers 2008 before
# using this to drive anything physical.
Base.@kwdef struct TempParams
    Tmean::Float64 = -10.0      # annual mean temperature at z_ref, °C
    Aann::Float64  = 15.0       # annual (seasonal) amplitude, °C
    Adiur::Float64 = 10.0       # diurnal amplitude, °C
    Gamma::Float64 = 0.0059     # lapse rate, °C/m  (5.9 °C/km)
    dpeak::Float64 = 200.0      # day of the annual peak
    phi::Float64   = 0.6        # diurnal peak phase, as a day fraction
end

const TP = TempParams()

"""
    base_temperature(t; p=TP)

Temperature at the reference elevation `Z_REF` (i.e. no lapse term), °C.
`t` in days; scalar or array. This is the series plotted in Flowers Fig 2(b).
"""
base_temperature(t; p::TempParams = TP) =
    @. p.Tmean + p.Aann * cos(2π * (t - p.dpeak) / 365) + p.Adiur * cos(2π * (t - p.phi))

"""
    lapse_correction(; p=TP)

Elevation correction `−Γ·(z_s − z_ref)` along x, °C. Length `NX`, time-independent.
Negative where the surface is above the reference elevation (colder uphill).
"""
lapse_correction(; p::TempParams = TP) = @. -p.Gamma * (Z_S - Z_REF)

"""
    temperature_profile(t; p=TP) -> Vector (length NX)

Temperature along x at a single time `t` (days), °C.
"""
temperature_profile(t::Real; p::TempParams = TP) =
    lapse_correction(p = p) .+ base_temperature(t, p = p)

"""
    temperature_field(t; p=TP) -> Matrix (NY, NX)

2D temperature map at time `t` (days), °C, in the **Glacier reshape convention**
`(ny, nx)` — the same layout as `reshape(state, ny, nx)` in the PF code, so it
can be dropped alongside a β field without transposing. Uniform in y.
"""
function temperature_field(t::Real; p::TempParams = TP)
    prof = temperature_profile(t, p = p)             # length NX
    return repeat(reshape(prof, 1, NX), NY, 1)       # (NY, NX), x-banded
end

"""
    temperature_matrix(t; p=TP) -> Matrix (NX, length(t))

The full space–time array: rows = x, columns = time. Direct equivalent of the
MATLAB `T = lapse.' + (Tseason + Tdiur)`.
"""
temperature_matrix(t::AbstractVector; p::TempParams = TP) =
    lapse_correction(p = p) .+ reshape(base_temperature(t, p = p), 1, :)

# ── Linear temperature → basal drag (β) mapping ──────────────────────────
# A deliberately simple, INVERSE linear caricature:
#
#     β(T) = β_center − k·(T − T_center),   clamped at min_beta
#
# so warm ⇒ low β (slippery bed) and cold ⇒ high β (sticky bed). The physical
# story it stands in for: warmer surface/near-surface temperatures ⇒ more melt
# ⇒ more water reaching the bed ⇒ lubrication ⇒ less basal drag. Real
# temperature–drag coupling is nonlinear, hysteretic and lagged (the drainage
# system evolves over days-to-weeks); this is a first-cut linear stand-in for
# setup and validation only, exactly like the temperature block above.
#
# Anchors are chosen to sit on the existing PF setup:
#   beta_center = 2000  — the prior centre used throughout the Exp-18 runs
#   T_center    = Tmean = −10 °C — so the ANNUAL, MEAN-ELEVATION temperature
#                 maps exactly onto the prior centre; departures move β either way
#   k = 40 Pa·s/m per °C — the observed T spread (≈ −37 … +18 °C) then maps to
#                 β ≈ 900 … 3100, comparable to the double-bump range [500, 3500]
#   min_beta = 10 — the same floor the dynamics clamp to
Base.@kwdef struct BetaTempParams
    beta_center::Float64 = 2000.0   # β at T = T_center, Pa·s/m
    T_center::Float64    = -10.0    # reference temperature, °C (= TP.Tmean)
    k::Float64           = 40.0     # sensitivity, Pa·s/m per °C (POSITIVE ⇒ inverse relation)
    min_beta::Float64    = 10.0     # floor, matches ice_experiment_dynamics.jl
end

const BTP = BetaTempParams()

"""
    beta_from_temperature(T; q=BTP)

Linear inverse map from temperature (°C) to basal drag β (Pa·s/m), clamped
below at `q.min_beta`. Works on scalars or arrays. High T → low β.
"""
beta_from_temperature(T; q::BetaTempParams = BTP) =
    @. max(q.beta_center - q.k * (T - q.T_center), q.min_beta)

"""
    beta_field_from_temperature(t; p=TP, q=BTP) -> Matrix (NY, NX)

β map at time `t` (days) implied by the full temperature forcing — seasonal +
diurnal + elevation lapse. `(ny, nx)` Glacier reshape convention, uniform in y.
"""
beta_field_from_temperature(t::Real; p::TempParams = TP, q::BetaTempParams = BTP) =
    beta_from_temperature(temperature_field(t, p = p), q = q)

# ── Plots (only when run as a script, not when included) ─────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
    const OUT = joinpath(REPO_ROOT, "glacier-code", "particleda",
                         "results", "temperature_forcing")
    mkpath(OUT)

    t = collect(0:(1 / 24):365)                     # days, hourly → 8761 samples
    Tbase = base_temperature(t)
    lapse = lapse_correction()

    @printf("Grid: %d×%d over %.0f km (dx = %.1f km)\n", NX, NY, L / 1e3, DX / 1e3)
    @printf("z_s range: %.1f – %.1f m,  z_ref = %.1f m\n", minimum(Z_S), maximum(Z_S), Z_REF)
    @printf("Lapse correction range: %+.2f – %+.2f °C\n", minimum(lapse), maximum(lapse))
    @printf("Base series (mean elevation): min %.2f, max %.2f, mean %.2f °C\n",
            minimum(Tbase), maximum(Tbase), mean(Tbase))

    # (1) Fig 2(b) reproduction: the base series at mean elevation.
    # Over 365 days the diurnal cycle is a solid band, so pair the full year
    # with a 5-day zoom that actually resolves it.
    p1 = plot(t, Tbase, lw = 0.4, color = :steelblue, legend = false,
              xlabel = "Day", ylabel = "T (°C)",
              title = "Mean elevation — compare to Flowers Fig 2(b)")
    plot!(p1, t, TP.Tmean .+ TP.Aann .* cos.(2π .* (t .- TP.dpeak) ./ 365),
          lw = 2, color = :firebrick)               # seasonal envelope centre

    zoom = (t .>= 198) .& (t .<= 203)
    p2 = plot(t[zoom], Tbase[zoom], lw = 1.6, color = :steelblue, legend = false,
              xlabel = "Day", ylabel = "T (°C)",
              title = "5-day zoom (diurnal cycle, peak phase φ = $(TP.phi) d)")

    plot(p1, p2, layout = (2, 1), size = (900, 700),
         left_margin = 6mm, bottom_margin = 5mm)
    savefig(joinpath(OUT, "temperature_mean_elevation.png"))

    # (2) 2D spatial snapshot at one instant (warm afternoon, day 200.6).
    tsnap = t[argmin(abs.(t .- 200.6))]
    Tfield = temperature_field(tsnap)               # (NY, NX)
    heatmap(XS ./ 1e3, YS ./ 1e3, Tfield,
            aspect_ratio = 1, color = :thermal, colorbar_title = "T (°C)",
            xlabel = "x (km)", ylabel = "y (km)",
            title = @sprintf("T snapshot, day %.2f (°C)", tsnap),
            size = (760, 640), right_margin = 8mm, left_margin = 4mm)
    savefig(joinpath(OUT, "temperature_snapshot.png"))
    @printf("Snapshot day %.2f: T range %.2f – %.2f °C (x-banded, uniform in y)\n",
            tsnap, minimum(Tfield), maximum(Tfield))

    # (3) Hovmöller: x vs time over the year, showing the seasonal swing and the
    # (much smaller) lapse-rate tilt in one picture. Uses the DAILY MEAN of the
    # hourly series — sampling at integer days instead would freeze the diurnal
    # term at one phase (a constant ≈ −8 °C offset), not remove it.
    tdays = collect(0.0:1.0:364.0)
    Thov = reduce(hcat, [vec(mean(temperature_matrix(collect(d:(1 / 24):(d + 1 - 1 / 24)));
                                 dims = 2)) for d in tdays])   # (NX, ndays)
    heatmap(tdays, XS ./ 1e3, Thov,
            color = :thermal, colorbar_title = "T (°C)",
            xlabel = "Day", ylabel = "x (km)",
            title = "T(x, t) — seasonal cycle with elevation lapse",
            size = (900, 520), right_margin = 8mm, left_margin = 4mm)
    savefig(joinpath(OUT, "temperature_hovmoller.png"))

    # ── β(T) figures ─────────────────────────────────────────────────────
    Tall = temperature_matrix(t)                    # (NX, nt), every x and hour
    Ball = beta_from_temperature(Tall)
    @printf("β from T: range %.1f – %.1f Pa·s/m (clamped at %.0f: %s)\n",
            minimum(Ball), maximum(Ball), BTP.min_beta,
            any(Ball .<= BTP.min_beta) ? "yes" : "no")
    @printf("corr(T, β) over the whole year/domain = %+.4f\n",
            cor(vec(Tall), vec(Ball)))

    # (4) The mapping itself + the mirrored annual series at mean elevation.
    Tline = range(minimum(Tall), maximum(Tall), length = 200)
    p4a = plot(Tline, beta_from_temperature(Tline), lw = 2.5, color = :purple,
               legend = false, xlabel = "T (°C)", ylabel = "β (Pa·s/m)",
               # sign-aware so a negative T_center reads "(T + 10)", not "(T − -10)"
               title = @sprintf("β(T) = %.0f − %.0f·(T %s %.0f), floor %.0f",
                                BTP.beta_center, BTP.k,
                                BTP.T_center < 0 ? "+" : "−", abs(BTP.T_center),
                                BTP.min_beta))
    scatter!(p4a, [BTP.T_center], [BTP.beta_center], ms = 6, color = :black)
    annotate!(p4a, BTP.T_center, BTP.beta_center + 250,
              text("anchor: annual mean T → prior centre β", 8, :left))

    # Mirrored time series: T up ⇒ β down, by construction.
    Bbase = beta_from_temperature(Tbase)
    p4b = plot(t, Tbase, lw = 0.4, color = :steelblue, label = "T (°C)",
               xlabel = "Day", ylabel = "T (°C)", legend = :topleft)
    plot!(twinx(), t, Bbase, lw = 0.4, color = :purple, label = "β (Pa·s/m)",
          ylabel = "β (Pa·s/m)", legend = :topright)
    title!(p4b, "Mean elevation: warm ⇒ slippery (low β), cold ⇒ sticky (high β)")

    plot(p4a, p4b, layout = (2, 1), size = (900, 720),
         left_margin = 8mm, right_margin = 12mm, bottom_margin = 5mm)
    savefig(joinpath(OUT, "beta_temperature_relation.png"))

    # (5) Paired snapshots — warmest and coldest instants of the year, each
    # shown as T and the β it implies. The β maps are the colour-reverse of
    # the T maps: that IS the point of the figure.
    iwarm = argmax(Tbase); icold = argmin(Tbase)
    twarm, tcold = t[iwarm], t[icold]
    Twarm, Tcold = temperature_field(twarm), temperature_field(tcold)
    Bwarm, Bcold = beta_field_from_temperature(twarm), beta_field_from_temperature(tcold)

    # SHARED colour limits per variable (T rows share one scale, β rows another),
    # padded 5% so both extremes sit inside the colourbar. This makes the
    # warm-vs-cold contrast read instantly — warm T panel is at the top of the
    # scale, cold at the bottom, and the β panels are exactly reversed.
    # Cost: the lapse-rate gradient (±3.5 °C / ±140 Pa·s/m) is ~7× smaller than
    # the seasonal+diurnal swing, so it is nearly invisible *within* a panel at
    # this scale — the two cross-section panels at the bottom carry that detail.
    padlims(v, frac = 0.05) = let (lo, hi) = extrema(v), d = frac * (hi - lo)
        (lo - d, hi + d)
    end
    Tlim = padlims(vcat(vec(Twarm), vec(Tcold)))
    Blim = padlims(vcat(vec(Bwarm), vec(Bcold)))

    hm(f, ttl, lims, cmap, cbtitle) =
        heatmap(XS ./ 1e3, YS ./ 1e3, f, aspect_ratio = 1, color = cmap,
                clims = lims, colorbar_title = cbtitle, title = ttl,
                xlabel = "x (km)", ylabel = "y (km)", titlefontsize = 9)

    xs_km = XS ./ 1e3
    labs = [@sprintf("warmest (day %.2f)", twarm) @sprintf("coldest (day %.2f)", tcold)]
    pT = plot(xs_km, [vec(Twarm[1, :]) vec(Tcold[1, :])], lw = 2.5,
              color = [:orangered :navy], label = labs, legend = :right,
              xlabel = "x (km)", ylabel = "T (°C)", titlefontsize = 9,
              title = "T along x — seasonal+diurnal offset (gap), lapse (slope)")
    pB = plot(xs_km, [vec(Bwarm[1, :]) vec(Bcold[1, :])], lw = 2.5,
              color = [:orangered :navy], label = labs, legend = :right,
              xlabel = "x (km)", ylabel = "β (Pa·s/m)", titlefontsize = 9,
              title = "β along x — inverted: warm is low, cold is high")
    hline!(pB, [BTP.beta_center], ls = :dash, color = :gray, label = "prior centre 2000")

    plot(hm(Twarm, @sprintf("T — warmest, day %.2f", twarm), Tlim, :thermal, "T (°C)"),
         hm(Bwarm, @sprintf("β — warmest, day %.2f (slippery)", twarm), Blim, :viridis, "β (Pa·s/m)"),
         hm(Tcold, @sprintf("T — coldest, day %.2f", tcold), Tlim, :thermal, "T (°C)"),
         hm(Bcold, @sprintf("β — coldest, day %.2f (sticky)", tcold), Blim, :viridis, "β (Pa·s/m)"),
         pT, pB,
         layout = @layout([a b; c d; e{0.26h} f]), size = (1100, 1150),
         right_margin = 8mm, left_margin = 6mm, bottom_margin = 4mm)
    savefig(joinpath(OUT, "beta_snapshot.png"))
    @printf("Shared colour limits — T: %.2f … %.2f °C,  β: %.1f … %.1f Pa·s/m\n",
            Tlim..., Blim...)
    @printf("Warmest day %.2f: T %.2f–%.2f °C → β %.1f–%.1f Pa·s/m\n",
            twarm, minimum(Twarm), maximum(Twarm), minimum(Bwarm), maximum(Bwarm))
    @printf("Coldest day %.2f: T %.2f–%.2f °C → β %.1f–%.1f Pa·s/m\n",
            tcold, minimum(Tcold), maximum(Tcold), minimum(Bcold), maximum(Bcold))

    # (6) β Hovmöller — same daily-mean construction as the T one, so the two
    # can be read side by side (identical structure, inverted colour sense).
    Bhov = beta_from_temperature(Thov)
    heatmap(tdays, XS ./ 1e3, Bhov,
            color = :viridis, colorbar_title = "β (Pa·s/m)",
            xlabel = "Day", ylabel = "x (km)",
            title = "β(x, t) implied by T — seasonal cycle with elevation lapse",
            size = (900, 520), right_margin = 8mm, left_margin = 4mm)
    savefig(joinpath(OUT, "beta_hovmoller.png"))

    println("Wrote 6 figures to $OUT")
end
