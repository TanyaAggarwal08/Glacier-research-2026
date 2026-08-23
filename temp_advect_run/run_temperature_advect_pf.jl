#!/usr/bin/env julia
# =============================================================================
#  IMPROVING THE β FIELD FROM VELOCITY OBSERVATIONS
#  — temperature drives BOTH the drag and the transport that carries it
# =============================================================================
#
#  ONE FILE, ONE COMMAND, ALL RESULTS:
#
#      julia -t 10 run_temperature_advect_pf.jl
#
#  Everything lands in  results/temperature_advect/
#
# -----------------------------------------------------------------------------
#  WHAT THIS IS, AND WHY
# -----------------------------------------------------------------------------
#  Third in a series. The question each run answers:
#
#    double_bump_run   advection with velocity (1 + ε·β)          →  62%
#    temp_field_run    temperature increment only, NO transport    →  11.8%
#    THIS RUN          temperature drives drag AND transport       →  ?
#
#  temp_field_run failed for a structural reason worth restating, because this
#  run is built specifically to fix it. There, temperature entered as an ADDED
#  NUMBER:
#
#      βₜ₊₁ = βₜ − k·(Tₜ₊₁ − Tₜ) + noise
#
#  Subtract the truth from any particle and that term cancels exactly:
#
#      (βₚ − β_true)ₜ₊₁ = (βₚ − β_true)ₜ + (ηₚ − η_true)
#
#  The error field is then a pure random walk — it never moves, never deforms.
#  Measured: after a full diurnal cycle the error was still 85% the same picture
#  it started as, and 16 fixed sensors spent 24 hours re-asking the same
#  unanswerable question about 1600 static cells. Result: 11.8%.
#
#  Here temperature ALSO enters as TRANSPORT:
#
#      βₜ₊₁(x) = βₜ(x − v·dt) − k·(Tₜ₊₁ − Tₜ) + noise
#
#  and now the error does NOT cancel — it is carried:
#
#      errorₜ₊₁(x) ≈ errorₜ(x − v·dt)
#
#  Each sensor stops being a point and becomes a TRACK. Over the run, material
#  from the whole domain passes beneath it. That is how 16 sensors managed 1600
#  cells in double_bump_run.
#
# -----------------------------------------------------------------------------
#  WHERE THE VELOCITY COMES FROM
# -----------------------------------------------------------------------------
#  double_bump_run used v = 1 + ε·β: high drag migrates faster. That is backwards
#  as physics and the constant ε was invented. Here the velocity follows from
#  something already in the model — a slippery bed transports its own pattern
#  faster:
#
#      v(x,y) = V_SCALE / β(x,y)          clamped to [V_MIN, V_MAX]
#
#  and β is set by temperature through the known map, so warm ⇒ low β ⇒ fast.
#  Temperature therefore drives the drag AND the rate at which the drag pattern
#  moves. One mechanism, two effects.
#
#  This is STATE-DEPENDENT: each particle advects at a rate set by its OWN β, so
#  particles with different β deform their errors differently. That is the extra
#  ingredient over a shared velocity field, and it is what double_bump_run had.
#
#  CALIBRATION. V_SCALE is set so the mean velocity matches double_bump_run's
#  2 m/s exactly, making this a controlled comparison rather than a new
#  free parameter. See the V_SCALE definition for the arithmetic.
#
#  HONEST LABEL: 2 m/s is ~10⁸ times faster than real ice, so this is a numerical
#  device for moving the bed pattern, not glaciology — exactly the same status as
#  double_bump_run's advection, whose own comment called it "a crude stand-in for
#  a bed whose properties evolve". It is kept identical so the comparison is fair.
#
# -----------------------------------------------------------------------------
#  WHAT THIS STILL CANNOT FIX
# -----------------------------------------------------------------------------
#  Advection is in +x only, so rows of y containing no sensor never pass beneath
#  one however fast the flow. double_bump_run had exactly the same limitation and
#  still reached 62%, because the 15 km noise correlation length bridges 3-4 rows
#  and the 4 sensor rows therefore cover most of the domain in y.
#
#  What temperature also does, in both runs, is modulate OBSERVABILITY. With
#  weertman_m = 1, speed ≈ driving stress / β, so
#
#      ∂ log(speed) / ∂β  =  −1/β
#
#  A given β error is more visible when β is small. Warm hours (slippery bed,
#  low β) therefore carry more information per sensor than cold hours. The
#  diurnal cycle sweeps β across roughly 800…2000 Pa·s/m, a 2.5× swing in
#  sensitivity — and, here, the same 2.5× swing in transport speed.
#
#  The question this run answers:
#      is TRANSPORT the missing ingredient? Does carrying the error field past
#      fixed sensors recover the gap between 11.8% and 62%?
#
#  Four diagnostics answer it directly — see DIAGNOSTICS below.
#
# -----------------------------------------------------------------------------
#  DIAGNOSTICS  (the point of this run)
# -----------------------------------------------------------------------------
#  1. RMSE NEAR vs FAR from sensors. Cells within 10 km of a sensor are ones an
#     observation genuinely speaks about; the rest are not. In temp_field_run the
#     split was 17.5% vs 9.5% — the filter worked where it could see, and 79% of
#     the domain was somewhere it could not. If transport works, the FAR number
#     should improve most, because far cells now get carried into view.
#
#  2. PATTERN CORRELATION of the error field at time t with the error field at
#     t = 0. THE KEY MEASUREMENT. In temp_field_run it stayed at 0.850 — the
#     error never moved. If transport works this should collapse toward zero.
#
#  3. OBSERVABILITY, 1/β̄ against time, so any improvement can be checked against
#     the warm part of the cycle.
#
#  4. TRANSPORT DISTANCE actually achieved, per step and cumulative, plus how
#     often the velocity clamp binds. Without this the run cannot be interpreted:
#     if β travelled 2 km the experiment tested nothing.
#
# -----------------------------------------------------------------------------
#  BOUNDARY CONDITIONS  (the most confusing part of WAVI)
# -----------------------------------------------------------------------------
#  WAVI's orientation names are ARRAY INDICES, not compass directions:
#      "north" → A[1,:]   = x = 0        "south" → A[end,:] = x = L
#      "west"  → A[:,1]   = y = 0        "east"  → A[:,end] = y = L
#  north/south are the X edges, east/west the Y edges. For any edge, zero the
#  component that CROSSES it: u for x-edges, v for y-edges.
#
#      u_iszero = ["north"]                → wall at x = 0 (ice divide)
#      v_iszero = ["south","east","west"]  → free-slip sidewalls at y = 0, y = L
#
#  Leaving the y-edges unconstrained makes them calving fronts: ~690 m/yr against
#  an interior of ~20 (32×), which buries the β signal under the colour scale.
#
# -----------------------------------------------------------------------------
#  USAGE
# -----------------------------------------------------------------------------
#    julia -t <threads> run_temperature_advect_pf.jl [N] [T] [K] [σ_obs] [seed_pf] [seed_obs]
#
#    N        particles                        (default 500)
#    T        hourly assimilation steps        (default 24 = one diurnal cycle)
#    K        tempering stages                 (default 10)
#    σ_obs    observation sd, log-speed units  (default 0.5)
#    seed_pf  filter RNG seed                  (default 42)
#    seed_obs truth + observation-noise seed   (default 123)
#
#    julia -t 4  run_temperature_advect_pf.jl 20 4 3    # smoke test, ~30 s
#    julia -t 10 run_temperature_advect_pf.jl           # standard run
#
#  Threads matter — the N likelihood evaluations per stage are independent WAVI
#  solves, spawned in parallel. Use -t 8..10.
# =============================================================================

using Statistics, Printf, Random, LinearAlgebra, HDF5
ENV["GKSwstype"] = "100"
using WAVI
using Plots, Plots.PlotMeasures

_arg(i, d) = length(ARGS) >= i ? ARGS[i] : d
const NPRT      = parse(Int,     _arg(1, "500"))
const NSTEP     = parse(Int,     _arg(2, "24"))
const K_TEMPER  = parse(Int,     _arg(3, "10"))
const SIGMA_OBS = parse(Float64, _arg(4, "0.5"))
const SEED_PF   = parse(Int,     _arg(5, "42"))
const SEED_OBS  = parse(Int,     _arg(6, "123"))

const OUTDIR = joinpath(@__DIR__, "results", "temperature_advect")
mkpath(OUTDIR)
const LOG = IOBuffer()
say(s) = (println(s); flush(stdout); println(LOG, s))

# =============================================================================
#  1. DOMAIN
# =============================================================================
const NX, NY  = 40, 40
const NSTATE  = NX * NY            # the state IS the β field — 1600 cells
const DOMAIN  = 160_000.0          # 160 km square
const DX      = DOMAIN / NX        # 4 km cells
const IN      = 4:37               # interior mask (drop the 3-cell border)

z_s(x) = 1060.0 * sqrt(clamp(1.0 - x / DOMAIN, 0.0, 1.0))
const XS    = [(i - 0.5) * DX for i in 1:NX]
const Z_S_X = z_s.(XS)
const Z_REF = mean(Z_S_X)

# 16 sensors on a 4×4 grid at x,y ∈ {20,50,110,140} km, embedded so this file
# needs no external data. All sit at x ≥ 20 km, clear of the x = 0 divide.
const SENSOR_XY_KM = [(x, y) for y in (20, 50, 110, 140) for x in (20, 50, 110, 140)]
const SENSORS = let idx = Int[]
    for (x, y) in SENSOR_XY_KM
        i = clamp(round(Int, x * 1000 / DX) + 1, 1, NX)      # column (x)
        j = clamp(round(Int, y * 1000 / DX) + 1, 1, NY)      # row    (y)
        push!(idx, (i - 1) * NY + j)                          # flat, (y,x) col-major
    end
    unique(idx)
end
const NOBS = length(SENSORS)

# =============================================================================
#  2. TEMPERATURE FORCING  (known to everyone — identical to temp_pf_run)
# =============================================================================
const T_MEAN  = -10.0     # annual mean at reference elevation, °C
const T_AANN  =  15.0     # seasonal amplitude, °C
const T_ADIUR =  10.0     # diurnal amplitude, °C — this is what drives β
const T_GAMMA =   0.0059  # lapse rate, °C/m
const T_DPEAK = 200.0     # day-of-year of the annual peak
const T_PHI   =   0.6     # diurnal phase, fraction of a day
const DAY0    = 200.0     # start day (near the annual peak, so ice is mobile)

const LAPSE = @. -T_GAMMA * (Z_S_X - Z_REF)     # length NX, time independent

base_temperature(t) =
    T_MEAN + T_AANN * cos(2π * (t - T_DPEAK) / 365) + T_ADIUR * cos(2π * (t - T_PHI))

"Clean temperature field at day `t`, (NY, NX), °C. Uniform in y."
clean_temperature(t::Real) = repeat(reshape(LAPSE .+ base_temperature(t), 1, NX), NY, 1)

# Smooth AR(1) wobble on top, so the forcing looks like a real record rather than
# a clean sinusoid. SHARED by truth and every particle — realistic forcing, not
# an error source. All uncertainty lives in β.
const T_NOISE_SD  = 1.0      # °C
const T_NOISE_TAU = 6.0      # hours

# =============================================================================
#  3. THE β(T) MAP  —  KNOWN, not estimated
# =============================================================================
# This is the whole difference from temp_pf_run. There C and k were unknown and
# the filter recovered them (to 1.0% and 0.6%). Here they are given, and the
# filter's only job is the field.
const C_MAP    = 2000.0      # β at the reference temperature, Pa·s/m
const K_MAP    =   40.0      # sensitivity, Pa·s/m per °C (warm ⇒ slippery)
const T_REF    =  -10.0      # reference temperature, °C
const MIN_BETA =   10.0      # floor on β

# =============================================================================
#  4. BACKGROUND, ENSEMBLE AND NOISE  (all matched to double_bump_run)
# =============================================================================
# background = truth + 300·(smooth random wave): ONE wrong field, shared by every
# particle, exactly as in the double-bump and pseudo-random-wave experiments.
# This is the systematic error the filter has to remove.
const BACKGROUND_STD  = 300.0
const BACKGROUND_SEED =  29
const MAX_WAVENUMBER  =   2

# Each particle is then a DIFFERENT guess around that background — this is the
# ensemble spread the filter explores with. Without it every particle is
# identical and there is nothing to select between.
const INIT_STD    = 200.0                 # spread of the initial ensemble
const PROCESS_STD =  10.0                 # process noise per step
const NOISE_LEN   = 15_000.0              # noise correlation length, m

const ESS_THRESHOLD = 0.5 * NPRT          # resample below N/2
const SIGMA_JITTER  = 20.0                # post-resample jitter (regularised PF)

# --- TRANSPORT: the whole point of this run ------------------------------
# A slippery bed carries its own pattern faster, so v = V_SCALE / β. Since
# temperature sets β through the known map, temperature also sets the transport
# rate: warm ⇒ low β ⇒ fast.
#
# CALIBRATION. double_bump_run used v = 1 + 5e-4·β, giving v ≈ 2.0 m/s at its
# mean β of 2000 — i.e. 2·3600 = 7.2 km per assimilation step, 1.8 cells, and
# ~173 km over 24 steps, more than the 160 km domain. To match that exactly:
#
#     V_SCALE = 2.0 m/s × (mean β here)
#
# The interior-mean β here swings 1020 (warm) to 1796 (cold), mean ≈ 1400, so
# V_SCALE = 2800. That reproduces double_bump_run's transport rate rather than
# introducing a new free parameter.
const V_SCALE = 2800.0                    # m/s × Pa·s/m  ⇒  v = V_SCALE/β in m/s

# Safety clamp. v = V_SCALE/β blows up if a particle's β approaches the MIN_BETA
# floor, and upwind advection is only stable while the Courant number
# v·dt_sub/DX stays below 1. With dt_sub = 360 s and DX = 4000 m, V_MAX = 5.0
# gives Courant 0.45 — comfortably stable. Over the plausible β range (truth
# 798-2007, particles ±500) the clamp corresponds to β < 560 or β > 5600, so it
# should almost never bind; the run REPORTS how often it does, because a clamp
# that binds often would silently mean the advection is not what it claims.
const V_MIN = 0.5                         # m/s
const V_MAX = 5.0                         # m/s

const TIME_STEP = 3600.0                  # seconds per assimilation step
const NSUBSTEP  = 10                      # advection substeps per step

# Instrumentation for diagnostic 4 — filled in by advect!.
const VSTAT_SUM   = Threads.Atomic{Float64}(0.0)   # Σ velocity over cells
const VSTAT_N     = Threads.Atomic{Int}(0)         # cells counted
const VSTAT_CLAMP = Threads.Atomic{Int}(0)         # times the clamp bound

# Cells close enough to a sensor that an observation genuinely speaks about them
# — see DIAGNOSTICS note 1. At 10 km the smoothing kernel exp(−r²/2ℓ²) with
# ℓ = 15 km is still 0.80, so these cells are strongly tied to a sensor. Using
# the full correlation length instead would mark 720 of 1600 cells as "near",
# too large a fraction to contrast against.
const NEAR_RADIUS = 10_000.0
const NEAR_MASK = let m = falses(NSTATE)
    for s in 1:NSTATE
        xs = ((s - 1) ÷ NY) * DX
        ys = ((s - 1) %  NY) * DX
        for q in SENSORS
            xq = ((q - 1) ÷ NY) * DX
            yq = ((q - 1) %  NY) * DX
            if (xs - xq)^2 + (ys - yq)^2 <= NEAR_RADIUS^2
                m[s] = true; break
            end
        end
    end
    m
end
const FAR_MASK = .!NEAR_MASK

clampβ!(v) = (@inbounds for i in eachindex(v); v[i] = max(v[i], MIN_BETA); end; v)

# =============================================================================
#  5. FIELD CONSTRUCTION
# =============================================================================
"""
    normalised_pseudorandom_wave(rng) -> flat length-NSTATE vector

A smooth random field built from a few sine modes (wavenumbers 0…MAX_WAVENUMBER
in each direction) with random amplitudes and phases, normalised to unit
standard deviation. Used to offset the truth into a wrong-but-plausible guess.
"""
function normalised_pseudorandom_wave(rng::AbstractRNG)
    wave = zeros(Float64, NY, NX)
    for kx in 0:MAX_WAVENUMBER, ky in 0:MAX_WAVENUMBER
        kkx = 2π * kx / NX; kky = 2π * ky / NY
        a = randn(rng); phx = randn(rng) * 2π; phy = randn(rng) * 2π
        @inbounds for j in 1:NY
            sy = sin(kky * j + phy)
            for i in 1:NX
                wave[j, i] += a * sin(kkx * i + phx) * sy
            end
        end
    end
    σ = std(vec(wave)); σ == 0 && (σ = 1.0)
    return vec(wave ./ σ)
end

"""
    build_noise_factor() -> L

Cholesky factor of a squared-exponential covariance exp(−r²/2ℓ²) over the grid,
flattened (y,x) column-major to match the state. `L*z` is a smooth field with
unit variance per cell, so an amplitude in Pa·s/m applies by multiplication.
1e-8 ridge keeps the factorisation well posed. Dense 1600×1600, about a second,
done once.
"""
function build_noise_factor()
    ℓ2 = NOISE_LEN^2
    K = Matrix{Float64}(undef, NSTATE, NSTATE)
    @inbounds for i2 in 1:NX, j2 in 1:NY
        idx2 = (i2 - 1) * NY + j2
        x2 = (i2 - 1) * DX; y2 = (j2 - 1) * DX
        for i1 in 1:NX, j1 in 1:NY
            idx1 = (i1 - 1) * NY + j1
            x1 = (i1 - 1) * DX; y1 = (j1 - 1) * DX
            K[idx1, idx2] = exp(-((x1 - x2)^2 + (y1 - y2)^2) / (2 * ℓ2))
        end
    end
    @inbounds for i in 1:NSTATE; K[i, i] += 1e-8; end
    return Matrix(cholesky(Symmetric(K)).L)
end

"Add σ·(smooth noise) to `state` in place."
function apply_noise!(state::AbstractVector, L::Matrix{Float64},
                      buf::Vector{Float64}, rng::AbstractRNG, σ::Real)
    @inbounds for i in eachindex(buf); buf[i] = randn(rng); end
    mul!(state, L, buf, σ, 1.0)
    return state
end

"""
    advect!(state)

Nonlinear upwind advection of β in +x, periodic in x, NSUBSTEP substeps.

The advecting velocity is `V_SCALE / β`, clamped to [V_MIN, V_MAX]: a SLIPPERY
bed carries its own pattern FASTER. Temperature sets β through the known map, so
temperature sets the transport rate too.

This is STATE-DEPENDENT — the velocity is read from the field being advected, so
a particle whose β is wrong transports its own error at the wrong rate, and the
error field deforms differently for every particle. That differential deformation
is what a shared velocity field cannot provide, and what temp_field_run lacked
entirely (there the error field never moved at all: pattern correlation 0.850
after a full cycle).

Structurally identical to double_bump_run's advect! — same upwind stencil, same
periodicity, same substepping — so the only change under comparison is where the
velocity comes from.
"""
function advect!(state::AbstractVector)
    β = reshape(state, NY, NX)
    β_new = similar(β)
    dt = TIME_STEP / NSUBSTEP
    vsum = 0.0; nclamp = 0
    for _ in 1:NSUBSTEP
        @inbounds for j in 1:NY, i in 1:NX
            im = mod1(i - 1, NX)
            dβdx = (β[j, i] - β[j, im]) / DX
            vraw = V_SCALE / max(β[j, i], 1.0)
            v = clamp(vraw, V_MIN, V_MAX)
            v != vraw && (nclamp += 1)
            vsum += v
            β_new[j, i] = β[j, i] - v * dt * dβdx
        end
        β, β_new = β_new, β
    end
    state .= vec(β)
    # Atomic because advect! runs inside the threaded forecast loop.
    Threads.atomic_add!(VSTAT_SUM,   vsum)
    Threads.atomic_add!(VSTAT_N,     NSTATE * NSUBSTEP)
    Threads.atomic_add!(VSTAT_CLAMP, nclamp)
    return state
end

# =============================================================================
#  6. WAVI OBSERVATION OPERATOR
# =============================================================================
const U_ISZERO   = ["north"]
const V_ISZERO   = ["south", "east", "west"]
const WEERTMAN_M = 1.0        # linear sliding: τ_b = β·u, so gh.β ≡ the field passed in

const GRID = Grid(nx = NX, ny = NY, dx = DX, dy = DX, x0 = 0.0, y0 = 0.0,
                  u_iszero = U_ISZERO, v_iszero = V_ISZERO)
const Z_B    = zeros(NX, NY)
const H_INIT = max.(z_s.(GRID.xxh) .- Z_B, 0.0)
const IC     = InitialConditions(initial_thickness = H_INIT)

quiet(f) = redirect_stdout(f, devnull)
const WAVI_CALLS = Threads.Atomic{Int}(0)

"Speed field (flat, (y,x)) for a β field given flat in (y,x)."
function wavi_speed_flat(beta::AbstractVector)
    Threads.atomic_add!(WAVI_CALLS, 1)
    β_xy = Matrix(transpose(reshape(beta, NY, NX)))
    model = Model(grid = GRID, bed_elevation = Z_B, initial_conditions = IC,
                  params = Params(weertman_c = β_xy, weertman_m = WEERTMAN_M))
    update_state!(model)
    u = vec(transpose(model.fields.gh.u))
    v = vec(transpose(model.fields.gh.v))
    return sqrt.(u .^ 2 .+ v .^ 2)
end

"Observation operator h(β): log surface speed at the 16 sensors."
logspeed_at_sensors(beta) = log.(max.(wavi_speed_flat(beta)[SENSORS], 1e-6))

function systematic_resample(w::AbstractVector, rng::AbstractRNG)
    N = length(w); c = cumsum(w); u0 = rand(rng) / N
    idx = Vector{Int}(undef, N); j = 1
    for i in 1:N
        u = u0 + (i - 1) / N
        while j < N && c[j] < u; j += 1; end
        idx[i] = j
    end
    return idx
end

# =============================================================================
#  7. SET UP
# =============================================================================
say("="^78)
say(" β FIELD FROM VELOCITY OBSERVATIONS — temperature drives drag AND transport")
say("="^78)
say(@sprintf(" filter     : N = %d, T = %d hourly steps (%.1f diurnal cycles), K = %d stages",
             NPRT, NSTEP, NSTEP / 24, K_TEMPER))
say(@sprintf(" state      : the β field itself, %d cells  (map C, k are KNOWN)", NSTATE))
say(@sprintf(" known map  : β = %.0f − %.0f·(T − %.0f)", C_MAP, K_MAP, T_REF))
say(@sprintf(" dynamics   : βₜ₊₁(x) = βₜ(x − v·dt) − k·(Tₜ₊₁ − Tₜ) + noise"))
say(@sprintf(" transport  : v = %.0f/β m/s, clamped [%.1f, %.1f], %d substeps of %.0f s",
             V_SCALE, V_MIN, V_MAX, NSUBSTEP, TIME_STEP / NSUBSTEP))
say(@sprintf(" background : truth + %.0f·(smooth wave, seed %d), ensemble spread %.0f",
             BACKGROUND_STD, BACKGROUND_SEED, INIT_STD))
say(@sprintf(" β noise    : process %.0f/step, jitter %.0f, correlation length %.0f km",
             PROCESS_STD, SIGMA_JITTER, NOISE_LEN / 1000))
say(@sprintf(" forcing    : diurnal ±%.0f °C, seasonal ±%.0f °C, wobble sd %.1f °C — KNOWN/SHARED",
             T_ADIUR, T_AANN, T_NOISE_SD))
say(@sprintf(" observation: %d sensors, log-speed, σ_obs = %.2f  (%d of %d cells within %.0f km)",
             NOBS, SIGMA_OBS, count(NEAR_MASK), NSTATE, NEAR_RADIUS / 1000))
say(@sprintf(" sliding    : weertman_m = %.1f   boundaries: u%s v%s",
             WEERTMAN_M, U_ISZERO, V_ISZERO))
say(@sprintf(" seeds      : filter %d, truth/obs %d   threads = %d",
             SEED_PF, SEED_OBS, Threads.nthreads()))
say("")

Lfac = build_noise_factor()
noise_buf = zeros(NSTATE)

# --- the shared temperature record --------------------------------------
# Hourly steps from DAY0. AR(1) wobble with unit stationary variance scaled by
# T_NOISE_SD, so the realised sd is T_NOISE_SD regardless of τ.
rng_forcing = MersenneTwister(SEED_OBS + 777)
ρT = exp(-1.0 / T_NOISE_TAU)                   # 1-hour step
Tfields = Vector{Vector{Float64}}(undef, NSTEP + 1)
let Nprev = reshape(Lfac * randn(rng_forcing, NSTATE), NY, NX)
    for t in 0:NSTEP
        if t > 0
            Nprev = ρT .* Nprev .+ sqrt(1 - ρT^2) .*
                    reshape(Lfac * randn(rng_forcing, NSTATE), NY, NX)
        end
        day = DAY0 + t / 24
        Tfields[t + 1] = vec(clean_temperature(day) .+ T_NOISE_SD .* Nprev)
    end
end
Tmean_series = [mean(reshape(T, NY, NX)[IN, IN]) for T in Tfields]
say(@sprintf(" temperature: %.2f … %.2f °C over the run (interior mean)",
             minimum(Tmean_series), maximum(Tmean_series)))

"""
    temperature_step!(state, ti_from, ti_to)

Half of the forecast model: β responds to the CHANGE in temperature through the
known map. The other half is `advect!`.

Note what is NOT here: `state` never appears on the right-hand side. The
increment depends only on the shared temperature record, so it is added
identically to the truth and to every particle and cancels out of the error —
on its own it can never help the filter. That is precisely what temp_field_run
measured (11.8%, error pattern correlation 0.850). It is kept because the
physics is real — the bed genuinely does stiffen at night — but the work of
making the error observable is done by `advect!`.
"""
function temperature_step!(state::AbstractVector, ti_from::Int, ti_to::Int)
    Tf = Tfields[ti_from]; Tt = Tfields[ti_to]
    @inbounds for i in eachindex(state)
        state[i] -= K_MAP * (Tt[i] - Tf[i])
    end
    return state
end

# =============================================================================
#  8. TRUTH TRAJECTORY AND SYNTHETIC OBSERVATIONS
# =============================================================================
say(" generating truth trajectory + WAVI observations ...")
rng_truth = MersenneTwister(SEED_OBS)
truth_states   = zeros(NSTATE, NSTEP + 1)
observations   = zeros(NOBS, NSTEP)
truth_logspeed = zeros(NOBS, NSTEP)

# The truth starts exactly on the map at the initial temperature, then walks.
s_truth = clampβ!(C_MAP .- K_MAP .* (Tfields[1] .- T_REF))
truth_states[:, 1] = s_truth
quiet() do
    for t in 1:NSTEP
        advect!(s_truth)                       # transport, then local response
        temperature_step!(s_truth, t, t + 1)
        apply_noise!(s_truth, Lfac, noise_buf, rng_truth, PROCESS_STD)
        clampβ!(s_truth)
        truth_states[:, t + 1] = s_truth
        h = logspeed_at_sensors(s_truth)
        truth_logspeed[:, t] = h
        observations[:, t] = h .+ SIGMA_OBS .* randn(rng_truth, NOBS)
    end
end
say(@sprintf(" truth β    : %.0f … %.0f Pa·s/m over the run",
             minimum(truth_states), maximum(truth_states)))

# The background: one wrong field shared by all particles.
background_prior = truth_states[:, 1] .+ BACKGROUND_STD .*
                   normalised_pseudorandom_wave(MersenneTwister(BACKGROUND_SEED))
say(@sprintf(" background : RMSE %.1f Pa·s/m vs truth (systematic, shared by all particles)",
             sqrt(mean((background_prior .- truth_states[:, 1]) .^ 2))))

# =============================================================================
#  9. TEMPERED PARTICLE FILTER
# =============================================================================
say(" running tempered particle filter ...")
rng_pf = MersenneTwister(SEED_PF)
particles = zeros(NSTATE, NPRT)
for pp in 1:NPRT
    v = view(particles, :, pp)
    v .= background_prior
    apply_noise!(v, Lfac, noise_buf, rng_pf, INIT_STD)   # different guess each
    clampβ!(v)
end

ensemble_mean = zeros(NSTATE, NSTEP + 1)
ensemble_mean[:, 1] = mean(particles; dims = 2)[:, 1]
ess_series    = zeros(NSTEP)
log_weights   = zeros(NPRT)
loglik        = zeros(NPRT)
phi           = collect(0:K_TEMPER) ./ K_TEMPER
inv2σ2        = 1.0 / (2 * SIGMA_OBS^2)
n_resample    = 0

"""
    logdens!(dest, parts, y_t)

Gaussian log-likelihood of each particle given the observations, evaluated in
parallel — the N WAVI solves are independent and `dest[pp]` writes are
per-index, so there is no race. One redirect_stdout wraps the whole region
because it is process-global (doing it per call inside threads would race).
"""
function logdens!(dest, parts, y_t)
    quiet() do
        @sync for pp in 1:NPRT
            Threads.@spawn begin
                h = logspeed_at_sensors(view(parts, :, pp))
                dest[pp] = -sum((h .- y_t) .^ 2) * inv2σ2
            end
        end
    end
end

tstart = time()
for t in 1:NSTEP
    global particles, log_weights, n_resample

    # --- forecast: transport each particle's field, then apply the local
    #     temperature change, then process noise. Same order as the truth.
    for pp in 1:NPRT
        v = view(particles, :, pp)
        advect!(v)                             # v = V_SCALE/β — state-dependent
        temperature_step!(v, t, t + 1)
        apply_noise!(v, Lfac, noise_buf, rng_pf, PROCESS_STD)
        clampβ!(v)
    end

    y_t = view(observations, :, t)
    logdens!(loglik, particles, y_t)                    # N WAVI solves

    # --- analysis: apply the likelihood in K tempered stages --------------
    stage_min_ess = Inf
    for k in 1:K_TEMPER
        log_weights .+= (phi[k + 1] - phi[k]) .* loglik
        w = exp.(log_weights .- maximum(log_weights)); w ./= sum(w)
        ess_k = 1.0 / sum(w .^ 2)
        stage_min_ess = min(stage_min_ess, ess_k)
        k == K_TEMPER && (ensemble_mean[:, t + 1] = particles * w)

        if ess_k < ESS_THRESHOLD
            particles = particles[:, systematic_resample(w, rng_pf)]
            # Jitter after resampling (regularised PF): duplicated particles
            # would otherwise be identical and the ensemble would lose spread.
            for pp in 1:NPRT
                v = view(particles, :, pp)
                apply_noise!(v, Lfac, noise_buf, rng_pf, SIGMA_JITTER)
                clampβ!(v)
            end
            log_weights .= 0.0
            n_resample += 1
            # Particles moved, so the likelihood must be recomputed before the
            # next stage can use it.
            k < K_TEMPER && logdens!(loglik, particles, y_t)
        end
    end

    ess_series[t] = stage_min_ess
    err_t = ensemble_mean[:, t + 1] .- truth_states[:, t + 1]
    say(@sprintf("   step %2d/%d  ESS = %6.1f  RMSE(β) = %6.1f  [near %6.1f  far %6.1f]  T̄ = %6.2f °C  (%.0f s)",
                 t, NSTEP, ess_series[t], sqrt(mean(err_t .^ 2)),
                 sqrt(mean(err_t[NEAR_MASK] .^ 2)), sqrt(mean(err_t[FAR_MASK] .^ 2)),
                 Tmean_series[t + 1], time() - tstart))
end
wall = time() - tstart

# =============================================================================
# 10. SUMMARY AND DIAGNOSTICS
# =============================================================================
err_field = ensemble_mean .- truth_states
rmse_beta = [sqrt(mean(err_field[:, t] .^ 2))            for t in 1:NSTEP+1]
rmse_near = [sqrt(mean(err_field[NEAR_MASK, t] .^ 2))    for t in 1:NSTEP+1]
rmse_far  = [sqrt(mean(err_field[FAR_MASK,  t] .^ 2))    for t in 1:NSTEP+1]

# Diagnostic 2: does the error field deform, or is it frozen? Pattern
# correlation of the error at step t with the error at step 0. Near 1 ⇒ the
# filter is being asked the same unanswerable question every step.
e0 = err_field[:, 1] .- mean(err_field[:, 1])
err_corr = map(1:NSTEP+1) do t
    et = err_field[:, t] .- mean(err_field[:, t])
    d = sqrt(sum(e0 .^ 2) * sum(et .^ 2))
    d < 1e-12 ? NaN : sum(e0 .* et) / d
end

# Diagnostic 3: observability. ∂log(speed)/∂β = −1/β, so 1/β̄ is how loudly a
# given β error speaks in the data at each step.
beta_bar     = [mean(reshape(truth_states[:, t], NY, NX)[IN, IN]) for t in 1:NSTEP+1]
observability = 1.0 ./ beta_bar

say("")
say("-"^78)
say(" RESULTS")
say("-"^78)
say(@sprintf(" RMSE(β)      : %.1f  →  %.1f   (%.1f%% reduction)",
             rmse_beta[1], rmse_beta[end], 100 * (1 - rmse_beta[end] / rmse_beta[1])))
say(@sprintf("   ├─ NEAR sensors (%4d cells, ≤%.0f km) : %.1f  →  %.1f   (%.1f%%)",
             count(NEAR_MASK), NEAR_RADIUS / 1000, rmse_near[1], rmse_near[end],
             100 * (1 - rmse_near[end] / rmse_near[1])))
say(@sprintf("   └─ FAR  sensors (%4d cells)          : %.1f  →  %.1f   (%.1f%%)",
             count(FAR_MASK), rmse_far[1], rmse_far[end],
             100 * (1 - rmse_far[end] / rmse_far[1])))
say(@sprintf(" best RMSE(β) : %.1f at step %d", minimum(rmse_beta), argmin(rmse_beta) - 1))
say(@sprintf(" error pattern correlation with step 0 : %.3f  (1 ⇒ the error field never deformed;",
             err_corr[end]))
say("                  temp_field_run, with no transport, ended at 0.850)")
# Diagnostic 4: did β actually travel? A run where the field moved 2 km would
# have tested nothing, so this is reported rather than assumed.
v_mean = VSTAT_SUM[] / max(VSTAT_N[], 1)
say(@sprintf(" transport    : mean v = %.2f m/s ⇒ %.1f km/step, %.0f km over the run (domain %.0f km)",
             v_mean, v_mean * TIME_STEP / 1000, v_mean * TIME_STEP * NSTEP / 1000,
             DOMAIN / 1000))
say(@sprintf("                Courant %.2f (stable < 1), velocity clamp bound on %.3f%% of cell-substeps",
             v_mean * (TIME_STEP / NSUBSTEP) / DX,
             100 * VSTAT_CLAMP[] / max(VSTAT_N[], 1)))
say(@sprintf(" observability 1/β̄ : %.2e … %.2e  (%.2f× swing over the cycle)",
             minimum(observability), maximum(observability),
             maximum(observability) / minimum(observability)))
say(@sprintf(" ESS          : mean %.1f, min %.1f at step %d  (threshold N/2 = %.0f)",
             mean(ess_series), minimum(ess_series), argmin(ess_series), ESS_THRESHOLD))
say(@sprintf(" resampling   : %d events", n_resample))
say(@sprintf(" WAVI solves  : %d  (base N·T = %d, overhead ×%.2f)",
             WAVI_CALLS[], NPRT * NSTEP, WAVI_CALLS[] / (NPRT * NSTEP)))
say(@sprintf(" wall clock   : %.0f s = %.1f min  (%.3f s per solve)",
             wall, wall / 60, wall / WAVI_CALLS[]))
say("-"^78)

# =============================================================================
# 11. SAVE
# =============================================================================
h5open(joinpath(OUTDIR, "tracking.h5"), "w") do f
    f["truth/beta"]         = reshape(truth_states,  NY, NX, NSTEP + 1)
    f["ensemble_mean/beta"] = reshape(ensemble_mean, NY, NX, NSTEP + 1)
    f["background"]         = reshape(background_prior, NY, NX)
    f["temperature"]        = reshape(hcat(Tfields...), NY, NX, NSTEP + 1)
    f["weights/ess"]        = ess_series
    f["observations"]       = observations
    f["truth_logspeed"]     = truth_logspeed
    f["sensor_indices"]     = collect(SENSORS)
    f["near_mask"]          = reshape(collect(NEAR_MASK), NY, NX)
    f["rmse_beta"]          = rmse_beta
    f["rmse_near"]          = rmse_near
    f["rmse_far"]           = rmse_far
    f["error_pattern_corr"] = err_corr
    f["observability"]      = observability
    f["temperature_mean"]   = Tmean_series
    for (k, v) in ("n_particles" => NPRT, "n_steps" => NSTEP, "k_temper" => K_TEMPER,
                   "sigma_obs" => SIGMA_OBS, "seed_pf" => SEED_PF, "seed_obs" => SEED_OBS,
                   "C_map" => C_MAP, "k_map" => K_MAP, "T_ref" => T_REF,
                   "background_std" => BACKGROUND_STD, "init_std" => INIT_STD,
                   "process_std" => PROCESS_STD, "sigma_jitter" => SIGMA_JITTER,
                   "noise_len" => NOISE_LEN, "weertman_m" => WEERTMAN_M,
                   "v_scale" => V_SCALE, "v_min" => V_MIN, "v_max" => V_MAX,
                   "n_substep" => NSUBSTEP, "time_step" => TIME_STEP,
                   "v_mean_achieved" => v_mean,
                   "clamp_fraction" => VSTAT_CLAMP[] / max(VSTAT_N[], 1))
        f["params/$k"] = v
    end
end

# =============================================================================
# 12. FIGURES
# =============================================================================
say("")
print(" plotting : ")
gx = collect(0:NX-1) .* (DX / 1000); gy = collect(0:NY-1) .* (DX / 1000)
sx = [((s - 1) ÷ NY) * DX / 1000 for s in SENSORS]
sy = [((s - 1) %  NY) * DX / 1000 for s in SENSORS]
steps = 0:NSTEP

# --- RMSE, total and split by sensor coverage ---------------------------
print("rmse ")
plot(steps, rmse_beta, lw = 2.5, marker = :circle, ms = 3, color = :crimson,
     label = "all cells", xlabel = "assimilation step (hours)",
     ylabel = "RMSE (Pa·s/m)", ylims = (0, :auto),
     title = @sprintf("β error: %.1f → %.1f  (%.1f%% reduction)",
                      rmse_beta[1], rmse_beta[end],
                      100 * (1 - rmse_beta[end] / rmse_beta[1])),
     size = (900, 560), left_margin = 6mm, bottom_margin = 5mm, legend = :best)
plot!(steps, rmse_near, lw = 2, ls = :dash, color = :steelblue,
      label = @sprintf("near sensors (≤%.0f km, %d cells)", NEAR_RADIUS / 1000, count(NEAR_MASK)))
plot!(steps, rmse_far, lw = 2, ls = :dot, color = :darkorange,
      label = @sprintf("far from sensors (%d cells)", count(FAR_MASK)))
savefig(joinpath(OUTDIR, "rmse_beta.png"))

print("| ess ")
plot(1:NSTEP, ess_series, lw = 2, marker = :circle, ms = 3, color = :steelblue, label = "ESS",
     xlabel = "assimilation step (hours)", ylabel = "effective sample size",
     title = @sprintf("ESS (mean %.1f, min %.1f), %d resamples",
                      mean(ess_series), minimum(ess_series), n_resample),
     ylims = (0, NPRT), size = (820, 500), left_margin = 6mm, bottom_margin = 5mm)
hline!([ESS_THRESHOLD], ls = :dash, color = :red, label = "resample threshold N/2")
savefig(joinpath(OUTDIR, "ess_tracking.png"))

# --- THE DIAGNOSTIC FIGURE ----------------------------------------------
# Does the error field deform, and does improvement track observability?
print("| diagnostics ")
d1 = plot(steps, err_corr, lw = 2.5, marker = :circle, ms = 3, color = :purple,
          ylims = (0, 1.05), ylabel = "pattern correlation", legend = :best,
          label = "this run (with transport)",
          title = "error field vs its own shape at step 0  (1 = frozen, never deformed)")
hline!(d1, [1.0], ls = :dash, color = :black, label = "")
hline!(d1, [0.850], ls = :dot, lw = 2, color = :gray,
       label = "temp_field_run final (no transport)")
d2 = plot(steps, observability, lw = 2.5, color = :orangered, legend = false,
          ylabel = "1/β̄  (∂log speed/∂β)", xlabel = "assimilation step (hours)",
          title = "observability — a β error is loudest when the bed is warm and slippery")
plot(d1, d2, layout = (2, 1), size = (950, 700),
     left_margin = 12mm, bottom_margin = 5mm)
savefig(joinpath(OUTDIR, "diagnostics.png"))

# --- forcing and response ------------------------------------------------
print("| forcing ")
bt_mean = [mean(reshape(truth_states[:, t],  NY, NX)[IN, IN]) for t in 1:NSTEP+1]
bm_mean = [mean(reshape(ensemble_mean[:, t], NY, NX)[IN, IN]) for t in 1:NSTEP+1]
p1 = plot(steps, Tmean_series, lw = 2, color = :orangered, legend = false,
          ylabel = "T (°C)", title = "known temperature forcing (diurnal + wobble)")
p2 = plot(steps, bt_mean, lw = 2, color = :black, label = "truth",
          ylabel = "β (Pa·s/m)", xlabel = "assimilation step (hours)",
          title = "interior-mean β: truth vs filter", legend = :best)
plot!(p2, steps, bm_mean, lw = 2, ls = :dash, color = :crimson, label = "ensemble mean")
plot(p1, p2, layout = (2, 1), size = (950, 640), left_margin = 10mm, bottom_margin = 5mm)
savefig(joinpath(OUTDIR, "forcing_timeseries.png"))

# --- animations -----------------------------------------------------------
# Fixed colour limits across frames (per-frame autoscaling would fake
# convergence); 2–98% clipping so one outlier frame cannot flatten the rest;
# error panels symmetric-diverging so convergence reads as fading to white.
pan(f, ttl, lm, cm, cb) =
    heatmap(gx, gy, f, aspect_ratio = 1, color = cm, clims = lm, colorbar_title = cb,
            title = ttl, titlefontsize = 9, xlabel = "x (km)", ylabel = "y (km)")

print("| beta gif ")
Tb = reshape(truth_states,  NY, NX, NSTEP + 1)
Mb = reshape(ensemble_mean, NY, NX, NSTEP + 1)
allb = vcat(vec(Tb), vec(Mb))
blim = (quantile(allb, 0.02), quantile(allb, 0.98))
berr = Mb .- Tb
belim = quantile(abs.(vec(berr)), 0.98); belim = (-belim, belim)
anim_b = @animate for t in 1:NSTEP+1
    q1 = pan(Tb[:, :, t], "truth β", blim, :viridis, "Pa·s/m")
    q2 = pan(Mb[:, :, t], "ensemble mean β", blim, :viridis, "Pa·s/m")
    q3 = pan(berr[:, :, t], "error (mean − truth)", belim, :balance, "Pa·s/m")
    for p in (q1, q2); scatter!(p, sx, sy, ms = 2.5, color = :red, label = ""); end
    scatter!(q3, sx, sy, ms = 2.5, color = :black, label = "")
    ttl = t == 1 ?
        @sprintf("β — t = 0 (background)   RMSE = %.1f   T̄ = %.1f °C", rmse_beta[t], Tmean_series[t]) :
        @sprintf("β — hour %d/%d   RMSE = %.1f  (near %.1f, far %.1f)   T̄ = %.1f °C",
                 t - 1, NSTEP, rmse_beta[t], rmse_near[t], rmse_far[t], Tmean_series[t])
    plot(q1, q2, q3, layout = (1, 3), size = (1500, 480), plot_title = ttl,
         plot_titlefontsize = 11, left_margin = 9mm, bottom_margin = 6mm,
         right_margin = 3mm, top_margin = 2mm)
end
gif(anim_b, joinpath(OUTDIR, "beta_evolution.gif"), fps = 3)

print("| velocity gif ")
# β is the complete state, so velocities are recovered afterwards by re-solving
# WAVI on the saved β — 2(T+1) solves, seconds — rather than stored during the run.
Strue = Array{Float64}(undef, NY, NX, NSTEP + 1); Smean = similar(Strue)
quiet() do
    for t in 1:NSTEP+1
        Strue[:, :, t] = reshape(wavi_speed_flat(truth_states[:, t]),  NY, NX)
        Smean[:, :, t] = reshape(wavi_speed_flat(ensemble_mean[:, t]), NY, NX)
    end
end
alls = vcat(vec(Strue), vec(Smean))
slim = (quantile(alls, 0.02), quantile(alls, 0.98))
serr = Smean .- Strue
selim = quantile(abs.(vec(serr)), 0.98); selim = (-selim, selim)
srmse = [sqrt(mean(serr[:, :, t] .^ 2)) for t in 1:NSTEP+1]
# Relative error: β falls while speed rises over the warm half of the cycle, so
# the same fractional error looks small in β and large in speed. This panel is
# the honest common measure.
srel = 100 .* serr ./ max.(Strue, 1e-6)
srlim = quantile(abs.(vec(srel)), 0.98); srlim = (-srlim, srlim)
anim_v = @animate for t in 1:NSTEP+1
    q1 = pan(Strue[:, :, t], "truth speed", slim, :thermal, "m/yr")
    q2 = pan(Smean[:, :, t], "ensemble mean speed", slim, :thermal, "m/yr")
    q3 = pan(serr[:, :, t], "error (mean − truth)", selim, :balance, "m/yr")
    q4 = pan(srel[:, :, t], "relative error", srlim, :balance, "%")
    for p in (q1, q2); scatter!(p, sx, sy, ms = 2.5, color = :cyan, label = ""); end
    for p in (q3, q4); scatter!(p, sx, sy, ms = 2.5, color = :black, label = ""); end
    ttl = t == 1 ?
        @sprintf("velocity — t = 0 (background)   speed RMSE = %.2f m/yr", srmse[t]) :
        @sprintf("velocity — hour %d/%d   speed RMSE = %.2f m/yr", t - 1, NSTEP, srmse[t])
    plot(q1, q2, q3, q4, layout = (1, 4), size = (1960, 470), plot_title = ttl,
         plot_titlefontsize = 11, left_margin = 9mm, bottom_margin = 6mm,
         right_margin = 3mm, top_margin = 2mm)
end
gif(anim_v, joinpath(OUTDIR, "velocity_evolution.gif"), fps = 3)
println("| done")

say(@sprintf(" speed RMSE   : %.2f  →  %.2f m/yr", srmse[1], srmse[end]))
open(joinpath(OUTDIR, "summary.txt"), "w") do f; write(f, String(take!(LOG))); end
say("")
say(" all outputs → $OUTDIR")
for fn in sort(readdir(OUTDIR)); println("   $fn"); end
