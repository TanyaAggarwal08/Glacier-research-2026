#!/usr/bin/env julia
# =============================================================================
#  LEARNING THE β(T) RELATIONSHIP FROM VELOCITY OBSERVATIONS
#  — a particle filter driven by a time-varying temperature field, no advection
# =============================================================================
#
#  ONE FILE, ONE COMMAND, ALL RESULTS:
#
#      julia run_temperature_pf.jl
#
#  Everything lands in  /results/temperature_pf/
#
#  Self-contained: temperature model, β dynamics, WAVI observation operator,
#  tempered particle filter 
#
# -----------------------------------------------------------------------------
#  THE PROBLEM
# -----------------------------------------------------------------------------
#  Basal drag is assumed to depend on surface temperature through a linear map
#
#      β(x,y,t) = C − k·(T(x,y,t) − T_ref) + δ(x,y,t)
#                 └──── the map ─────┘       └ residual ┘
#
#  but the map is NOT known — C and k are exactly what we want to find out. The
#  only data are noisy observations of ice surface SPEED at 16 sensors. WAVI is
#  the forward operator that turns a β field into a velocity field.
#
#  Each particle therefore carries THREE unknowns:
#
#      C          1     β at the reference temperature, Pa·s/m
#      k          1     sensitivity, Pa·s/m per °C  (positive ⇒ warm = slippery)
#      δ(x,y)  1600     smooth residual: where the bed departs from the map.
#                       This is what carries the initial error and accumulates
#                       process noise, exactly as the β field did in the
#                       double-bump experiment.
#
#  State dimension 1602.
#
# -----------------------------------------------------------------------------
#  WHAT MAKES IT IDENTIFIABLE
# -----------------------------------------------------------------------------
#  There is NO advection here. The entire time evolution comes from the diurnal
#  temperature cycle: as the day warms and cools, the truth's β swings and the
#  ice measurably speeds up and slows down. A particle with the wrong k predicts
#  the wrong AMOUNT of speed-up, and the cycle re-tests it every hour. The
#  temperature swing is the signal that identifies the map.
#
#  That also sets the run length: the default is 24 hourly steps = one full
#  diurnal cycle, so every particle is tested across the whole temperature range
#  rather than a narrow slice of it.
#
#  IDENTIFIABILITY CAVEAT, worth knowing before reading the results: C and δ are
#  only separable through their spatial structure. A uniform shift in δ is
#  indistinguishable from a shift in C at any single instant — both raise β
#  everywhere by the same amount. What breaks the tie is that δ is spatially
#  smooth-but-structured while C is exactly uniform, and that k is identified by
#  the TIME variation, which δ (nearly static) cannot mimic. Expect k to be
#  recovered more sharply than C.
#
# -----------------------------------------------------------------------------
#  TEMPERATURE IS KNOWN
# -----------------------------------------------------------------------------
#  One temperature field is generated up front and shared by the truth and every
#  particle. It is seasonal + diurnal + elevation lapse, PLUS a smooth AR(1)
#  perturbation so it looks like a real, wobbling temperature record rather than
#  a clean sinusoid. Because everyone sees the same T, it is realistic forcing
#  and NOT a source of estimation error. All the uncertainty lives in β, via C,
#  k and δ.
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
#  Adding the sidewalls drops that to ~1.1× and was verified not to change the
#  physics (interior 17.8 → 17.3 m/yr; a full PF rerun moved RMSE 113.4 → 112.6).
#
# -----------------------------------------------------------------------------
#  OUTPUTS  →  results/temperature_pf/
# -----------------------------------------------------------------------------
#    parameter_recovery.png    THE KEY FIGURE — C and k posteriors vs truth
#    parameter_scatter.png     particle cloud in (C,k) space, prior vs posterior
#    rmse_beta.png             RMSE(β) against time
#    ess_tracking.png          ESS per step vs the N/2 threshold
#    beta_evolution.gif        truth β | mean β | error, per step
#    velocity_evolution.gif    truth speed | mean speed | error, per step
#    forcing_timeseries.png    the temperature forcing and the β/speed response
#    tracking.h5               everything numeric
#    summary.txt               every number printed below
#
# -----------------------------------------------------------------------------
#  USAGE
# -----------------------------------------------------------------------------
#    julia -t <threads> run_temperature_pf.jl [N] [T] [K] [σ_obs] [seed_pf] [seed_obs]
#
#    N        particles                              (default 500)
#    T        hourly assimilation steps              (default 24 = one day)
#    K        tempering stages                       (default 10)
#    σ_obs    observation sd, log-speed units        (default 0.5)
#    seed_pf  filter RNG seed                        (default 42)
#    seed_obs truth + observation-noise seed         (default 123)
#
#  Examples
#    julia -t 4  run_temperature_pf.jl 20 4 3     # smoke test, ~30 s
#    julia -t 10 run_temperature_pf.jl            # standard run, ~40 min
#    julia -t 10 run_temperature_pf.jl 500 48 10  # two diurnal cycles
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

const OUTDIR = joinpath(@__DIR__, "results", "temperature_pf")
mkpath(OUTDIR)
const LOG = IOBuffer()
say(s) = (println(s); flush(stdout); println(LOG, s))

# =============================================================================
#  1. DOMAIN
# =============================================================================
const NX, NY = 40, 40
const NFIELD = NX * NY              # 1600 residual cells
const NSTATE = NFIELD + 2           # + C + k
const IC_IDX = 1                    # row index of C in the state matrix
const IK_IDX = 2                    # row index of k
const IDELTA = 3:NSTATE             # rows holding δ

const DOMAIN = 160_000.0
const DX     = DOMAIN / NX
const IN     = 4:37                 # interior mask (drop 3-cell border)

z_s(x) = 1060.0 * sqrt(clamp(1.0 - x / DOMAIN, 0.0, 1.0))
const XS    = [(i - 0.5) * DX for i in 1:NX]
const Z_S_X = z_s.(XS)
const Z_REF = mean(Z_S_X)

# 16 sensors on a 4×4 grid at x,y ∈ {20,50,110,140} km, embedded so this file
# needs no external data. All sit at x ≥ 20 km, clear of the x = 0 divide.
const SENSOR_XY_KM = [(x, y) for y in (20, 50, 110, 140) for x in (20, 50, 110, 140)]
const SENSORS = let idx = Int[]
    for (x, y) in SENSOR_XY_KM
        i = clamp(round(Int, x * 1000 / DX) + 1, 1, NX)
        j = clamp(round(Int, y * 1000 / DX) + 1, 1, NY)
        push!(idx, (i - 1) * NY + j)          # flat, (y,x) column-major
    end
    unique(idx)
end
const NOBS = length(SENSORS)

# =============================================================================
#  2. TEMPERATURE FORCING  (known to everyone)
# =============================================================================
const T_MEAN  = -10.0     # annual mean at reference elevation, °C
const T_AANN  =  15.0     # seasonal amplitude, °C
const T_ADIUR =  10.0     # diurnal amplitude, °C — the signal that identifies k
const T_GAMMA =   0.0059  # lapse rate, °C/m
const T_DPEAK = 200.0     # day-of-year of the annual peak
const T_PHI   =   0.6     # diurnal phase, fraction of a day
const DAY0    = 200.0     # start day (near the annual peak, so ice is mobile)

const LAPSE = @. -T_GAMMA * (Z_S_X - Z_REF)     # length NX, time independent

base_temperature(t) =
    T_MEAN + T_AANN * cos(2π * (t - T_DPEAK) / 365) + T_ADIUR * cos(2π * (t - T_PHI))

"Clean temperature field at day `t`, (NY, NX), °C. Uniform in y."
clean_temperature(t::Real) = repeat(reshape(LAPSE .+ base_temperature(t), 1, NX), NY, 1)

# Smooth AR(1) wobble added on top, so the forcing looks like a real record.
# Shared by truth and particles — realistic forcing, not an error source.
const T_NOISE_SD  = 1.0      # °C
const T_NOISE_TAU = 6.0      # hours

# =============================================================================
#  3. THE UNKNOWN β(T) MAP
# =============================================================================
# TRUTH values the filter must find. β_true at T = −10 °C is 2000 Pa·s/m and
# rises 40 Pa·s/m for every degree of cooling. Over the ±10 °C diurnal swing
# that moves β by ±400 — a large, observable signal.
const C_TRUE   = 2000.0
const K_TRUE   =   40.0
const T_REF    =  -10.0      # reference temperature, fixed and known
const MIN_BETA =   10.0

# PRIOR over the map — deliberately WRONG so recovery is visible. The prior mean
# for k is half the truth, so the filter must roughly double it.
const C_PRIOR_MEAN =  2300.0
const C_PRIOR_SD   =   300.0
const K_PRIOR_MEAN =    20.0
const K_PRIOR_SD   =    12.0

# Residual field δ: truth starts at zero (the map is exactly right everywhere)
# and drifts under process noise; particles start with a smooth initial error.
#
# SIZING THE δ PRIOR — this matters more than it looks. The truth's δ is a random
# walk of step sd DELTA_PROC_SD, so after T steps it reaches sd ≈ PROC_SD·√T
# (≈ 50 at the defaults). δ has 1600 degrees of freedom constrained by only 16
# sensors, so the filter cannot distinguish between δ fields that fit the data
# equally well — it simply commits to whichever prior draw helps most. If the
# prior is WIDER than the truth ever gets, that commitment injects structure
# which is not there and RMSE(β) climbs even while the map is recovered
# perfectly. Measured with DELTA_INIT_SD = 150 and JITTER_DELTA = 20: the map
# error fell 440 → 4, but the filter's δ inflated to sd 154 against a truth of
# 51, and total RMSE(β) rebounded 77 → 175 after step 4.
# These values are matched to the scale the truth actually reaches.
const DELTA_PROC_SD  =  10.0     # process noise per step
const DELTA_INIT_SD  =  50.0     # ≈ PROC_SD·√NSTEP — the scale δ truly reaches
const NOISE_LEN      = 15_000.0  # correlation length of the smooth fields, m

# Post-resample jitter (regularised PF). Without jitter on C and k the parameter
# ensemble collapses to a handful of duplicated values and can never recover.
# The δ jitter is kept small for the same reason as the prior above: every
# resample injects it, and it accumulates in a direction the data cannot check.
const JITTER_DELTA =  5.0
const JITTER_C     = 25.0
const JITTER_K     =  1.5

const ESS_THRESHOLD = 0.5 * NPRT

beta_of(C, k, Tvec, delta) = max.(C .- k .* (Tvec .- T_REF) .+ delta, MIN_BETA)

# =============================================================================
#  4. SMOOTH FIELD GENERATOR
# =============================================================================
"""
    build_noise_factor() -> L

Cholesky factor of a squared-exponential covariance exp(−r²/2ℓ²) over the grid,
flattened (y,x) column-major to match the state. `L*z` is a smooth field with
unit variance per cell, so an amplitude in Pa·s/m (or °C) applies by
multiplication. 1e-8 ridge keeps the factorisation well posed. Dense 1600×1600,
about a second, done once.
"""
function build_noise_factor()
    ℓ2 = NOISE_LEN^2
    K = Matrix{Float64}(undef, NFIELD, NFIELD)
    @inbounds for i2 in 1:NX, j2 in 1:NY
        idx2 = (i2 - 1) * NY + j2
        x2 = (i2 - 1) * DX; y2 = (j2 - 1) * DX
        for i1 in 1:NX, j1 in 1:NY
            idx1 = (i1 - 1) * NY + j1
            x1 = (i1 - 1) * DX; y1 = (j1 - 1) * DX
            K[idx1, idx2] = exp(-((x1 - x2)^2 + (y1 - y2)^2) / (2 * ℓ2))
        end
    end
    @inbounds for i in 1:NFIELD; K[i, i] += 1e-8; end
    return Matrix(cholesky(Symmetric(K)).L)
end

"Add σ·(smooth field) to a length-NFIELD view, in place."
function add_smooth!(v::AbstractVector, L::Matrix{Float64},
                     buf::Vector{Float64}, rng::AbstractRNG, σ::Real)
    @inbounds for i in eachindex(buf); buf[i] = randn(rng); end
    mul!(v, L, buf, σ, 1.0)
    return v
end

# =============================================================================
#  5. WAVI OBSERVATION OPERATOR
# =============================================================================
const U_ISZERO = ["north"]
const V_ISZERO = ["south", "east", "west"]
const WEERTMAN_M = 1.0        # linear sliding: τ_b = β·u, so gh.β ≡ the field we pass

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
#  6. SET UP
# =============================================================================
say("="^78)
say(" LEARNING β(T) FROM VELOCITY OBSERVATIONS — temperature-driven particle filter")
say("="^78)
say(@sprintf(" filter     : N = %d, T = %d hourly steps (%.1f diurnal cycles), K = %d stages",
             NPRT, NSTEP, NSTEP / 24, K_TEMPER))
say(@sprintf(" state      : C (1) + k (1) + δ field (%d) = %d dimensions", NFIELD, NSTATE))
say(@sprintf(" truth map  : β = %.0f − %.0f·(T − %.0f) + δ", C_TRUE, K_TRUE, T_REF))
say(@sprintf(" prior map  : C ~ N(%.0f, %.0f)   k ~ N(%.0f, %.0f)   ← deliberately wrong",
             C_PRIOR_MEAN, C_PRIOR_SD, K_PRIOR_MEAN, K_PRIOR_SD))
say(@sprintf(" residual   : δ init sd %.0f, process sd %.0f, correlation length %.0f km",
             DELTA_INIT_SD, DELTA_PROC_SD, NOISE_LEN / 1000))
say(@sprintf(" forcing    : diurnal ±%.0f °C, seasonal ±%.0f °C, smooth wobble sd %.1f °C — KNOWN",
             T_ADIUR, T_AANN, T_NOISE_SD))
say(@sprintf(" observation: %d sensors, log-speed, σ_obs = %.2f", NOBS, SIGMA_OBS))
say(@sprintf(" sliding    : weertman_m = %.1f   boundaries: u%s v%s",
             WEERTMAN_M, U_ISZERO, V_ISZERO))
say(@sprintf(" seeds      : filter %d, truth/obs %d   threads = %d",
             SEED_PF, SEED_OBS, Threads.nthreads()))
say("")

Lfac = build_noise_factor()
buf  = zeros(NFIELD)

# --- the shared temperature record --------------------------------------
# Hourly steps starting at DAY0. AR(1) wobble with unit stationary variance,
# scaled by T_NOISE_SD, so the sd is exactly T_NOISE_SD regardless of τ.
rng_forcing = MersenneTwister(SEED_OBS + 777)
ρT = exp(-1.0 / T_NOISE_TAU)                   # 1-hour step
Tfields = Vector{Vector{Float64}}(undef, NSTEP + 1)
let Nprev = reshape(Lfac * randn(rng_forcing, NFIELD), NY, NX)
    for t in 0:NSTEP
        if t > 0
            Nprev = ρT .* Nprev .+ sqrt(1 - ρT^2) .*
                    reshape(Lfac * randn(rng_forcing, NFIELD), NY, NX)
        end
        day = DAY0 + t / 24
        Tfields[t + 1] = vec(clean_temperature(day) .+ T_NOISE_SD .* Nprev)
    end
end
Tmean_series = [mean(reshape(T, NY, NX)[IN, IN]) for T in Tfields]
say(@sprintf(" temperature over the run: %.2f … %.2f °C (interior mean)",
             minimum(Tmean_series), maximum(Tmean_series)))

# =============================================================================
#  7. TRUTH TRAJECTORY AND OBSERVATIONS
# =============================================================================
say(" generating truth trajectory + WAVI observations ...")
rng_truth = MersenneTwister(SEED_OBS)
truth_beta     = zeros(NFIELD, NSTEP + 1)
observations   = zeros(NOBS, NSTEP)
truth_logspeed = zeros(NOBS, NSTEP)

delta_true = zeros(NFIELD)                      # truth starts on the map exactly
truth_beta[:, 1] = beta_of(C_TRUE, K_TRUE, Tfields[1], delta_true)
quiet() do
    for t in 1:NSTEP
        # Residual drifts under process noise; the map itself is constant.
        add_smooth!(delta_true, Lfac, buf, rng_truth, DELTA_PROC_SD)
        truth_beta[:, t + 1] = beta_of(C_TRUE, K_TRUE, Tfields[t + 1], delta_true)
        h = logspeed_at_sensors(truth_beta[:, t + 1])
        truth_logspeed[:, t] = h
        observations[:, t]   = h .+ SIGMA_OBS .* randn(rng_truth, NOBS)
    end
end
say(@sprintf(" truth β range over the run: %.0f … %.0f Pa·s/m",
             minimum(truth_beta), maximum(truth_beta)))

# =============================================================================
#  8. TEMPERED PARTICLE FILTER
# =============================================================================
say(" running tempered particle filter ...")
rng_pf = MersenneTwister(SEED_PF)

# state[1] = C, state[2] = k, state[3:end] = δ
particles = zeros(NSTATE, NPRT)
for pp in 1:NPRT
    particles[IC_IDX, pp] = C_PRIOR_MEAN + C_PRIOR_SD * randn(rng_pf)
    particles[IK_IDX, pp] = K_PRIOR_MEAN + K_PRIOR_SD * randn(rng_pf)
    add_smooth!(view(particles, IDELTA, pp), Lfac, buf, rng_pf, DELTA_INIT_SD)
end

"β field implied by particle `pp` at time index `ti`."
particle_beta(parts, pp, ti) =
    beta_of(parts[IC_IDX, pp], parts[IK_IDX, pp], Tfields[ti], view(parts, IDELTA, pp))

ensemble_mean_beta = zeros(NFIELD, NSTEP + 1)
C_mean = zeros(NSTEP + 1); C_sd = zeros(NSTEP + 1)
K_mean = zeros(NSTEP + 1); K_sd = zeros(NSTEP + 1)
ess_series = zeros(NSTEP)
log_weights = zeros(NPRT)
loglik = zeros(NPRT)
phi = collect(0:K_TEMPER) ./ K_TEMPER
inv2σ2 = 1.0 / (2 * SIGMA_OBS^2)
n_resample = 0
particles_C0 = copy(particles[IC_IDX, :])       # keep the prior cloud for the scatter
particles_K0 = copy(particles[IK_IDX, :])

# t = 0 diagnostics (prior, before any data)
ensemble_mean_beta[:, 1] = mean(hcat([particle_beta(particles, pp, 1) for pp in 1:NPRT]...); dims = 2)[:, 1]
C_mean[1] = mean(particles[IC_IDX, :]); C_sd[1] = std(particles[IC_IDX, :])
K_mean[1] = mean(particles[IK_IDX, :]); K_sd[1] = std(particles[IK_IDX, :])

"""
    logdens!(dest, parts, ti, y)

Gaussian log-likelihood of every particle. Each particle's β is rebuilt from its
own (C, k, δ) at time index `ti`, then pushed through WAVI. The N solves are
independent and dest[pp] writes are per-index, so the loop is safely threaded;
one redirect_stdout wraps the whole region because it is process-global.
"""
function logdens!(dest, parts, ti, y_t)
    quiet() do
        @sync for pp in 1:NPRT
            Threads.@spawn begin
                h = logspeed_at_sensors(particle_beta(parts, pp, ti))
                dest[pp] = -sum((h .- y_t) .^ 2) * inv2σ2
            end
        end
    end
end

tstart = time()
for t in 1:NSTEP
    global particles, log_weights, n_resample

    # --- forecast -------------------------------------------------------
    # No advection. Temperature moves on its own (Tfields[t+1]); the only
    # stochastic evolution is process noise on the residual. C and k are
    # constant parameters — they change only through weighting and jitter.
    for pp in 1:NPRT
        add_smooth!(view(particles, IDELTA, pp), Lfac, buf, rng_pf, DELTA_PROC_SD)
    end

    y_t = view(observations, :, t)
    logdens!(loglik, particles, t + 1, y_t)

    # --- analysis: tempered likelihood ----------------------------------
    stage_min_ess = Inf
    for k in 1:K_TEMPER
        log_weights .+= (phi[k + 1] - phi[k]) .* loglik
        w = exp.(log_weights .- maximum(log_weights)); w ./= sum(w)
        ess_k = 1.0 / sum(w .^ 2)
        stage_min_ess = min(stage_min_ess, ess_k)

        if k == K_TEMPER
            C_mean[t + 1] = sum(w .* particles[IC_IDX, :])
            K_mean[t + 1] = sum(w .* particles[IK_IDX, :])
            C_sd[t + 1] = sqrt(max(sum(w .* (particles[IC_IDX, :] .- C_mean[t + 1]) .^ 2), 0.0))
            K_sd[t + 1] = sqrt(max(sum(w .* (particles[IK_IDX, :] .- K_mean[t + 1]) .^ 2), 0.0))
            acc = zeros(NFIELD)
            for pp in 1:NPRT
                acc .+= w[pp] .* particle_beta(particles, pp, t + 1)
            end
            ensemble_mean_beta[:, t + 1] = acc
        end

        if ess_k < ESS_THRESHOLD
            particles = particles[:, systematic_resample(w, rng_pf)]
            # Jitter everything, parameters included — without it the (C,k)
            # cloud collapses to duplicates and can never adapt again.
            for pp in 1:NPRT
                add_smooth!(view(particles, IDELTA, pp), Lfac, buf, rng_pf, JITTER_DELTA)
                particles[IC_IDX, pp] += JITTER_C * randn(rng_pf)
                particles[IK_IDX, pp] += JITTER_K * randn(rng_pf)
            end
            log_weights .= 0.0
            n_resample += 1
            k < K_TEMPER && logdens!(loglik, particles, t + 1, y_t)
        end
    end

    ess_series[t] = stage_min_ess
    say(@sprintf("   step %2d/%d  ESS = %6.1f  C = %7.1f ± %5.1f  k = %6.2f ± %5.2f  (%.0f s)",
                 t, NSTEP, ess_series[t], C_mean[t + 1], C_sd[t + 1],
                 K_mean[t + 1], K_sd[t + 1], time() - tstart))
end
wall = time() - tstart

# =============================================================================
#  9. SUMMARY
# =============================================================================
rmse_beta = [sqrt(mean((ensemble_mean_beta[:, t] .- truth_beta[:, t]) .^ 2)) for t in 1:NSTEP+1]

say("")
say("-"^78)
say(" RESULTS")
say("-"^78)
say(@sprintf(" C (β at T_ref) : prior %7.1f  →  final %7.1f ± %.1f   TRUTH %7.1f   error %+.1f (%.1f%%)",
             C_mean[1], C_mean[end], C_sd[end], C_TRUE,
             C_mean[end] - C_TRUE, 100 * abs(C_mean[end] - C_TRUE) / C_TRUE))
say(@sprintf(" k (sensitivity): prior %7.2f  →  final %7.2f ± %.2f   TRUTH %7.2f   error %+.2f (%.1f%%)",
             K_mean[1], K_mean[end], K_sd[end], K_TRUE,
             K_mean[end] - K_TRUE, 100 * abs(K_mean[end] - K_TRUE) / K_TRUE))
say(@sprintf(" RMSE(β)        : %.1f  →  %.1f   (%.1f%% reduction)",
             rmse_beta[1], rmse_beta[end], 100 * (1 - rmse_beta[end] / rmse_beta[1])))

# Split RMSE(β) into the part caused by a wrong MAP and the part caused by a
# wrong RESIDUAL. These are the two things the filter is estimating and they
# behave very differently: the map is identified by 16 sensors easily, while δ
# has 1600 degrees of freedom and is largely unconstrained. Without this split a
# rising RMSE(β) looks like filter failure when the map may be perfect.
map_err = zeros(NSTEP + 1); res_err = zeros(NSTEP + 1); dsd_est = zeros(NSTEP + 1); dsd_true = zeros(NSTEP + 1)
for t in 1:NSTEP+1
    map_true = C_TRUE   .- K_TRUE   .* (Tfields[t] .- T_REF)
    map_est  = C_mean[t] .- K_mean[t] .* (Tfields[t] .- T_REF)
    d_true = truth_beta[:, t]          .- map_true
    d_est  = ensemble_mean_beta[:, t]  .- map_est
    map_err[t]  = sqrt(mean((map_est .- map_true) .^ 2))
    res_err[t]  = sqrt(mean((d_est .- d_true) .^ 2))
    dsd_est[t]  = std(d_est); dsd_true[t] = std(d_true)
end
say(@sprintf("   ├─ from the MAP      : %.1f  →  %.1f", map_err[1], map_err[end]))
say(@sprintf("   └─ from the RESIDUAL : %.1f  →  %.1f", res_err[1], res_err[end]))
say(@sprintf(" δ spread       : filter %.1f vs truth %.1f  (filter/truth = %.2f; ≫1 means",
             dsd_est[end], dsd_true[end], dsd_est[end] / max(dsd_true[end], 1e-9)))
say("                  the δ prior is wider than reality and the filter is")
say("                  committing to structure the 16 sensors cannot check)")
say(@sprintf(" ESS            : mean %.1f, min %.1f  (threshold N/2 = %.0f), %d resamples",
             mean(ess_series), minimum(ess_series), ESS_THRESHOLD, n_resample))
say(@sprintf(" WAVI solves    : %d  (base N·T = %d, overhead ×%.2f)",
             WAVI_CALLS[], NPRT * NSTEP, WAVI_CALLS[] / (NPRT * NSTEP)))
say(@sprintf(" wall clock     : %.0f s = %.1f min  (%.3f s per solve)",
             wall, wall / 60, wall / WAVI_CALLS[]))
say("-"^78)

# =============================================================================
# 10. SAVE
# =============================================================================
h5open(joinpath(OUTDIR, "tracking.h5"), "w") do f
    f["truth/beta"]         = reshape(truth_beta, NY, NX, NSTEP + 1)
    f["ensemble_mean/beta"] = reshape(ensemble_mean_beta, NY, NX, NSTEP + 1)
    f["temperature"]        = reshape(hcat(Tfields...), NY, NX, NSTEP + 1)
    f["weights/ess"]        = ess_series
    f["observations"]       = observations
    f["truth_logspeed"]     = truth_logspeed
    f["sensor_indices"]     = collect(SENSORS)
    f["rmse_beta"]          = rmse_beta
    f["rmse_map"]           = map_err
    f["rmse_residual"]      = res_err
    f["delta_sd_filter"]    = dsd_est
    f["delta_sd_truth"]     = dsd_true
    f["params/C_mean"] = C_mean; f["params/C_sd"] = C_sd
    f["params/K_mean"] = K_mean; f["params/K_sd"] = K_sd
    f["params/C_final_particles"] = particles[IC_IDX, :]
    f["params/K_final_particles"] = particles[IK_IDX, :]
    f["params/C_prior_particles"] = particles_C0
    f["params/K_prior_particles"] = particles_K0
    g = create_group(f, "truth_values")
    g["C"] = C_TRUE; g["k"] = K_TRUE; g["T_ref"] = T_REF
end

# =============================================================================
# 11. FIGURES
# =============================================================================
say("")
print(" plotting : ")
gx = collect(0:NX-1) .* (DX / 1000); gy = collect(0:NY-1) .* (DX / 1000)
sx = [((s - 1) ÷ NY) * DX / 1000 for s in SENSORS]
sy = [((s - 1) %  NY) * DX / 1000 for s in SENSORS]
steps = 0:NSTEP

# --- THE KEY FIGURE: parameter recovery ---------------------------------
print("parameter recovery ")
pC = plot(steps, C_mean, ribbon = C_sd, lw = 2, color = :steelblue,
          fillalpha = 0.25, label = "posterior mean ± sd",
          ylabel = "C  (Pa·s/m)", title = "C — β at the reference temperature")
hline!(pC, [C_TRUE], ls = :dash, lw = 2, color = :black, label = "truth")
pK = plot(steps, K_mean, ribbon = K_sd, lw = 2, color = :crimson,
          fillalpha = 0.25, label = "posterior mean ± sd",
          xlabel = "assimilation step (hours)", ylabel = "k  (Pa·s/m per °C)",
          title = "k — sensitivity of β to temperature")
hline!(pK, [K_TRUE], ls = :dash, lw = 2, color = :black, label = "truth")
plot(pC, pK, layout = (2, 1), size = (950, 720),
     left_margin = 10mm, bottom_margin = 5mm)
savefig(joinpath(OUTDIR, "parameter_recovery.png"))

# --- particle cloud, prior vs posterior ---------------------------------
print("| scatter ")
scatter(particles_C0, particles_K0, ms = 3, msw = 0, color = :gray70, alpha = 0.5,
        label = "prior particles", xlabel = "C (Pa·s/m)", ylabel = "k (Pa·s/m per °C)",
        title = "particle cloud in map-parameter space", size = (820, 640),
        legend = :best, left_margin = 6mm, bottom_margin = 5mm)
scatter!(particles[IC_IDX, :], particles[IK_IDX, :], ms = 3, msw = 0,
         color = :crimson, alpha = 0.6, label = "posterior particles")
scatter!([C_TRUE], [K_TRUE], ms = 10, marker = :star5, color = :black, label = "truth")
savefig(joinpath(OUTDIR, "parameter_scatter.png"))

# --- RMSE and ESS --------------------------------------------------------
print("| rmse/ess ")
plot(steps, rmse_beta, lw = 2.5, marker = :circle, ms = 3, color = :crimson, label = "total RMSE(β)",
     xlabel = "assimilation step (hours)", ylabel = "RMSE (Pa·s/m)", ylims = (0, :auto),
     title = @sprintf("β error: %.1f → %.1f  (map %.1f, residual %.1f)",
                      rmse_beta[1], rmse_beta[end], map_err[end], res_err[end]),
     size = (900, 560), left_margin = 6mm, bottom_margin = 5mm, legend = :best)
plot!(steps, map_err, lw = 2, ls = :dash, color = :steelblue, label = "from the MAP (C, k)")
plot!(steps, res_err, lw = 2, ls = :dot, color = :darkorange, label = "from the RESIDUAL δ")
savefig(joinpath(OUTDIR, "rmse_beta.png"))

plot(1:NSTEP, ess_series, lw = 2, marker = :circle, ms = 3, color = :steelblue, label = "ESS",
     xlabel = "assimilation step (hours)", ylabel = "effective sample size",
     title = @sprintf("ESS (mean %.1f, min %.1f)", mean(ess_series), minimum(ess_series)),
     ylims = (0, NPRT), size = (820, 500), left_margin = 6mm, bottom_margin = 5mm)
hline!([ESS_THRESHOLD], ls = :dash, color = :red, label = "resample threshold N/2")
savefig(joinpath(OUTDIR, "ess_tracking.png"))

# --- forcing and response ------------------------------------------------
print("| forcing ")
bt_mean = [mean(reshape(truth_beta[:, t], NY, NX)[IN, IN]) for t in 1:NSTEP+1]
bm_mean = [mean(reshape(ensemble_mean_beta[:, t], NY, NX)[IN, IN]) for t in 1:NSTEP+1]
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
Tb = reshape(truth_beta, NY, NX, NSTEP + 1)
Mb = reshape(ensemble_mean_beta, NY, NX, NSTEP + 1)
allb = vcat(vec(Tb), vec(Mb))
blim = (quantile(allb, 0.02), quantile(allb, 0.98))
berr = Mb .- Tb
belim = quantile(abs.(vec(berr)), 0.98); belim = (-belim, belim)
anim_b = @animate for t in 1:NSTEP+1
    p1 = pan(Tb[:, :, t], "truth β", blim, :viridis, "Pa·s/m")
    p2 = pan(Mb[:, :, t], "ensemble mean β", blim, :viridis, "Pa·s/m")
    p3 = pan(berr[:, :, t], "error (mean − truth)", belim, :balance, "Pa·s/m")
    for p in (p1, p2); scatter!(p, sx, sy, ms = 2.5, color = :red, label = ""); end
    scatter!(p3, sx, sy, ms = 2.5, color = :black, label = "")
    ttl = t == 1 ?
        @sprintf("β — t = 0 (prior)   RMSE = %.1f   T̄ = %.1f °C", rmse_beta[t], Tmean_series[t]) :
        @sprintf("β — hour %d/%d   RMSE = %.1f   T̄ = %.1f °C   k = %.1f (truth %.0f)",
                 t - 1, NSTEP, rmse_beta[t], Tmean_series[t], K_mean[t], K_TRUE)
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 480), plot_title = ttl,
         plot_titlefontsize = 11, left_margin = 9mm, bottom_margin = 6mm,
         right_margin = 3mm, top_margin = 2mm)
end
gif(anim_b, joinpath(OUTDIR, "beta_evolution.gif"), fps = 3)

print("| velocity gif ")
# β is the complete field, so velocities are recovered by re-solving WAVI on the
# saved β — 2(T+1) solves, seconds — rather than storing them during the run.
Strue = Array{Float64}(undef, NY, NX, NSTEP + 1); Smean = similar(Strue)
quiet() do
    for t in 1:NSTEP+1
        Strue[:, :, t] = reshape(wavi_speed_flat(truth_beta[:, t]), NY, NX)
        Smean[:, :, t] = reshape(wavi_speed_flat(ensemble_mean_beta[:, t]), NY, NX)
    end
end
alls = vcat(vec(Strue), vec(Smean))
slim = (quantile(alls, 0.02), quantile(alls, 0.98))
serr = Smean .- Strue
selim = quantile(abs.(vec(serr)), 0.98); selim = (-selim, selim)
srmse = [sqrt(mean(serr[:, :, t] .^ 2)) for t in 1:NSTEP+1]
anim_v = @animate for t in 1:NSTEP+1
    p1 = pan(Strue[:, :, t], "truth speed", slim, :thermal, "m/yr")
    p2 = pan(Smean[:, :, t], "ensemble mean speed", slim, :thermal, "m/yr")
    p3 = pan(serr[:, :, t], "error (mean − truth)", selim, :balance, "m/yr")
    for p in (p1, p2); scatter!(p, sx, sy, ms = 2.5, color = :cyan, label = ""); end
    scatter!(p3, sx, sy, ms = 2.5, color = :black, label = "")
    ttl = t == 1 ?
        @sprintf("velocity — t = 0 (prior)   speed RMSE = %.2f m/yr", srmse[t]) :
        @sprintf("velocity — hour %d/%d   speed RMSE = %.2f m/yr", t - 1, NSTEP, srmse[t])
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 480), plot_title = ttl,
         plot_titlefontsize = 11, left_margin = 9mm, bottom_margin = 6mm,
         right_margin = 3mm, top_margin = 2mm)
end
gif(anim_v, joinpath(OUTDIR, "velocity_evolution.gif"), fps = 3)
println("| done")

say(@sprintf(" speed RMSE     : %.2f  →  %.2f m/yr", srmse[1], srmse[end]))
open(joinpath(OUTDIR, "summary.txt"), "w") do f; write(f, String(take!(LOG))); end
say("")
say(" all outputs → $OUTDIR")
for fn in sort(readdir(OUTDIR)); println("   $fn"); end
