# Experiment 18 — the full "base" setup with WAVI as the observation operator.
#
#   upwind advection (nonlinear, v = 1 + ε·β)  +  pseudo-random-wave prior
#   +  likelihood tempering (K stages)  +  WAVI ice-flow observation operator.
#
# β is the state, evolving by first-order upwind advection with a β-dependent
# velocity. The observation operator is the REAL ice-sheet solver:
#     h(β) = log( |u_WAVI(β)| )   at the sensor cells,
# applied PER PARTICLE inside the PF likelihood — not post-hoc on the mean.
#
# Tempering (Del Moral–Doucet–Jasra 2006): the likelihood is introduced in K
# stages φ_k = k/K, resampling (+ RPF jitter) between stages when ESS < N/2,
# with the WAVI likelihood recomputed at moved particle positions after each
# intra-step resample. That recompute is the dominant cost, so the script
# counts every WAVI solve and reports the effective cost for extrapolation.
#
# Dynamics from ice_experiment_dynamics.jl (ParticleDA-free port of
# glacier_model.jl, validated bit-for-bit on both advection branches), because
# WAVI and ParticleDA will not co-load in one environment.
#
# RUN UNDER THE DEFAULT ENV (has WAVI):
#   julia glacier-code/particleda/experiment_18_wavi_obs.jl [N T K SIGMA EPS MODE]
# MODE = "calib" skips plots/saving and just reports timing.

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Random, Statistics, LinearAlgebra, Printf, HDF5, Base.Threads
ENV["GKSwstype"] = "100"
using Plots
include(joinpath(@__DIR__, "ice_experiment_dynamics.jl")); using .IceExpDyn
include(joinpath(@__DIR__, "ice_flow.jl"));               using .IceFlow
# Parallelism: the WAVI likelihood is embarrassingly parallel over particles.
# Thread over particles (Threads.@spawn) and keep each WAVI solve single-BLAS-
# threaded to avoid oversubscription. Concurrent WAVI calls verified bit-for-bit
# identical to serial; measured ~2.5× speedup (GC-bound, WAVI allocs per call).
LinearAlgebra.BLAS.set_num_threads(1)

# ── parameters (ARGS-overridable) ────────────────────────────────────────
_arg(i, default) = length(ARGS) >= i ? ARGS[i] : default
const NPRT      = parse(Int,     _arg(1, "1000"))
const T         = parse(Int,     _arg(2, "50"))
const K_TEMPER  = parse(Int,     _arg(3, "10"))
const SIGMA_OBS = parse(Float64, _arg(4, "0.5"))     # log-speed units
const EPS_ADV   = parse(Float64, _arg(5, "5e-4"))    # nonlinear advection coeff
const MODE      = _arg(6, "full")
const CALIB     = (MODE == "calib")

const SEED_PF  = parse(Int, _arg(7, "42"))
const SEED_OBS = parse(Int, _arg(8, "123"))
const RUN_TAG  = _arg(9, "experiment_18_wavi_obs")   # output dir name (seeds → replicate)
# Prior family. Default keeps the pseudo_random_wave setup untouched; pass
# "double_bump" (+ amplitude, n_modes) to run the legacy sinusoid truth instead.
const PRIOR_MODE   = _arg(10, "pseudo_random_wave")           # or "double_bump"
const PRIOR_AMP    = parse(Float64, _arg(11, "2000.0"))       # double-bump amplitude
const PRIOR_NMODES = parse(Int,     _arg(12, "3"))            # double-bump periods/axis
const ESS_THRESHOLD = 0.5 * NPRT
const SIGMA_JITTER  = 20.0

const OUT = joinpath("glacier-code", "particleda", "results", RUN_TAG)
const _SSD_BASE = "/Volumes/ZX20/USRA 2026"
mkpath(OUT)

p = IceExpDyn.Params(
    nx=40, ny=40, x_length=160_000.0, y_length=160_000.0,
    station_filename="glacier-code/particleda/stations_grid_16.txt",
    prior_center_beta=2000.0, prior_signal_scale_beta=300.0,
    background_std_beta=300.0, prior_max_wavenumber=2,
    prior_truth_seed=11, prior_background_seed=29,
    prior_mode=PRIOR_MODE, prior_amplitude_beta=PRIOR_AMP, prior_n_modes=PRIOR_NMODES,
    init_std_beta=200.0, process_std_beta=10.0, min_beta=10.0,
    advection_epsilon=EPS_ADV, n_integration_step=10, time_step=3600.0,
    noise_length_scale=15_000.0, advection_type="nonlinear")   # nonlinear upwind
model = IceExpDyn.init(p)
sensors = model.sensor_indices
n_state = p.nx * p.ny
n_obs   = length(sensors)
nx, ny  = p.nx, p.ny
@assert n_obs == 16
phi = collect(0:K_TEMPER) ./ K_TEMPER

println("Experiment 18 — WAVI obs operator, upwind+nonlinear+tempering")
println("N=$NPRT T=$T K=$K_TEMPER σ_obs=$SIGMA_OBS ε=$EPS_ADV advection=nonlinear(upwind) mode=$MODE")
println("SEED_PF=$SEED_PF SEED_OBS=$SEED_OBS run_tag=$RUN_TAG")
println("prior_mode=$PRIOR_MODE amplitude=$PRIOR_AMP n_modes=$PRIOR_NMODES")

# SSD output dir + periodic checkpoint (long runs must survive a crash).
const CKPT_EVERY = 5
const EXT_OUT = let cand = joinpath(_SSD_BASE, RUN_TAG)
    try; mkpath(cand); cand; catch; OUT; end
end

# ── WAVI solve counter + thread-safe stdout suppression ──────────────────
# WAVI prints solver progress to stdout; redirect_stdout is process-global, so
# it is applied ONCE around each WAVI region (never per-call inside threads,
# which would race). The counter is atomic (incremented from worker threads).
quiet(f) = redirect_stdout(f, devnull)
const WAVI_CALLS = Threads.Atomic{Int}(0)
function wavi_logspeed(state::AbstractVector)
    Threads.atomic_add!(WAVI_CALLS, 1)
    uv = IceFlow.velocity_flat(state)
    speed = sqrt.(uv.u .^ 2 .+ uv.v .^ 2)
    return log.(max.(speed[sensors], 1e-6)), speed
end
logspeed_at_sensors(state) = wavi_logspeed(state)[1]

function systematic_resample(w::AbstractVector, rng::AbstractRNG)
    N = length(w); c = cumsum(w); u0 = rand(rng)/N
    idx = Vector{Int}(undef, N); j = 1
    for i in 1:N
        u = u0 + (i-1)/N
        while j < N && c[j] < u; j += 1; end
        idx[i] = j
    end
    return idx
end

# ── truth + WAVI observations ────────────────────────────────────────────
println("Generating truth trajectory + WAVI observations ...")
rng_truth = MersenneTwister(SEED_OBS)
truth_states = zeros(n_state, T+1)
observations = zeros(n_obs, T)
truth_logspeed = zeros(n_obs, T)
s_truth = copy(model.truth_prior_mean); s_truth .= max.(s_truth, p.min_beta)
truth_states[:, 1] = s_truth
quiet() do
    for t in 1:T
        IceExpDyn.update_state_deterministic!(s_truth, model)
        IceExpDyn.update_state_stochastic!(s_truth, model, rng_truth)
        truth_states[:, t+1] = s_truth
        h = logspeed_at_sensors(s_truth)
        truth_logspeed[:, t] = h
        observations[:, t] = h .+ SIGMA_OBS .* randn(rng_truth, n_obs)
    end
end

# ── tempered bootstrap PF with WAVI likelihood ──────────────────────────
println("Running tempered PF ...")
rng_pf = MersenneTwister(SEED_PF)
particles = zeros(n_state, NPRT)
for pp in 1:NPRT
    IceExpDyn.sample_initial_state!(view(particles, :, pp), model, rng_pf)
end
all_particles = CALIB ? zeros(0,0,0) : zeros(n_state, NPRT, T+1)
CALIB || (all_particles[:, :, 1] = particles)
ensemble_mean = zeros(n_state, T+1)
ensemble_mean[:, 1] = mean(particles; dims=2)[:, 1]
ess_series    = zeros(T)
mean_logspeed = zeros(n_obs, T)
log_weights   = zeros(NPRT)
loglik        = zeros(NPRT)
inv2σ2 = 1.0 / (2 * SIGMA_OBS^2)
n_resample = 0

# Threaded over particles: each WAVI solve is independent (RNG-free); dest[pp]
# writes are per-index so no races. One redirect_stdout wraps the whole region.
function logdens!(dest, parts, y_t)
    quiet() do
        @sync for pp in 1:NPRT
            Threads.@spawn begin
                h = logspeed_at_sensors(view(parts, :, pp))
                dest[pp] = -sum((h .- y_t).^2) * inv2σ2
            end
        end
    end
end

tstart = time()
for t in 1:T
    global particles, log_weights, n_resample
    for pp in 1:NPRT
        IceExpDyn.update_state_deterministic!(view(particles, :, pp), model)
        IceExpDyn.update_state_stochastic!(view(particles, :, pp), model, rng_pf)
    end
    y_t = view(observations, :, t)
    logdens!(loglik, particles, y_t)                        # N WAVI solves

    stage_min_ess = Inf
    for k in 1:K_TEMPER
        dphi = phi[k+1] - phi[k]
        log_weights .+= dphi .* loglik
        lmax = maximum(log_weights)
        w = exp.(log_weights .- lmax); w ./= sum(w)
        ess_k = 1.0 / sum(w .^ 2)
        stage_min_ess = min(stage_min_ess, ess_k)
        if k == K_TEMPER
            ensemble_mean[:, t+1] = particles * w
        end
        if ess_k < ESS_THRESHOLD
            idx = systematic_resample(w, rng_pf)
            particles = particles[:, idx]
            for pp in 1:NPRT
                IceExpDyn.apply_noise!(view(particles, :, pp), model, rng_pf, SIGMA_JITTER)
                for i in 1:n_state; particles[i, pp] = max(particles[i, pp], p.min_beta); end
            end
            log_weights .= 0.0
            n_resample += 1
            if k < K_TEMPER
                logdens!(loglik, particles, y_t)            # recompute at moved positions
            end
        end
    end
    ess_series[t] = stage_min_ess
    mean_logspeed[:, t] = quiet(() -> logspeed_at_sensors(ensemble_mean[:, t+1]))
    CALIB || (all_particles[:, :, t+1] = particles)
    @printf("  step %2d/%d  ESS=%.1f  WAVIcalls=%d  (%.0fs)\n",
            t, T, ess_series[t], WAVI_CALLS[], time()-tstart)
    flush(stdout)
    # checkpoint: overwrite a recovery file every CKPT_EVERY steps so a crash
    # in a multi-hour run loses at most CKPT_EVERY steps of work.
    if !CALIB && (t % CKPT_EVERY == 0 || t == T)
        tmp = joinpath(EXT_OUT, "checkpoint.h5.tmp")
        h5open(tmp, "w") do f
            f["steps_done"]         = t
            f["truth/beta"]         = reshape(truth_states, ny, nx, T+1)
            f["ensemble_mean/beta"] = reshape(ensemble_mean, ny, nx, T+1)
            f["particles_all/beta"] = reshape(all_particles, ny, nx, NPRT, T+1)
            f["weights/ess"]        = ess_series
            f["observations"]       = observations
            f["truth_logspeed"]     = truth_logspeed
            f["mean_logspeed"]      = mean_logspeed
            f["sensor_indices"]     = collect(sensors)
        end
        mv(tmp, joinpath(EXT_OUT, "checkpoint.h5"); force=true)
    end
end
wall = time() - tstart

rmse_beta = [sqrt(mean((ensemble_mean[:, t] .- truth_states[:, t]).^2)) for t in 1:T+1]
ess_min, ess_argmin = findmin(ess_series)
@printf("\nmean ESS=%.1f  min ESS=%.1f @t=%d  initial RMSE=%.1f  final RMSE=%.1f\n",
        mean(ess_series), ess_min, ess_argmin, rmse_beta[1], rmse_beta[end])
@printf("WAVI solves total=%d  (base N*T=%d, tempering overhead ×%.2f)  n_resample=%d\n",
        WAVI_CALLS[], NPRT*T, WAVI_CALLS[]/(NPRT*T), n_resample)
@printf("PF wall=%.0fs = %.2f min ; per WAVI solve=%.3fs\n", wall, wall/60, wall/WAVI_CALLS[])

if CALIB
    solves_per_ps = WAVI_CALLS[] / (NPRT*T)
    persolve = wall / WAVI_CALLS[]
    println("\n=== CALIBRATION extrapolation (per-solve=$(round(persolve,digits=2))s, overhead ×$(round(solves_per_ps,digits=2))) ===")
    for (Nf, Tf) in [(1000,50),(500,50),(250,50),(100,50)]
        est = Nf*Tf*solves_per_ps*persolve
        @printf("  N=%-5d T=%-3d  ≈ %.1f hr (%.0f WAVI solves)\n", Nf, Tf, est/3600, Nf*Tf*solves_per_ps)
    end
    println("Calibration done (no plots/saving).")
    exit(0)
end

# ── save particles to SSD ────────────────────────────────────────────────
h5open(joinpath(EXT_OUT, "tracking.h5"), "w") do f
    f["truth/beta"]         = reshape(truth_states, ny, nx, T+1)
    f["ensemble_mean/beta"] = reshape(ensemble_mean, ny, nx, T+1)
    f["particles_all/beta"] = reshape(all_particles, ny, nx, NPRT, T+1)
    f["weights/ess"]        = ess_series
    f["observations"]       = observations
    f["truth_logspeed"]     = truth_logspeed
    f["mean_logspeed"]      = mean_logspeed
    f["sensor_indices"]     = collect(sensors)
    g = create_group(f, "parameters")
    for (k,v) in (("N",NPRT),("T",T),("K",K_TEMPER),("sigma_obs",SIGMA_OBS),
                  ("eps",EPS_ADV),("nx",nx),("ny",ny),("time_step",p.time_step))
        attributes(g)[k] = v
    end
end
println("Saved particles → $(joinpath(EXT_OUT, "tracking.h5"))")

# ── field snapshots (few extra WAVI solves) ──────────────────────────────
truth_speed_final = reshape(quiet(() -> wavi_logspeed(truth_states[:, end])[2]), ny, nx)
mean_speed_final  = reshape(quiet(() -> wavi_logspeed(ensemble_mean[:, end])[2]), ny, nx)
truth_beta_final  = reshape(truth_states[:, end], ny, nx)
mean_beta_final   = reshape(ensemble_mean[:, end], ny, nx)

sx = Float64[]; sy = Float64[]; dx = p.x_length/nx; dy = p.y_length/ny
for idx in sensors
    j = ((idx-1)%ny)+1; i = ((idx-1)÷ny)+1
    push!(sx, (i-1)*dx/1000); push!(sy, (j-1)*dy/1000)
end
gx = collect(0:nx-1).*(dx/1000); gy = collect(0:ny-1).*(dy/1000)
t_axis_h = collect(0:T) .* p.time_step ./ 3600
part_grid = reshape(all_particles, ny, nx, NPRT, T+1)

# ESS
p_ess = plot(1:T, ess_series; xlabel="timestep", ylabel="min-stage ESS", ylim=(0,NPRT),
             title="Exp 18 — ESS (WAVI obs, upwind+nonlinear+tempering, N=$NPRT)",
             lw=2, color=:seagreen, label="ESS")
hline!(p_ess,[ESS_THRESHOLD];ls=:dash,color=:gray,label="N/2")
scatter!(p_ess,[ess_argmin],[ess_min];color=:red,ms=6,label=@sprintf("min %.0f @t=%d",ess_min,ess_argmin))
savefig(p_ess, joinpath(OUT,"ess_tracking.png"))

# RMSE
savefig(plot(t_axis_h, rmse_beta; xlabel="time (h)", ylabel="RMSE(β)",
             title="Exp 18 — global RMSE(β), WAVI operator", lw=2.5, color=:darkgreen, legend=false),
        joinpath(OUT,"rmse_beta.png"))

# β field truth vs mean
cb=(quantile(vec(truth_beta_final),0.02),quantile(vec(truth_beta_final),0.98))
hb1=heatmap(gx,gy,truth_beta_final;clims=cb,c=:viridis,aspect_ratio=1,title="truth β (final)",xlabel="x (km)",ylabel="y (km)")
hb2=heatmap(gx,gy,mean_beta_final;clims=cb,c=:viridis,aspect_ratio=1,title="mean β (final)",xlabel="x (km)",ylabel="y (km)")
scatter!(hb1,sx,sy;mc=:red,ms=3,msw=0,label=false); scatter!(hb2,sx,sy;mc=:red,ms=3,msw=0,label=false)
savefig(plot(hb1,hb2;layout=(1,2),size=(1100,430)), joinpath(OUT,"beta_field_truth_vs_mean.png"))

# WAVI velocity field truth vs mean
cv=(quantile(vec(truth_speed_final),0.02),quantile(vec(truth_speed_final),0.98))
hv1=heatmap(gx,gy,truth_speed_final;clims=cv,c=:thermal,aspect_ratio=1,title="truth WAVI speed (final)",xlabel="x (km)",ylabel="y (km)",colorbar_title="m/yr")
hv2=heatmap(gx,gy,mean_speed_final;clims=cv,c=:thermal,aspect_ratio=1,title="mean WAVI speed (final)",xlabel="x (km)",ylabel="y (km)",colorbar_title="m/yr")
scatter!(hv1,sx,sy;mc=:cyan,ms=3,msw=0,label=false); scatter!(hv2,sx,sy;mc=:cyan,ms=3,msw=0,label=false)
savefig(plot(hv1,hv2;layout=(1,2),size=(1100,430)), joinpath(OUT,"velocity_field_truth_vs_mean.png"))

# cross-section β(x) with particle band at the row carrying most sensors
sensor_rows=[((idx-1)%ny)+1 for idx in sensors]; rc=Dict{Int,Int}()
for r in sensor_rows; rc[r]=get(rc,r,0)+1; end
row_mid=argmax(rc); row_y_km=(row_mid-1)*(dy)/1000
xs_km=collect(0:nx-1).*(dx/1000)
son_x=[(((idx-1)÷ny))*(dx/1000) for idx in sensors if (((idx-1)%ny)+1)==row_mid]
Pf=part_grid[row_mid,:,:,end]
lo90=[quantile(Pf[i,:],0.05) for i in 1:nx]; hi90=[quantile(Pf[i,:],0.95) for i in 1:nx]
lo50=[quantile(Pf[i,:],0.25) for i in 1:nx]; hi50=[quantile(Pf[i,:],0.75) for i in 1:nx]
clb=(quantile(vec(truth_states),0.02),quantile(vec(truth_states),0.98))
p_cs=plot(xs_km,hi90;fillrange=lo90,fillalpha=0.18,color=:steelblue,lw=0,label="particles 5–95%",
          xlabel="x (km)",ylabel="β (Pa·s/m)",title=@sprintf("Exp18 — β cross-section y=%.0f km (final)",row_y_km),
          ylim=(minimum(clb),maximum(clb)),legend=:topright)
plot!(p_cs,xs_km,hi50;fillrange=lo50,fillalpha=0.30,color=:steelblue,lw=0,label="particles 25–75%")
plot!(p_cs,xs_km,mean_beta_final[row_mid,:];lw=2,ls=:dash,color=:navy,label="ensemble mean")
plot!(p_cs,xs_km,truth_beta_final[row_mid,:];lw=3,color=:black,label="truth")
for (i_s,sxk) in enumerate(son_x); vline!(p_cs,[sxk];color=:red,ls=:dot,lw=1.2,label=(i_s==1 ? "sensor x" : false)); end
savefig(p_cs, joinpath(OUT,"crosssection_final.png"))

# sensor trail: β(t) at a probe cell (first sensor) with particles+truth+mean
probe = sensors[1]
p_tr=plot(;xlabel="time (h)",ylabel="β (Pa·s/m)",title="Exp18 — β(t) at sensor cell $probe",legend=:topright)
nshow=min(NPRT,150); si=round.(Int,range(1,NPRT;length=nshow))
for (n,ip) in enumerate(si)
    plot!(p_tr,t_axis_h,all_particles[probe,ip,:];lw=0.4,color=:gray,alpha=0.08,label=(n==1 ? "particles" : false))
end
plot!(p_tr,t_axis_h,ensemble_mean[probe,:];lw=2.5,ls=:dash,color=:steelblue,label="mean")
plot!(p_tr,t_axis_h,truth_states[probe,:];lw=3,color=:black,label="truth")
savefig(p_tr, joinpath(OUT,"sensor_trail.png"))

# observation-space fit
show_s=[1,8,16]
p_obs=plot(;xlabel="time (h)",ylabel="log WAVI speed",title="Exp18 — observation fit at sample sensors",legend=:outertopright)
for (c,k) in zip([:steelblue,:seagreen,:firebrick],show_s)
    plot!(p_obs,t_axis_h[2:end],truth_logspeed[k,:];lw=2.5,color=c,label="sensor $k truth")
    plot!(p_obs,t_axis_h[2:end],mean_logspeed[k,:];lw=2,ls=:dash,color=c,label="sensor $k est")
    scatter!(p_obs,t_axis_h[2:end],observations[k,:];mc=c,ms=3,msw=0,label=false)
end
savefig(p_obs, joinpath(OUT,"obs_fit.png"))

println("\nSaved plots → $OUT")
for fn in ("ess_tracking.png","rmse_beta.png","beta_field_truth_vs_mean.png",
           "velocity_field_truth_vs_mean.png","crosssection_final.png","sensor_trail.png","obs_fit.png")
    println("  $fn")
end
println("Done.")
