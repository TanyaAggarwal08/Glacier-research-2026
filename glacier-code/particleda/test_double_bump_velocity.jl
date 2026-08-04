# TEST (not a main run) — does the WAVI surface velocity actually respond to a
# structured β field? We swap the pseudo-random-wave prior for the legacy
# double-bump (large amplitude, n_modes=1 → clean large-scale bumps) and look at
# the velocity it produces. This is a physics sanity check, so it does NOT run
# the particle filter — it just evaluates WAVI on three β fields (3 solves):
#
#   truth β      = double bump (center ± amplitude · sin·sin)
#   initial guess = truth + background offset (seed2-style, so guess ≠ truth)
#   uniform β     = center everywhere  (baseline to subtract for the anomaly)
#
# Everything else (grid, sensors, noise, init_std, seeds) is identical to
# experiment_18. The pseudo_random_wave setup is untouched: we only pass
# prior_mode="double_bump" here. Revert simply by not setting it.
#
# Outputs (results/experiment_18_double_bump_test/):
#   beta_field.png        initial guess β | truth β
#   velocity_field.png    initial guess speed | truth speed | uniform baseline  (RAW, for comparison vs pseudo-random-wave)
#   velocity_anomaly.png  initial guess anomaly | truth anomaly  (speed − uniform, isolates the β effect)
#
# RUN UNDER THE DEFAULT ENV (has WAVI):
#   julia glacier-code/particleda/test_double_bump_velocity.jl

const REPO_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
cd(REPO_ROOT)

using Random, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots, Plots.PlotMeasures
include(joinpath(@__DIR__, "ice_experiment_dynamics.jl")); using .IceExpDyn
include(joinpath(@__DIR__, "ice_flow.jl"));               using .IceFlow

const OUT = joinpath("glacier-code", "particleda", "results", "experiment_18_double_bump_test")
mkpath(OUT)

# ── double-bump setup: everything as Exp 18, only the prior changed ───────
#   center 2000, amplitude 1500  → β ∈ [500, 3500]  (drag varies ~7×, big signal)
#   n_modes 1                    → one 2×2 checkerboard of opposite bumps
p = IceExpDyn.Params(
    nx=40, ny=40, x_length=160_000.0, y_length=160_000.0,
    station_filename="glacier-code/particleda/stations_grid_16.txt",
    prior_mode="double_bump", prior_center_beta=2000.0,
    prior_amplitude_beta=1500.0, prior_n_modes=1,
    background_std_beta=300.0, init_std_beta=200.0, process_std_beta=10.0,
    min_beta=10.0, noise_length_scale=15_000.0,
    advection_epsilon=5e-4, n_integration_step=10, advection_type="nonlinear")
model = IceExpDyn.init(p)
nx, ny = p.nx, p.ny; n_state = nx * ny
sensors = model.sensor_indices

# ── the three β fields ────────────────────────────────────────────────────
truth_beta = model.truth_prior_mean                         # double bump
rng_pf = MersenneTwister(42)                                # same seed as Exp 18
particles = zeros(n_state, 500)
for pp in 1:500
    IceExpDyn.sample_initial_state!(view(particles, :, pp), model, rng_pf)
end
guess_beta   = vec(mean(particles; dims=2))                 # initial ensemble mean = "initial guess"
uniform_beta = fill(p.prior_center_beta, n_state)           # baseline for anomaly

# ── WAVI velocity for each (3 solves) ─────────────────────────────────────
quiet(f) = redirect_stdout(f, devnull)
function wavi_speed(state)
    uv = quiet(() -> IceFlow.velocity_flat(state))
    reshape(sqrt.(uv.u .^ 2 .+ uv.v .^ 2), ny, nx)
end
println("Running WAVI on truth / initial-guess / uniform β (3 solves) ...")
truth_spd   = wavi_speed(truth_beta)
guess_spd   = wavi_speed(guess_beta)
uniform_spd = wavi_speed(uniform_beta)

truth_anom = truth_spd .- uniform_spd     # velocity change caused by the β bumps
guess_anom = guess_spd .- uniform_spd

# ── geometry + sensor markers (matches Exp 18 plots) ──────────────────────
dx = p.x_length / nx; dy = p.y_length / ny
gx = collect(0:nx-1) .* (dx / 1000); gy = collect(0:ny-1) .* (dy / 1000)
sx = Float64[]; sy = Float64[]
for idx in sensors
    j = ((idx - 1) % ny) + 1; i = ((idx - 1) ÷ ny) + 1
    push!(sx, (i - 1) * dx / 1000); push!(sy, (j - 1) * dy / 1000)
end
shared_clims(fs...) = (v = vcat(vec.(fs)...); (quantile(v, 0.02), quantile(v, 0.98)))
panel(field, ttl, clims, cmap, mcolor; cbtitle="") =
    let h = heatmap(gx, gy, field; clims=clims, c=cmap, aspect_ratio=1,
                    title=ttl, xlabel="x (km)", ylabel="y (km)", colorbar_title=cbtitle)
        scatter!(h, sx, sy; mc=mcolor, ms=3, msw=0, label=false); h
    end
mm3 = (left_margin=8mm, right_margin=8mm, bottom_margin=6mm, top_margin=3mm)

# β field: initial guess | truth
gb = reshape(guess_beta, ny, nx); tb = reshape(truth_beta, ny, nx)
cb = shared_clims(gb, tb)
savefig(plot(panel(gb, "initial guess β", cb, :viridis, :red),
             panel(tb, "truth β (double bump)", cb, :viridis, :red);
             layout=(1, 2), size=(1150, 460), mm3...),
        joinpath(OUT, "beta_field.png"))

# RAW velocity: initial guess | truth | uniform baseline
cv = shared_clims(guess_spd, truth_spd, uniform_spd)
savefig(plot(panel(guess_spd,   "initial guess WAVI speed", cv, :thermal, :cyan; cbtitle="m/yr"),
             panel(truth_spd,    "truth WAVI speed",         cv, :thermal, :cyan; cbtitle="m/yr"),
             panel(uniform_spd,  "uniform-β baseline speed", cv, :thermal, :cyan; cbtitle="m/yr");
             layout=(1, 3), size=(1700, 460), mm3...),
        joinpath(OUT, "velocity_field.png"))

# ANOMALY: speed − uniform baseline (isolates the β effect), diverging + symmetric
amax = maximum(abs, vcat(vec(truth_anom), vec(guess_anom)))
ca = (-amax, amax)
savefig(plot(panel(guess_anom, "initial guess speed anomaly", ca, :balance, :black; cbtitle="Δ m/yr"),
             panel(truth_anom, "truth speed anomaly (vs uniform β)", ca, :balance, :black; cbtitle="Δ m/yr");
             layout=(1, 2), size=(1150, 460), mm3...),
        joinpath(OUT, "velocity_anomaly.png"))

# ── numbers: does velocity track β? ───────────────────────────────────────
@printf("\nβ range           : truth [%.0f, %.0f]  guess [%.0f, %.0f]\n",
        minimum(tb), maximum(tb), minimum(gb), maximum(gb))
@printf("WAVI speed range  : truth [%.0f, %.0f]  uniform [%.0f, %.0f] m/yr\n",
        minimum(truth_spd), maximum(truth_spd), minimum(uniform_spd), maximum(uniform_spd))
@printf("speed anomaly     : truth [%.0f, %.0f] m/yr  (±%.0f)\n",
        minimum(truth_anom), maximum(truth_anom), amax)
# anti-correlation: higher β (more drag) should give NEGATIVE speed anomaly
r = cor(vec(tb) .- p.prior_center_beta, vec(truth_anom))
@printf("corr(β−center, speed anomaly) = %.3f   (negative ⇒ higher β → slower ice)\n", r)
println("\nSaved → $OUT")
for fn in ("beta_field.png", "velocity_field.png", "velocity_anomaly.png"); println("  $fn"); end
