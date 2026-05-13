# pf_with_wavi_beta_estimation_logbeta.jl
# Goal: estimate basal friction field β from surface velocity observations.
# Changes vs your version (guided by typical glaciology DA practice):
#   • Run PF in θ = log(β) space → positivity + roughly Gaussian.
#   • Rescale surrogate so ux ≈ O(1) → makes 5% noise (0.05) realistic.
#   • Fix sensor plotting (linear index → (row, col) uses Ny, not Nx).
#   • Calmer β dynamics noise (≈0.7% per step in log space).
#   • Safer initial spread (~5% in log space).
#   • Optional, guarded rejuvenation after resampling (if PF exposes particles).

# importing imp libraries
using Random
using LinearAlgebra
using Statistics
using Distributions
using Plots
using LowLevelParticleFilters
using WAVI  # ok if installed; we won't call it unless use_surrogate=false

# 
function main(; 
    Nx::Int = 40, Ny::Int = 40,            # grid
    T::Int = 100,                           # time steps
    #more the number of particles, more time for wavi to run
    Np::Int = 10000,                          # particles (log-space helps, so fewer ok)
    sensor_stride::Int = 16,                # pick every k-th grid index as a sensor
    use_surrogate::Bool = true,             # true = fast and practical
    save_dir::String = "./pf_wavi_outputs_withsurrogatetest",  # output dir
    # imp for reproducibility
    rng_seed::Int = 1234
)
    Random.seed!(rng_seed)
    isdir(save_dir) || mkpath(save_dir)

    # -----------------------------
    # Grid + simple geometry
    # -----------------------------
    N = Nx * Ny
    L = 160000.0
    x = range(0.0, stop=L, length=Nx)
    y = range(0.0, stop=L, length=Ny)
    # Create meshgrid for plotting
    X = [xi for yi in y, xi in x]  # Ny×Nx
    Y = [yi for yi in y, xi in x]  # Ny×Nx

    # surface/bed (only needed if you later use WAVI)
    z_s(x) = 1060 * sqrt(max(1.0 - (x / L), 0.0))
    z_s_mat = z_s.(X)
    z_b_mat = zeros(Ny, Nx)

    # -----------------------------
    # True β field and prior β
    # -----------------------------
    ω = 2π / L
    beta_true_mat  = 1000 .+ 1000 .* sin.(ω .* X) .* sin.(ω .* Y)          # ground truth
    # prior β (different, for initial conditions)
    beta_prior_mat = 1000 .+  500 .* sin.(ω .* X) .* sin.(ω .* Y)          # prior (different)

    # -----------------------------
    # Optional WAVI model (only used if use_surrogate=false)
    # -----------------------------
    grid_L(; L, nx = 40, ny = 40) = WAVI.Grid(nx = nx, ny = ny, dx = L/nx, dy = L/ny,
                                              y0 = 0.0, x0 = 0.0, u_iszero = ["north"], v_iszero = ["south"])
    grid = grid_L(L=L, nx=Nx, ny=Ny)
    initial_conditions = WAVI.InitialConditions(initial_thickness = max.(z_s_mat .- z_b_mat, 0.0))
    base_params = WAVI.Params(beta = copy(beta_true_mat))
    base_model = WAVI.Model(grid=grid, bed_elevation = z_b_mat, initial_conditions = initial_conditions, params = base_params)

    # -----------------------------
    # Sensors + noises
    # -----------------------------
    sensor_indices = collect(1:sensor_stride:N)  # linear indices through Ny×Nx (column-major)

    # Observation noise (velocity units). With scaled surrogate (ux ~ 1), 5% noise is realistic.
    vel_noise_std = 0.05
    measurement_noise = MvNormal(zeros(length(sensor_indices)), vel_noise_std^2 * I(length(sensor_indices)))

    # Correct linear-index → (row, col) for Ny×Nx arrays, and map to x/y coords
    idx_to_rc(i, Ny) = ((i - 1) % Ny) + 1, ((i - 1) ÷ Ny) + 1
    sensor_rc = [idx_to_rc(i, Ny) for i in sensor_indices]  # (row, col)
    x_coords = [x[c] for (r, c) in sensor_rc]
    y_coords = [y[r] for (r, c) in sensor_rc]

    # Process noise in log space (θ = log β). ~0.7% per step.
    # Too large (>0.05) → particles become too spread, estimates are noisy and unstable.
    process_noise_std_θ = 0.007
    process_noise = MvNormal(zeros(N), process_noise_std_θ^2 * I(N))

    # Initial particle spread around prior in log space (~5%)
    θ0 = log.(vec(beta_prior_mat))
    # % error relative to prior.
    init_std_θ = 0.05
    initial_state_dist = MvNormal(θ0, init_std_θ^2 * I(N))

    # -----------------------------
    # Surrogate: β ⟶ (ux, uy) (fast), scaled so ux ≈ O(1)
    # -----------------------------
    function surrogate_iceflow(beta_vec::AbstractVector{<:Real})
        β = reshape(beta_vec, Ny, Nx)
        # light smoothing (5-point)
        # because in reality β is not so noisy so we are creating a blur effect to avoid spikes
        Ms = copy(β)
        # for j in 2:Ny-1, i in 2:Nx-1
            # Ms[j,i] = (β[j,i] + β[j-1,i] + β[j+1,i] + β[j,i-1] + β[j,i+1]) / 5
        # end
        speed = 1e3 ./ (Ms .+ 1e-6) 
                  # inverse relation to β, scaled
        # x-direction velocity
        ux = vec(speed)
        # little velocity in y-direction t 
        uy = vec(0.2 .* speed .^ 0.8)
        return ux, uy
    end

    # -----------------------------
    # WAVI forward: β ⟶ (ux, uy) (expensive)
    # -----------------------------
    function wavi_iceflow!(beta_vec::AbstractVector{<:Real}, model::WAVI.Model)
        β_mat = reshape(beta_vec, Ny, Nx)
        model.params.beta .= β_mat
        WAVI.update_state!(model)
        ux = vec(model.fields.gh.u)
        uy = vec(model.fields.gh.v)
        return ux, uy
    end

    # -----------------------------
    # Measurement operator h(θ): θ=logβ ⟶ observed ux at sensors
    # -----------------------------
    function measurement(theta_vec, u, p, t)
        β_vec = exp.(theta_vec)
        if use_surrogate
            ux, _ = surrogate_iceflow(β_vec)
            return ux[sensor_indices]
        else
            local_model = deepcopy(base_model)
            ux, _ = wavi_iceflow!(β_vec, local_model)
            return ux[sensor_indices]
        end
    end

    # -----------------------------
    # Dynamics f(θ): advect β then return log(β)
    # (PF will add process noise from process_noise in log space)
    # -----------------------------
    function dynamics(theta_vec, u, p, t)


        # non linear 
        ε = 0.0005
        dx = L / Nx
        β = reshape(exp.(theta_vec), Ny, Nx)
        β_new = similar(β)

        # Periodic boundary
        function wrap(i, n)
            return mod1(i, n)
        end

        max_speed = maximum(1 .+ ε .* β)
        dt = 0.2 * dx / max_speed  # CFL condition

        for j in 1:Ny
            for i in 1:Nx
                im = wrap(i - 1, Nx)
                dβdx = (β[j, i] - β[j, im]) / dx
                velocity = 1 + ε * β[j, i]
                β_new[j, i] = β[j, i] - velocity * dt * dβdx
            end
        end

        return log.(vec(β_new))
        # non linear

    end

    # -----------------------------
    # Build Particle Filter (state is θ = log β)
    # -----------------------------
    pf = ParticleFilter(Np, dynamics, measurement, process_noise, measurement_noise, initial_state_dist)

    # -----------------------------
    # Generate synthetic truth + observations
    # -----------------------------
    true_states_θ = Vector{Vector{Float64}}(undef, T)
    true_vels     = Vector{Vector{Float64}}(undef, T)
    observations  = Vector{Vector{Float64}}(undef, T)

    # storing true states in log space
    θ_true = log.(vec(beta_true_mat))
    for t in 1:T
        # get beta at time t using dynamics 
        θ_true = dynamics(θ_true, nothing, nothing, t) + rand(process_noise)
        true_states_θ[t] = θ_true
        β_true_t = exp.(θ_true)
        # get velocity at time t using either surrogate or WAVI
        if use_surrogate
            ux, _ = surrogate_iceflow(β_true_t)
        else
            wavi_truth_model = deepcopy(base_model)
            ux, _ = wavi_iceflow!(β_true_t, wavi_truth_model)
        end
        true_vels[t] = ux
        observations[t] = ux[sensor_indices] + rand(measurement_noise)
    end

    # -----------------------------
    # Run PF
    # -----------------------------
    estimates_θ = Vector{Vector{Float64}}(undef, T)
    estimated_vels = Vector{Vector{Float64}}(undef, T)

    for t in 1:T
        pf(nothing, observations[t])f
        θ_est = weighted_mean(pf)
        estimates_θ[t] = θ_est

        β_est = exp.(θ_est)
        if use_surrogate
            ux_e, _ = surrogate_iceflow(β_est)
        else
            model_e = deepcopy(base_model)
            ux_e, _ = wavi_iceflow!(β_est, model_e)
        end
        estimated_vels[t] = ux_e

        # Optional: tiny rejuvenation if the PF object exposes particles
        try
            if hasproperty(pf, :particles)
                pf.particles .= pf.particles .+ (0.003 .* randn(size(pf.particles)))
            end
        catch
            # ignore if PF internals differ
        end
        @info "t = $t finished"
    end

    # -----------------------------
    # Visualizations
    # -----------------------------
    β_true_store = [reshape(exp.(true_states_θ[t]), Ny, Nx) for t in 1:T]
    β_est_store  = [reshape(exp.(estimates_θ[t]),   Ny, Nx) for t in 1:T]
    β_err_store  = [β_true_store[t] .- β_est_store[t] for t in 1:T]

    climβ = (0,2000)
    climE = (minimum([minimum(β_err_store[t]) for t in 1:T]), maximum([maximum(β_err_store[t]) for t in 1:T]))

    # True β GIF
    anim1 = @animate for t in 1:T
        heatmap(x,y,β_true_store[t]; title="True β  (t=$t)", aspect_ratio=1, clims=climβ, colorbar=true)
        scatter!(x_coords, y_coords, markershape=:circle, color=:red, label="Observations", markersize=1)
    end
    gif(anim1, joinpath(save_dir, "true_beta.gif"), fps=6)

    # Estimated β GIF
    anim2 = @animate for t in 1:T
        heatmap(x,y,β_est_store[t]; title="Estimated β  (t=$t)", aspect_ratio=1, clims=climβ, colorbar=true)
        scatter!(x_coords, y_coords, markershape=:circle, color=:red, label="Observations", markersize=1)
    end
    gif(anim2, joinpath(save_dir, "est_beta.gif"), fps=6)

    # Error β GIF
    anim3 = @animate for t in 1:T
        heatmap(x,y,β_err_store[t]; title="Error β = est − true  (t=$t)", aspect_ratio=1, clims=climE, colorbar=true)
        scatter!(x_coords, y_coords, markershape=:circle, color=:red, label="Observations", markersize=1)
    end
    gif(anim3, joinpath(save_dir, "error_beta.gif"), fps=6)

    # RMSE over time (β global), (velocity at sensors)
    rmse_beta = zeros(T)
    rmse_vel  = zeros(T)
    for t in 1:T
        β_true_t = exp.(true_states_θ[t])
        β_est_t  = exp.(estimates_θ[t])
        rmse_beta[t] = sqrt(mean((β_est_t .- β_true_t).^2))
        rmse_vel[t]  = sqrt(mean((estimated_vels[t][sensor_indices] .- true_vels[t][sensor_indices]).^2))
    end

    p1 = plot(1:T, rmse_beta; xlabel="Time step", ylabel="RMSE(β)", title="Global RMSE of β", legend=false)
    savefig(p1, joinpath(save_dir, "rmse_beta.png"))

    p2 = plot(1:T, rmse_vel; xlabel="Time step", ylabel="RMSE(velocity @ sensors)", title="Sensor-velocity RMSE", legend=false)
    savefig(p2, joinpath(save_dir, "rmse_vel.png"))

    # Pointwise β trace at first sensor location (for intuition)
    target_index = sensor_indices[1]
    true_vals = [exp.(true_states_θ[t])[target_index] for t in 1:T]
    est_vals  = [exp.(estimates_θ[t])[target_index]   for t in 1:T]

    p3 = plot(1:T, true_vals; label="True β", linewidth=2, title="β at first sensor index")
    plot!(1:T, est_vals;  label="Estimated β", linestyle=:dash)
    savefig(p3, joinpath(save_dir, "beta_pointwise.png"))

    println("\n✅ Done. Saved to: $(abspath(save_dir))")
    println("   - true_beta.gif, est_beta.gif, error_beta.gif")
    println("   - rmse_beta.png, rmse_vel.png, beta_pointwise.png")

    # Spatial error map at final time
p4 = heatmap(x, y, β_err_store[T]; title="Final β Error", aspect_ratio=1, clims=climE, colorbar=true)
scatter!(x_coords, y_coords, markershape=:circle, color=:red, label="Observations", markersize=1)
savefig(p4, joinpath(save_dir, "final_beta_error.png"))

# Particle variance over time
particle_var = zeros(T)
for t in 1:T
    if hasproperty(pf, :particles)
        particle_var[t] = mean(var(pf.particles, dims=2))  # Mean variance across dimensions
    end
end
p5 = plot(1:T, particle_var; xlabel="Time step", ylabel="Mean particle variance", title="Particle Variance")
savefig(p5, joinpath(save_dir, "particle_variance.png"))
end

# ---- run ----
main();
