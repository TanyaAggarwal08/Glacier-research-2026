### pf_predict_every_step_correct_every10.jl
using Pkg
# (uncomment the Pkg.add lines if running on a fresh environment)
# Pkg.add("LowLevelParticleFilters")
# Pkg.add("LinearAlgebra")
# Pkg.add("Random")
# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("Plots")
# Pkg.build("FFMPEG")

using LowLevelParticleFilters
using LinearAlgebra
using Random
using Statistics
using Distributions
using Plots

function main()
    # -------------------------------
    # USER PARAMETERS (keep as you provided)
    # -------------------------------
    Nx, Ny = 40, 40
    N = Nx * Ny
    T = 100
    Np = 10000                       # WARNING: large; reduce for debugging if needed
    x = LinRange(0, 1.0, Nx)
    y = LinRange(0, 1.0, Ny)
    w = 2π / 1.0                      # wavenumber

    obs_interval = 10                 # assimilate every 10 timesteps

    # -------------------------------
    # Initial field
    # -------------------------------
    function initial_field()
        return [sin(w * xi) * sin(w * yi) for yi in y, xi in x] |> vec
    end

    # -------------------------------
    # Dynamics (keep your signature)
    # -------------------------------
    function dynamics(x_state, u, p, t)
        β = reshape(x_state, Ny, Nx)  # (rows = y, cols = x)
        u_x = 1.0
        dx = 1 / Nx
        dt = 0.2 * dx / u_x           # CFL condition
        β_left = circshift(β, (0, -1)) # periodic shift along x
        dbdx = (β - β_left) / dx
        β_new = β .- u_x * dt .* dbdx
        return vec(β_new)
    end

    # -------------------------------
    # Measurement operator (coarse sensors)
    # -------------------------------
    # sensor_indices chosen as before (coarse sampling)
    sensor_indices = collect(1:16:N)
    nobs = length(sensor_indices)

    function measurement(x_state, u, p, t)
        return x_state[sensor_indices]
    end

    # -------------------------------
    # Noise & initial distribution
    # -------------------------------
    process_noise = 0.01                         # std dev per state component (we'll sample)
    measurement_noise_std = 0.1                  # obs noise std dev
    # package constructor below expects process/measurement covariances if giving MvNormal; 
    # we'll create the PF using the convenience constructor that accepts distributions:
    process_noise_dist = MvNormal(zeros(N), 0.01^2 * I(N))
    meas_noise_dist = MvNormal(zeros(nobs), measurement_noise_std^2 * I(nobs))
    initial_state_dist = MvNormal(initial_field(), 0.8^2 * I(N))

    # -------------------------------
    # Initialize ParticleFilter
    # -------------------------------
    pf = ParticleFilter(Np, dynamics, measurement, process_noise_dist, meas_noise_dist, initial_state_dist)

    # -------------------------------
    # Simulate True Dynamics and generate observations only at obs times (every obs_interval)
    # -------------------------------
    x_true = initial_field()
    true_states = Vector{Vector{Float64}}(undef, T)
    # store observations in dictionary keyed by time t (only for times multiple of obs_interval)
    observations = Dict{Int, Vector{Float64}}()

    rng = MersenneTwister(1234)
    for t in 1:T
        # propagate true state (use same dynamics signature)
        x_true = dynamics(x_true, nothing, nothing, t) .+ rand(rng, process_noise_dist)
        true_states[t] = copy(x_true)
        if t % obs_interval == 0
            # measurement at this time
            y_obs = measurement(x_true, nothing, nothing, t) .+ rand(rng, meas_noise_dist)
            observations[t] = y_obs
        end
    end

    # -------------------------------
    # Precompute sensor grid coords (for plotting)
    # Linear index -> row,col mapping: row = ((i-1) % Ny)+1, col = ((i-1) ÷ Ny)+1
    # -------------------------------
    sensor_coords = [(((i - 1) % Ny) + 1, ((i - 1) ÷ Ny) + 1) for i in sensor_indices]
    # plt_beta = heatmap(
    #     x, y, β_field,
    #     xlabel = "x (km)",
    #     ylabel = "y (km)",
    #     colorbar_title = "β(x,y)",
    #     title = "True β Field with Observations at t = $t_show",
    #     aspect_ratio = 1
    # )

    # # Overlay observation points
    # scatter!(
    #     x_coords, y_coords,
    #     markershape = :circle,
    #     color = :red,
    #     label = "Observations",
    #     markersize = 5
    # )
    # savefig(plt_beta, "beta_field_with_observations.png")

    # -------------------------------
    # Run Particle Filter only at selected steps
    # -------------------------------
    # time_array = collect(1:5:T)
    # time_length = length(time_array)
    estimates = Vector{Vector{Float64}}(undef, T)

    for t in 1:T
        pf(nothing, observations[t])                 # Update PF
        estimates[t] = weighted_mean(pf)             # Store estimate only at selected step
    end


    # -------------------------------
    # heatmaps
    # -------------------------------
    # Convert to physical x,y positions
    x_coords = [x[p[1]] for p in sensor_coords]
    y_coords = [y[p[2]] for p in sensor_coords]
    # changing this part of the code to get heatmap for all timestep 
    β_storage = Vector{Matrix{Float64}}()
    estimate_β_storage = Vector{Matrix{Float64}}()
    err_storage = Vector{Matrix{Float64}}()
    for t in 1:T
        β_t = reshape(true_states[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(β_storage, β_t)
    end
    for t in 1:nobs
        estimate_β_t = reshape(estimates[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(estimate_β_storage, estimate_β_t)
    end

    for t in 1:T
        err_t = reshape(estimates[t] - true_states[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(err_storage, err_t)
    end

    tem = 1
    anim = @animate for β_t in β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "True β Field with Observations at t = $tem",
            aspect_ratio = 1,
            clim=(-1, 1))
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem += 1
    end
    gif(anim, "beta_wave_motion1.gif", fps = 4)

    tem1 = 1
    anim = @animate for β_t in estimate_β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "Estimated β Field with Observations at t = $tem1",
            aspect_ratio = 1,
            clim=(-1, 1))
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem1 += 1
    end
    gif(anim, "beta_estimate_wave_motion1.gif", fps = 4)

    tem2 = 1
    anim = @animate for err in err_storage
        heatmap(x, y, err,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "Error",
            title = "Error Field at t = $tem2",
            aspect_ratio = 1,
            clim=(-1, 1))
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem2 += 1
    end
    gif(anim, "error_wave_motion1.gif", fps = 4)


    # -------------------------------
    # Compute RMSE (Global and Single Target)
    # -------------------------------
    target_index = sensor_indices[1]  # linear index (1-based) of point to track
    rmse_history_global = zeros(T)
    rmse_history_target = zeros(T)

    for t in 1:T
        err_global = estimates[t] - true_states[t]
        rmse_history_global[t] = sqrt(mean(err_global.^2))

        err_target = estimates[t][target_index] - true_states[t][target_index]
        rmse_history_target[t] = abs(err_target)  # single point RMSE = abs error
    end

    # Expand RMSE to full time array for plotting
    rmse_full_global = fill(NaN, T)
    rmse_full_target = fill(NaN, T)
    for t in 1:T
        rmse_full_global[t] = rmse_history_global[t]
        rmse_full_target[t] = rmse_history_target[t]
    end

    # Plot Global RMSE
    plt_rmse_global = plot(
        1:T, rmse_full_global;
        xlabel = "Time step", ylabel = "RMSE",
        title = "Global RMSE Evolution (Sparse Estimates)",
        markershape = :circle, markersize = 2
    )
    savefig(plt_rmse_global, "rmse_particle_filter_global1.png")

    # Plot Single-Point RMSE
    plt_rmse_target = plot(
        1:T, rmse_full_target;
        xlabel = "Time step", ylabel = "RMSE at point $(target_index)",
        title = "Single-Point RMSE Evolution",
        markershape = :circle, markersize = 2, color = :red
    )
    savefig(plt_rmse_target, "rmse_particle_filter_target1.png")

    # -------------------------------
    # Plot β evolution at the target point
    # -------------------------------
    true_vals = [true_states[t][target_index] for t in 1:T]
    est_vals = [estimates[t][target_index] for t in 1:T]
    obs_vals = [observations[t][findfirst(i -> i == target_index, sensor_indices)] for t in 1:T]

    plt_pointwise = plot(1:T, true_vals, label = "True β", linewidth=2, alpha=0.9)
    plot!(1:T, est_vals, label = "Estimated β", linewidth=2, linestyle = :dash)
    plot!(1:T, obs_vals, label = "Observed β", linewidth=2, seriestype=:scatter, alpha = 0.5, color = :grey)

    xlabel!("Time step")
    ylabel!("β value at point $(target_index)")
    title!("β Evolution at Target Point")
    savefig(plt_pointwise, "beta_comparison_single_point1.png")

    println("✅ Particle Filter completed.")
    println("📊 Plots saved: beta_field_with_observations.png, rmse_particle_filter_global.png, rmse_particle_filter_target.png, beta_comparison_single_point.png")
end

@time main()
