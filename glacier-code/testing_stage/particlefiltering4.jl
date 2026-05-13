# lets try two things 
# 1. run particle filter every 10 seconds when obs is there
# 2. run particle filter every second but obs only for every 10 seconds

# using Pkg
# Pkg.add("LowLevelParticleFilters")
# Pkg.add("LinearAlgebra")
# Pkg.add("Random")
# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("Plots")
using Pkg
# Pkg.build("FFMPEG")
using LowLevelParticleFilters
using LinearAlgebra
using Random
using Statistics
using Distributions
using Plots

function main()
    # -------------------------------
    # Grid and Particle Filter Setup
    # -------------------------------
    Nx, Ny = 40, 40
    N = Nx * Ny
    T = 100
    Np = 1000 # number of particles
    x = LinRange(0, 1.0, Nx)
    y = LinRange(0, 1.0, Ny)
    w = 2π / 1.0  # wavenumber
    time_array = collect(1:10:T)
    time_length = length(time_array)
    # Initial field
    function initial_field()
        # Iterate over y first for correct heatmap orientation
        return [sin(w * xi) * sin(w * yi) for yi in y, xi in x] |> vec
    end

    # Dynamics: linear advection to the right
    function dynamics(x_state, u, p, t)
        L = 1.0
        dx = L / Nx
        ε = 0.0005

        β = reshape(x_state, Ny, Nx)
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

        return vec(β_new)
    end

    # Random sensor locations (linear indices)
    sensor_indices = collect(1:16:N)
    # nobs = length(sensor_indices)
    function measurement(x_state, u, p, t)
        return x_state[sensor_indices]
    end

    # Noise models
    process_noise = MvNormal(zeros(N), 0.01^2 * I(N))
    measurement_noise = MvNormal(zeros(length(sensor_indices)), 0.1^2 * I(length(sensor_indices)))
    initial_state_dist = MvNormal(initial_field(), 0.9^2 * I(N))

    # Initialize Particle Filter
    pf = ParticleFilter(Np, dynamics, measurement, process_noise, measurement_noise, initial_state_dist)

    # -------------------------------
    # Simulate True Dynamics
    # -------------------------------
    x_true = initial_field()
    true_states = Vector{Vector{Float64}}(undef, T)
    observations = Vector{Vector{Float64}}(undef, 10 )

    for t in 1:T
        x_true = dynamics(x_true, nothing, nothing, t) + rand(process_noise)
        # y_obs = measurement(x_true, nothing, nothing, t) + rand(measurement_noise)
        true_states[t] = x_true
        # observations[t] = y_obs
    end
    

    # -------------------------------
    # get observations after every 10 seconds
    # -------------------------------
    for (i,t) in enumerate(1:10:T)
        y_obs = measurement(x_true, nothing, nothing, t) + rand(measurement_noise)
        observations[i] = y_obs
    end
    

    # -------------------------------
    # Plot 2D heatmap of true β with observation points
    # -------------------------------

    # Compute sensor coordinates (grid indices)
    sensor_coords = [[((i - 1) % Nx) + 1, ((i - 1) ÷ Nx) + 1] for i in sensor_indices]

    

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
 
    estimates = Vector{Vector{Float64}}(undef, time_length)

    # for (i, t) in enumerate(time_array)
    #     pf(nothing, observations[t])                 # Update PF
    #     estimates[i] = weighted_mean(pf)             # Store estimate only at selected step
    # end

    # test
    for i in 1:time_length
        # Observation available
        pf(nothing, observations[i])
        estimates[i] = weighted_mean(pf)
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
    for t in 1:T
        β_t = reshape(true_states[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(β_storage, β_t)
    end
    for i in 1:time_length
        estimate_β_t = reshape(estimates[i], Ny, Nx)  # Ny x Nx for heatmap
        push!(estimate_β_storage, estimate_β_t)
    end

    tem = 1
    anim = @animate for β_t in β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "True β Field with Observations at t = $tem",
            aspect_ratio = 1, 
            clim = (-1, 1))
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem += 1
    end
    gif(anim, "beta_wave_motion.gif", fps = 5)

    tem1 = 1
    anim = @animate for β_t in estimate_β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "Estimated β Field with Observations at t = $tem1",
            aspect_ratio = 1, 
            clim = (-1,1))
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 5)
        tem1 += 1
    end
    gif(anim, "beta_estimate_wave_motion.gif", fps = 5)


    # -------------------------------
    # Compute RMSE (Global and Single Target)
    # -------------------------------
    target_index = sensor_indices[1]  # linear index (1-based) of point to track
    rmse_history_global = zeros(time_length)
    rmse_history_target = zeros(time_length)

    for (i,t) in enumerate(1:10:T)
        err_global = estimates[i] - true_states[t]
        rmse_history_global[i] = sqrt(mean(err_global.^2))

        err_target = estimates[i][target_index] - true_states[t][target_index]
        rmse_history_target[i] = abs(err_target)  # single point RMSE = abs error
    end

    # Expand RMSE to full time array for plotting
    # rmse_full_global = fill(NaN, T)
    # rmse_full_target = fill(NaN, T)
    # for (i, t) in enumerate(1:T)
    #     rmse_full_global[t] = rmse_history_global[t]
    #     rmse_full_target[t] = rmse_history_target[t]
    # end

    # Plot Global RMSE
    plt_rmse_global = plot(
        1:time_length, rmse_history_global;
        xlabel = "Time step", ylabel = "RMSE",
        title = "Global RMSE Evolution (Sparse Estimates)",
        markershape = :circle, markersize = 2
    )
    savefig(plt_rmse_global, "rmse_particle_filter_global--discreteee.png")

    # Plot Single-Point RMSE
    plt_rmse_target = plot(
        1:time_length, rmse_history_target;
        xlabel = "Time step", ylabel = "RMSE at point $(target_index)",
        title = "Single-Point RMSE Evolution",
        markershape = :circle, markersize = 2, color = :red
    )
    savefig(plt_rmse_target, "rmse_particle_filter_target--discreteee.png")

    # -------------------------------
    # Plot β evolution at the target point
    # -------------------------------
    true_vals = [true_states[t][target_index] for t in 1:T]
    est_vals = [estimates[t][target_index] for t in 1:time_length]
    obs_vals = [observations[t][findfirst(i -> i == target_index, sensor_indices)] for t in 1:time_length]

    plt_pointwise = plot(1:T, true_vals, label = "True β", linewidth=2, alpha=0.9)
    plot!(time_array, est_vals, label = "Estimated β", linewidth=2, linestyle = :dash)
    plot!(1:T, obs_vals, label = "Observed β", linewidth=2, seriestype=:scatter, alpha = 0.5, color = :grey)

    xlabel!("Time step")
    ylabel!("β value at point $(target_index)")
    title!("β Evolution at Target Point")
    savefig(plt_pointwise, "beta_comparison_single_point--discreteee.png")

    println("✅ Particle Filter completed.")
    println("📊 Plots saved: beta_field_with_observations.png, rmse_particle_filter_global.png, rmse_particle_filter_target.png, beta_comparison_single_point.png")
end

@time main()
