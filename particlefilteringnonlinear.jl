# Importing necessary packages
using Pkg
# Pkg.add("LowLevelParticleFilters")
# Pkg.add("LinearAlgebra")
# Pkg.add("Random")
# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("Plots")

using LowLevelParticleFilters
using LinearAlgebra
using Random
using Statistics
using Distributions
# Use headless GR backend to avoid GKS port/broken-pipe issues when saving files.
ENV["GKSwstype"] = "100"
using Plots

function main()
    # -------------------------------
    # Grid and Particle Filter Setup
    # -------------------------------
    Nx, Ny = 40, 40
    N = Nx * Ny
    T = 200
    Np = 10000 # number of particles
    x = LinRange(0, 1.0, Nx)
    y = LinRange(0, 1.0, Ny)
    w = 2π / 1.0  # wavenumber

    # -------------------------------
    # Initial field
    # -------------------------------
    function initial_field()
        return [ sin(w * xi) * sin(w * yi) for yi in y, xi in x] |> vec
    end

    # -------------------------------
    # Dynamics: non-linear advection to the right
    # -------------------------------
    function dynamics(x_state, u, p, t)
        L = 1.0
        dx = L / Nx
        ε = 0.001
        β = reshape(x_state, Ny, Nx)
        β_new = similar(β)

        # Periodic boundary
        function wrap(i, n)
            return mod1(i, n)
        end

        max_speed = maximum(1 .+ ε .* β)
        # CFL condition
        dt = 0.2 * dx / max_speed  

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

    # -------------------------------
    # Random sensor locations (linear indices)
    # -------------------------------
    sensor_indices = collect(1:16:N)
    function measurement(x_state, u, p, t)
        return x_state[sensor_indices]
    end

    # -------------------------------
    # Noise models
    # -------------------------------
    process_noise = MvNormal(zeros(N), 0.01^2 * I(N))
    measurement_noise = MvNormal(zeros(length(sensor_indices)), 0.1^2 * I(length(sensor_indices)))
    initial_state_dist = MvNormal(initial_field(), 0.8^2 * I(N))

    # -------------------------------
    # Initialize Particle Filter
    # -------------------------------
    pf = ParticleFilter(Np, dynamics, measurement, process_noise, measurement_noise, initial_state_dist)
    @info "Initialized PF with Np = $Np"


    # -------------------------------
    # Particle plots need an actual filter update first, so we do them later
    # using the particle ensemble from the running filter.

    # -------------------------------
    # Simulate True Dynamics
    # -------------------------------
    x_true = initial_field()
    true_states = Vector{Vector{Float64}}(undef, T)
    observations = Vector{Vector{Float64}}(undef, T)

    for t in 1:T
        x_true = dynamics(x_true, nothing, nothing, t) + rand(process_noise)
        y_obs = measurement(x_true, nothing, nothing, t) + rand(measurement_noise)
        true_states[t] = x_true
        observations[t] = y_obs
    end

    # -------------------------------
    # Compute sensor coordinates (grid indices)
    # -------------------------------
    sensor_coords = [[((i - 1) % Nx) + 1, ((i - 1) ÷ Nx) + 1] for i in sensor_indices]

    # -------------------------------
    # Run Particle Filter only at selected steps
    # -------------------------------
    estimates = Vector{Vector{Float64}}(undef, T)
    # Effective Sample Size (ESS) history
    ess_history = zeros(Float64, T)
    # Particle weight health metrics
    max_weight_history = zeros(Float64, T)
    min_weight_history = zeros(Float64, T)
    var_weight_history = zeros(Float64, T)
    entropy_history = zeros(Float64, T)

    for t in 1:T
        pf(nothing, observations[t])                 
        estimates[t] = weighted_mean(pf)             
        # --- ESS computation ---
        # Use linear weights from the filter (weights(pf) may be log-weights)
        w = expweights(pf)
        w_normalized = w ./ sum(w)
        ess_history[t] = 1.0 / sum(w_normalized .^ 2)
        max_weight_history[t] = maximum(w_normalized)
        min_weight_history[t] = minimum(w_normalized)
        var_weight_history[t] = var(w_normalized)
        entropy_history[t] = -sum(w_normalized .* log.(w_normalized .+ eps()))
        if t <= 3 || t == T
            @info "t=$t: ESS=$(ess_history[t]) max_w=$(max_weight_history[t]) min_w=$(min_weight_history[t])"
        end
    end


    # -------------------------------
    # Convert to physical x,y positions for observation points
    # -------------------------------
    x_coords = [x[p[1]] for p in sensor_coords]
    y_coords = [y[p[2]] for p in sensor_coords]
    β_storage = Vector{Matrix{Float64}}()
    estimate_β_storage = Vector{Matrix{Float64}}()
    difference_β= Vector{Matrix{Float64}}()
    for t in 1:T
        β_t = reshape(true_states[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(β_storage, β_t)
    end
    for t in 1:T
        estimate_β_t = reshape(estimates[t], Ny, Nx)  # Ny x Nx for heatmap
        push!(estimate_β_storage, estimate_β_t)
    end



    # -------------------------------
    # Create animations for true and estimated β fields
    # -------------------------------
    tem = 1
    anim = @animate for β_t in β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "True β Field with Observations at t = $tem",
            aspect_ratio = 1,
            clim = (-1,1),
            c = :Blues)
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 0.8)
        tem += 1
    end
    gif(anim, "beta_wave_motionnon--linear.gif", fps = 6)



    # pick one fixed point (x_mid, y_mid)
    x_idx = round(Int, Nx/4)
    y_idx = round(Int, Ny/4)

    # extract β(t) at that location
    β_point = [β[y_idx, x_idx] for β in β_storage]

    # plot over time
    p = plot(1:T, β_point, xlabel="Time step", ylabel="β at (x_mid, y_mid)",
            title="β evolution at a single point", lw=2, marker=:circle)

    savefig(p, "beta_point_evolution.png")

    # Visualize the full particle ensemble at this same grid point.
    # This is the closest analogue to the particle scatter in test1.jl,
    # but here the state is a 2D field, so we plot the marginal over one cell.
    point_index = (y_idx - 1) * Nx + x_idx
    particle_state = particles(pf)
    particle_values = [ps[point_index] for ps in particle_state]

    plt_particles = histogram(
        particle_values,
        bins = 60,
        xlabel = "β value",
        ylabel = "Particle count",
        title = "Particle ensemble at (x/4, y/4)",
        label = "Particles",
    )
    savefig(plt_particles, "particle_ensemble_point.png")

    tem1 = 1
    anim = @animate for β_t in estimate_β_storage
        heatmap(x, y, β_t,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            colorbar_title = "β(x, y, t)",
            title = "Estimated β Field with Observations at t = $tem1",
            aspect_ratio = 1,
            clim=(-1,1),
            c = :Blues)
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem1 += 1
    end
    gif(anim, "beta_estimate_wave_motionnon--linear.gif", fps = 4)


    # -------------------------------
    # Compute RMSE (Global and Single Target)
    # -------------------------------
    # getting first sensor as target point 
    target_index = sensor_indices[1]  
    rmse_history_global = zeros(T)
    rmse_history_target = zeros(T)
    



    for t in 1:T
        difference = estimates[t] - true_states[t]
        β_t_diff = reshape(difference, Ny, Nx)  # Ny x Nx for heatmap
        push!(difference_β, β_t_diff)
    end
    tem2 = 1
    anim = @animate for β_t_diff in difference_β
        heatmap(x, y, β_t_diff,  # note the transpose
            xlabel = "x (km)",
            ylabel = "y (km)",
            title = "True β - Estimate β at t = $tem2",
            aspect_ratio = 1,
            clim=(-1,1),
            c = :Blues)
        scatter!(x_coords, y_coords,
            markershape = :circle,
            color = :red,
            label = "Observations",
            markersize = 1)
        tem2 += 1
    end
    gif(anim, "beta_difference_wave_motionnon--linear.gif", fps = 4)



    # averaged rmse over every time step
    for t in 1:T
        err_global = estimates[t] - true_states[t]
        rmse_history_global[t] = sqrt(mean(err_global.^2))
        err_target = estimates[t][target_index] - true_states[t][target_index]
        rmse_history_target[t] = abs(err_target) 
    end

    # Expand RMSE to full time array for plotting
    rmse_full_global = fill(NaN, T)
    rmse_full_target = fill(NaN, T)
    for t in 1:T
        rmse_full_global[t] = rmse_history_global[t]
        rmse_full_target[t] = rmse_history_target[t]
    end

  
    # -------------------------------  
    # Plot Global RMSE
    # -------------------------------
    plt_rmse_global = plot(
        1:T, rmse_full_global;
        xlabel = "Time step", ylabel = "RMSE",
        title = "Global RMSE Evolution (Sparse Estimates)",
        markershape = :circle, markersize = 2
    )
    savefig(plt_rmse_global, "rmse_particle_filter_globalnon--linear.png")


    # -------------------------------
    # Plot Single-Point RMSE
    # -------------------------------
    plt_rmse_target = plot(
        1:T, rmse_full_target;
        xlabel = "Time step", ylabel = "RMSE at point $(target_index)",
        title = "Single-Point RMSE Evolution",
        markershape = :circle, markersize = 2, color = :red
    )
    savefig(plt_rmse_target, "rmse_particle_filter_targetnon--linear.png")

   # ----------------------------------------
    # Collect values at a single point (x/4, y/4)
    # ----------------------------------------
    x_idx = round(Int, Nx/4)
    y_idx = round(Int, Ny/4)
    point_index = (y_idx - 1) * Nx + x_idx   # convert 2D index to linear index

    # ----------------------------------------
    # Visualize the particle ensemble at one grid point
    # ----------------------------------------
    particle_state = particles(pf)
    particle_values = [ps[point_index] for ps in particle_state]

    plt_particles = histogram(
        particle_values,
        bins = 60,
        xlabel = "β value",
        ylabel = "Particle count",
        title = "Particle ensemble at (x/4, y/4)",
        label = "Particles",
    )
    savefig(plt_particles, "particle_ensemble_point.png")

    estimate_point = zeros(T)    # PF weighted mean
    true_point = zeros(T)        # truth
    obs_point = fill(NaN, T)     # observations (NaN when not observed)

    for t in 1:T
        # Weighted mean estimate at this point
        estimate_point[t] = estimates[t][point_index]

        # True β at this point
        true_point[t] = true_states[t][point_index]

        # Observation (only if sensor exists at this location)
        idx_in_sensors = findfirst(i -> i == point_index, sensor_indices)
        if idx_in_sensors !== nothing
            obs_point[t] = observations[t][idx_in_sensors]
        end
    end

    # ----------------------------------------
    # Plot: estimate, truth, and observations
    # ----------------------------------------
    plt_point = plot(1:T, true_point, color=:red, lw=2, linestyle=:dash, label="True β",
                    xlabel="Time step", ylabel="β value",
                    title="β Evolution at (x/4, y/4)",
                    )
    plot!(1:T, estimate_point, color=:blue, lw=2, label="PF Estimate")
    scatter!(1:T, obs_point, color=:black, marker=:x, ms=4, label="Observations")

    savefig(plt_point, "beta_point_obs_estimate.png")

    # -------------------------------
    # Plot ESS over time
    # -------------------------------
    plt_ess = plot(1:T, ess_history;
        xlabel = "Time step",
        ylabel = "ESS",
        title = "ESS Evolution",
        legend = :topright,
        ylim = (0, Np),
        markershape = :circle,
        markersize = 3)
    hline!(plt_ess, [0.5 * Np], linestyle = :dash, color = :red, label = "0.5*Np threshold")
    savefig(plt_ess, "ess_evolution_non--linear.png")

    # -------------------------------
    # Plot weight diagnostics over time
    # -------------------------------
    plt_weight_extremes = plot(1:T, max_weight_history;
        xlabel = "Time step",
        ylabel = "Weight value",
        title = "Particle Weight Extremes",
        label = "max weight",
        legend = :topright,
        markershape = :circle,
        markersize = 2)
    plot!(plt_weight_extremes, 1:T, min_weight_history; label = "min weight")
    savefig(plt_weight_extremes, "weight_extremes_non--linear.png")

    plt_weight_var = plot(1:T, var_weight_history;
        xlabel = "Time step",
        ylabel = "Var(normalized weights)",
        title = "Weight Variance Evolution",
        label = "weight variance",
        legend = :topright,
        markershape = :circle,
        markersize = 2)
    savefig(plt_weight_var, "weight_variance_non--linear.png")

    plt_entropy = plot(1:T, entropy_history;
        xlabel = "Time step",
        ylabel = "Entropy",
        title = "Weight Entropy Evolution",
        label = "weight entropy",
        legend = :topright,
        markershape = :circle,
        markersize = 2)
    hline!(plt_entropy, [log(Np)], linestyle = :dash, color = :red, label = "log(Np)")
    savefig(plt_entropy, "weight_entropy_non--linear.png")

end

@time main()
