using Plots

# Safe periodic indexing
wrap(i, n) = mod1(i, n)

function run_simple_nonlinear_advection()
    # Grid setup
    L = 1.0
    nx, ny = 80, 80
    x = range(0, L, length=nx+1)[1:end-1]
    y = range(0, L, length=ny+1)[1:end-1]
    dx = L / nx
    dy = L / ny
    X = [x[i] for j in 1:ny, i in 1:nx]
    Y = [y[j] for j in 1:ny, i in 1:nx]

    # Initial condition
    β =  1000 .+ 1000 .* sin.(2π .* X ./ L) .* sin.(2π .* Y ./ L)

    # Parameters
    ε = 0.001
    max_speed = maximum(1 .+ ε .* β)
    dt = 0.2 * dx / max_speed  # CFL condition
    t_final = 1.0
    nt = Int(round(t_final / dt))
    println("Running with dt = $dt, steps = $nt")

    # Save every few frames to keep GIF small
    frame_interval = max(1, nt ÷ 50)

    # Pick one point to track (midpoint of grid)
    x_idx = round(Int, nx/4)
    y_idx = round(Int, ny/4)

    # Store beta at that point over time
    β_at_point = zeros(nt)
    anim = @animate for n in 1:nt
        β_new = similar(β)
        for j in 1:ny
            for i in 1:nx
                im = wrap(i - 1, nx)
                dβdx = (β[j, i] - β[j, im]) / dx
                velocity = 1 + ε * β[j, i]
                β_new[j, i] = β[j, i] - velocity * dt * dβdx
            end
        end
        β = β_new
        β_at_point[n] = β[y_idx, x_idx]
        
        heatmap(
                x, y, β,
                clims=(0, 2000),
                xlabel="x", ylabel="y",
                title="Quasilinear Advection Step $n",
                aspect_ratio=1
        )
    end

    gif(anim, "quasilinear_advection.gif", fps=15)
      # Plot time series
    plot(1:nt, β_at_point,
         xlabel="Timestep",
         ylabel="β value",
         title="β evolution",
         lw=2)
    savefig("beta_timeseries.png")
    
end

run_simple_nonlinear_advection()
