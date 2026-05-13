using Pkg 
# Pkg.add(url="https://github.com/WAVI-ice-sheet-model/WAVI.jl")
using WAVI, Plots

function model_code()

    # -----------------------------------------------------------------
    # define topography and grid 
    # -----------------------------------------------------------------
    z_s(x) = 1060*sqrt(1-(x/160000.0))


    # z_s(x, y) = 1060 * sqrt(clamp(1 - ((x^2 + y^2)/160000.0^2), 0.0, 1.0))
    # z_s(x, y) = 1060*sqrt(x/160000.0)
    # z_s(x,y) = 1060*sqrt(clamp(1 - ((2x-160000.0)/160000.0)^2,0.0,1.0))
    # z_s(x,y) = 1060*sqrt(clamp(1-(((2x-160000.0)/160000.0)^2+((2y)/160000.0)^2),0.0,1.0))
    # z_s(x,y) = 100*(x+200)^(1/4)+x/60-(2*10^10)^(1/4)+1
    # z_b(x,y; α) = -x * tand(α) - z_s(x)
    # z_b(x,y; α) = -x * tand(α) 
  
    grid_L(; L, nx = 80,ny = 80) = Grid(nx = nx, ny = ny, dx = L/nx, dy = L/ny, y0 = 0.0, x0 = 0.0,  u_iszero = ["north"], v_iszero=["south"]);
    L = 160000.  #80 km, so everything is in meters
    grid80 = grid_L(L = L)
    # z_b80 = z_b.(grid80.xxh,grid80.yyh; α = 0.5 ); 
    z_b80 = zeros(grid80.nx, grid80.ny)
    # z_s80 = z_s.(grid80.xxh,grid80.yyh); #calculate surface elevation
    z_s80 = z_s.(grid80.xxh);

    h_init = z_s80 .- z_b80; #initial thickness of the ice
    h_init = max.(h_init, 0.0)


    # Plot
    nx, ny = 80, 80
    x = range(0, L, length=nx)
    plot(x ./ 1000, z_s80; lw=2, color=:blue)

    # Fill from surface down to baseline (0 elevation)
    plot!(x ./ 1000, z_s80[1, :]; seriestype=:shape, color=:blue, alpha=0.3, legend=false)
  

    xlabel!("x (km)")
    ylabel!("Elevation (m)")
    title!("Glacier Surface (2D Cross-Section)")
    savefig("glacier_surface_2D.png")


    # #actual model
    # initial_conditions = InitialConditions(initial_thickness = 1000. .* ones(grid80.nx, grid80.ny))

    # model80 = Model(grid = grid80, 
    #             bed_elevation = z_b80,
    #             initial_conditions = initial_conditions)
    # update_state!(model80)
    # Plots.heatmap(model80.grid.xxh[:,1]/1e3, model80.grid.yyh[1,:]/1e3, model80.fields.gh.u', 
    #                         xlabel = "x (km)", 
    #                         ylabel = "y (km)",
    #                         colorbar_title = "ice velocity in x-direction (m/yr)")
    # plot!(size = (800,600))
    # # savefig(plt,"velocityexpC")
    

    # ------------------------------------------------------------------
    # create sinoidal basal drag coefficient
    # ------------------------------------------------------------------
    ω = 2π / L 
    beta_array = 1000 .+ 1000 .* sin.(ω .* grid80.xxh) .* sin.(ω .* grid80.yyh)
    # beta_array = 100000.0 * ones(size(grid80.xxh))
    # beta_array = 1000 .+ 1000 .* sin.(ω .* grid80.xxh) 

    # ------------------------------------------------------------------
    # plot the basal drag coefficient
    # ------------------------------------------------------------------
    plt_beta = Plots.heatmap(grid80.xxh[:,1]/1e3, grid80.yyh[1,:]/1e3, beta_array,
    xlabel = "x (km)",
    ylabel = "y (km)",
    colorbar_title = "β(x,y)",
    c = :Blues,
    title = "Basal Drag Coefficient β(x, y)")
    plot!(size = (800,800))
    display(plt_beta)
    savefig(plt_beta, "beta_field_expC.png")



    # ------------------------------------------------------------------
    # build and solve model 
    # ------------------------------------------------------------------
    initial_conditions = InitialConditions(initial_thickness = h_init); #initial thickness of 1000m everywhere
    my_params = Params(beta = beta_array)
    model = Model(grid = grid80, 
                bed_elevation = z_b80,
                initial_conditions = initial_conditions,
                params = my_params);  #build model
    update_state!(model); #update the model state to get the velocity associated with the geometry


    # # ------------------------------------------------------------------
    # # plot u_zero hearmap
    # # ------------------------------------------------------------------
    # plt2 = Plots.heatmap(grid80.u', title="u_iszero mask", xlabel="x", ylabel="y")
    # plot!(size = (800,600))
    # savefig(plt2,"u_iszero_mask_expC.png")


    # ------------------------------------------------------------------
    # plot the velocity heatmap
    # ------------------------------------------------------------------
    umin = minimum(model.fields.gh.u)
    umax = maximum(model.fields.gh.u)
    plt = Plots.heatmap(model.grid.xxh[:,1]/1e3, model.grid.yyh[1,:]/1e3, model.fields.gh.u', 
                            xlabel = "x (km)", 
                            ylabel = "y (km)",
                            colorbar_title = "ice velocity in x-direction (m/yr)",
                            c = :Blues,
                            title = "Ice velocity field",
                            clims = (umin, umax))
    plot!(size = (600,600))
    display(plt)
    savefig(plt,"velocityheatmapexpC.png")


    # ------------------------------------------------------------------
    # get velocity and x-coordinates along a vertical line at x = L/4
    # ------------------------------------------------------------------
    center_index = round(Int, grid80.nx/4);
    println("Center index: ", center_index);
    grid_flowline = model.grid.xxh[:,center_index]; #extract coordinates along line
    U_flowline = model.fields.gh.u[:, center_index]; #get velocity along line

    # ------------------------------------------------------------------
    # plot the velocity along the flowline
    # ------------------------------------------------------------------
    U_flowline_clipped = max.(U_flowline, 0.0); #clip negative velocities to zero
    p = plot(grid_flowline./L, U_flowline, 
        xlabel = "x (km)", 
        ylabel = "horizontal velocity (m/yr)",
        label = "L = $(L/1e3) km",
        framestyle = :box)
    plot!(size = (1000,550))
    display(p)
    savefig(p, "infintetifinite3.png")
end




model_code()