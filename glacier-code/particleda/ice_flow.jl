# Ice-flow forward model wrapper.
#
# Thin wrapper around WAVI.jl that turns a basal-drag β field into an ice
# velocity field. Mirrors the geometry/setup used in glacier-code/main.jl
# but rescaled to the 40×40 grid we use throughout the particle-filter
# runs (main.jl used 80×80 originally).
#
# Usage:
#   include("glacier-code/particleda/ice_flow.jl")
#   using .IceFlow
#   u, v = IceFlow.velocity(beta_matrix)     # beta_matrix is (NX, NY)
#   u_flat, v_flat = IceFlow.velocity_flat(beta_state)   # beta_state is length NX*NY, Glacier convention
#
# Geometry (same as main.jl):
#   domain L = 160 km, both x and y directions
#   surface elevation z_s(x) = 1060 · √(1 − x/L)   (only x-dependent)
#   bed elevation z_b = 0 everywhere
#   initial thickness h_init = max(z_s − z_b, 0)
#
# Only β changes between calls; everything else (grid, bed, thickness,
# initial conditions) is built once at module load.

module IceFlow

using WAVI

# ── Grid parameters (match the PF setup: 40×40 over 160 km) ─────────────
const NX = 40
const NY = 40
const L  = 160_000.0

# Surface elevation profile (only depends on x). Clamped to keep the
# square root well-defined outside [0, L].
z_s(x) = 1060.0 * sqrt(clamp(1.0 - x / L, 0.0, 1.0))

# Build grid, topography, and initial conditions once. Reused for every β.
const GRID = Grid(nx = NX, ny = NY,
                  dx = L / NX, dy = L / NY,
                  x0 = 0.0, y0 = 0.0,
                  u_iszero = ["north"], v_iszero = ["south"])
const Z_B    = zeros(NX, NY)
const Z_S    = z_s.(GRID.xxh)
const H_INIT = max.(Z_S .- Z_B, 0.0)
const IC     = InitialConditions(initial_thickness = H_INIT)

"""
    velocity(beta_matrix) -> (u, v)

Run the WAVI flow model on the given β field and return horizontal
velocity components on the h-grid. `beta_matrix` must be `(NX, NY)`
matching the WAVI/`GRID.xxh` convention (x is first index, y is second).

Each call constructs a fresh `Model` because WAVI's `Params` embeds β.
The grid, bed, and initial thickness are reused from the module cache.
"""
function velocity(beta_matrix::AbstractMatrix{<:Real})
    size(beta_matrix) == (NX, NY) || throw(ArgumentError(
        "beta_matrix must be $(NX)×$(NY), got $(size(beta_matrix))"))
    # WAVI names the basal-drag coefficient field `weertman_c` (τ_b = C·u^m).
    # main.jl used the older `beta` keyword; we pass the same field but under
    # the new name so this works with the currently-installed WAVI.jl.
    params = Params(weertman_c = Matrix{Float64}(beta_matrix))
    model  = Model(grid = GRID, bed_elevation = Z_B,
                   initial_conditions = IC, params = params)
    update_state!(model)
    return (u = copy(model.fields.gh.u),
            v = copy(model.fields.gh.v))
end

"""
    velocity_flat(beta_state) -> (u, v)

Convenience wrapper for the PF: accepts β as a flat length-`NX·NY`
vector in the *Glacier* convention (`reshape(state, NY, NX)` → (y, x)),
transposes it to WAVI's (x, y) layout, runs the solver, and returns
`u`, `v` also transposed back into the Glacier layout so downstream
plots/RMSE code can share the same reshape logic.
"""
function velocity_flat(beta_state::AbstractVector{<:Real})
    length(beta_state) == NX * NY || throw(ArgumentError(
        "beta_state must have length $(NX * NY), got $(length(beta_state))"))
    β_yx = reshape(beta_state, NY, NX)          # Glacier convention (y, x)
    β_xy = Matrix(transpose(β_yx))              # WAVI convention  (x, y)
    fields = velocity(β_xy)
    # Flip back so the caller can reshape(u_flat, NY, NX) if desired.
    return (u = vec(transpose(fields.u)),
            v = vec(transpose(fields.v)))
end

end  # module IceFlow
