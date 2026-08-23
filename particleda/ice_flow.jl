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
# ── Boundary conditions ─────────────────────────────────────────────────
# CAUTION: WAVI's orientation names map to ARRAY indices, not compass sense
# (Grid.jl `orientations2bc`): arrays are (x,y), so
#   "north" → A[1,:]   = x = 0        "south" → A[end,:] = x = L
#   "west"  → A[:,1]   = y = 0        "east"  → A[:,end] = y = L
# i.e. north/south are the x-edges and east/west are the y-edges.
#
# The original setup (u=["north"], v=["south"]) constrained only 81 velocity
# points, both on x-edges, leaving y=0 and y=L completely unconstrained. Those
# free edges act as calving fronts: nothing resists them, so they ran ~680 m/yr
# against an interior of ~18 (38× ratio), hijacking every colour scale and
# polluting full-field β/velocity correlations (−0.54 vs −0.92 interior-only).
#
# Adding "east","west" to v_iszero makes the lateral boundaries FREE-SLIP walls:
# no flow *through* them (v=0), but ice still slides *along* them in x. Measured
# effect: edge speed 679 → 18 m/yr, interior essentially unchanged (17.8 → 17.3).
# Zeroing u there as well (no-slip) instead creates dead zones — not used.
#
# Runs experiment_18_wavi_obs{,_seed2,_weertman_m}, experiment_18_double_bump
# {,_weertman_m,_center1000} all predate this. Reproduce them with:
#   WAVI_V_ISZERO=south julia ...
const U_ISZERO = String.(split(get(ENV, "WAVI_U_ISZERO", "north"), ","))
const V_ISZERO = String.(split(get(ENV, "WAVI_V_ISZERO", "south,east,west"), ","))

const GRID = Grid(nx = NX, ny = NY,
                  dx = L / NX, dy = L / NY,
                  x0 = 0.0, y0 = 0.0,
                  u_iszero = U_ISZERO, v_iszero = V_ISZERO)
# ── Sliding-law exponent ────────────────────────────────────────────────
# WAVI's Weertman law is  τ_b = C·|u|^(1/m−1)·u , and internally it forms
#   gh.β = weertman_c · |u_bed|^(1/m − 1)
# which is what the linear solver actually uses. With m = 1 the exponent is
# zero, that velocity-dependent factor collapses to 1, and gh.β becomes
# IDENTICALLY the β field we pass in (verified: max|gh.β − weertman_c| = 0).
# The sliding law is then linear/viscous, τ_b = β·u — a standard choice for
# basal-drag inversion, and it makes the state we assimilate exactly the drag
# coefficient the momentum solve uses, rather than a velocity-dependent
# transform of it.
#
# m = 3 (WAVI's default) is what experiments 18 seed1/seed2 and the double-bump
# runs used. Override to reproduce those:  WAVI_WEERTMAN_M=3.0 julia ...
const WEERTMAN_M = parse(Float64, get(ENV, "WAVI_WEERTMAN_M", "1.0"))

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
    params = Params(weertman_c = Matrix{Float64}(beta_matrix),
                    weertman_m = WEERTMAN_M)
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
