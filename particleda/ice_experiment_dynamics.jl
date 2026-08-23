# Standalone β-dynamics for the WAVI observation-operator experiment (Exp 18).
#
# WAVI.jl loads only in the default Julia environment; glacier_model.jl needs
# ParticleDA, which is only in the dev/test environment. Since Exp 18 must call
# WAVI *inside* the PF loop, both have to live in one process — so this module
# reproduces exactly the β-dynamics of glacier_model.jl (pseudo-random-wave
# prior, squared-exponential smooth noise, Lax–Wendroff advection, process
# noise, RPF jitter) with NO ParticleDA dependency, so it can run alongside
# WAVI under the default env.
#
# The function bodies are copied verbatim from glacier_model.jl; only the
# ParticleDA method dispatch and the model struct are stripped. A validation
# check in experiment_18 confirms this port reproduces glacier_model.jl's
# truth prior, sensor indices, LW step, and noise factor bit-for-bit.

module IceExpDyn

using Random, LinearAlgebra, Statistics, DelimitedFiles

Base.@kwdef struct Params
    nx::Int = 40
    ny::Int = 40
    x_length::Float64 = 160_000.0
    y_length::Float64 = 160_000.0
    station_filename::String = ""
    init_std_beta::Float64 = 200.0
    process_std_beta::Float64 = 10.0
    min_beta::Float64 = 10.0
    advection_epsilon::Float64 = 0.0
    n_integration_step::Int = 10
    time_step::Float64 = 3600.0
    noise_length_scale::Float64 = 15_000.0
    advection_type::String = "lax_wendroff"
    prior_center_beta::Float64 = 2000.0
    prior_signal_scale_beta::Float64 = 300.0
    background_std_beta::Float64 = 300.0
    prior_max_wavenumber::Int = 2
    prior_truth_seed::Int = 11
    prior_background_seed::Int = 29
    # --- double-bump prior (legacy sinusoid), OFF by default -----------------
    # Kept alongside pseudo_random_wave so the WAVI β→velocity response can be
    # tested against a clean, large-scale bump pattern WITHOUT disturbing the
    # pseudo_random_wave default. Set prior_mode="double_bump" to enable.
    prior_mode::String = "pseudo_random_wave"   # or "double_bump"
    prior_amplitude_beta::Float64 = 2000.0      # double-bump amplitude
    prior_n_modes::Int = 3                       # double-bump periods per axis
end

struct Model
    p::Params
    beta_prior_mean::Vector{Float64}    # background field (particle init mean)
    truth_prior_mean::Vector{Float64}   # distinct truth field
    sensor_indices::Vector{Int}
    noise_factor::Matrix{Float64}       # Cholesky L of squared-exp covariance
    noise_buffer::Matrix{Float64}
end

# ── prior field (verbatim from glacier_model.jl) ─────────────────────────
function _normalised_pseudorandom_wave(p::Params, rng::AbstractRNG)
    nx, ny = p.nx, p.ny
    wave = zeros(Float64, ny, nx)
    for kx in 0:p.prior_max_wavenumber
        for ky in 0:p.prior_max_wavenumber
            kkx = 2π * kx / nx
            kky = 2π * ky / ny
            a   = randn(rng)
            phx = randn(rng) * 2π
            phy = randn(rng) * 2π
            @inbounds for j in 1:ny
                sy = sin(kky * j + phy)
                for i in 1:nx
                    wave[j, i] += a * sin(kkx * i + phx) * sy
                end
            end
        end
    end
    σ = std(vec(wave)); σ == 0 && (σ = 1.0)
    wave ./= σ
    return vec(wave)
end

function _pseudorandom_truth_background(p::Params)
    truth_wave      = _normalised_pseudorandom_wave(p, MersenneTwister(p.prior_truth_seed))
    background_wave = _normalised_pseudorandom_wave(p, MersenneTwister(p.prior_background_seed))
    truth_prior      = p.prior_center_beta .+ p.prior_signal_scale_beta .* truth_wave
    background_prior = truth_prior .+ p.background_std_beta .* background_wave
    return truth_prior, background_prior
end

# Legacy double-bump field: center + amplitude·sin(ωx)·sin(ωy), ω = n_modes·2π/L.
# Deterministic (no RNG); prior_n_modes controls how many bump pairs tile the
# domain (=1 → one clean 2×2 checkerboard of opposite bumps).
function _double_bump_field(p::Params)
    ω  = p.prior_n_modes * 2π / p.x_length
    xs = range(0.0, p.x_length; length=p.nx)
    ys = range(0.0, p.y_length; length=p.ny)
    field = [p.prior_center_beta + p.prior_amplitude_beta * sin(ω * xi) * sin(ω * yj)
             for yj in ys, xi in xs]        # (ny, nx), matches reshape convention
    return vec(field)
end

# Truth = double bump. The background (initial guess) keeps the SAME seed2-style
# offset as pseudo_random_wave (truth + background_std·random wave), so there is
# a genuine initial guess to compare against — unlike the legacy double_bump twin
# where truth == prior (glacier_model.jl returned β_prior, β_prior).
function _double_bump_truth_background(p::Params)
    truth = _double_bump_field(p)
    background = truth .+ p.background_std_beta .*
                 _normalised_pseudorandom_wave(p, MersenneTwister(p.prior_background_seed))
    return truth, background
end

_build_prior_fields(p::Params) =
    p.prior_mode == "double_bump" ? _double_bump_truth_background(p) :
                                    _pseudorandom_truth_background(p)

function _build_sensor_indices(p::Params)
    path = isabspath(p.station_filename) ? p.station_filename : realpath(p.station_filename)
    coords = readdlm(path, ',', Float64, '\n'; comments=true, comment_char='#')
    dx = p.x_length / p.nx; dy = p.y_length / p.ny
    idxs = Int[]
    for row in 1:size(coords, 1)
        x_m, y_m = coords[row, 1], coords[row, 2]
        i = clamp(round(Int, x_m / dx) + 1, 1, p.nx)
        j = clamp(round(Int, y_m / dy) + 1, 1, p.ny)
        push!(idxs, (i - 1) * p.ny + j)
    end
    seen = Set{Int}(); uniq = Int[]
    for idx in idxs
        if idx ∉ seen; push!(seen, idx); push!(uniq, idx); end
    end
    return uniq
end

function _build_noise_factor(p::Params)
    n = p.nx * p.ny
    dx = p.x_length / p.nx; dy = p.y_length / p.ny
    ℓ2 = p.noise_length_scale^2
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i2 in 1:p.nx, j2 in 1:p.ny
        idx2 = (i2 - 1) * p.ny + j2
        x2 = (i2 - 1) * dx; y2 = (j2 - 1) * dy
        for i1 in 1:p.nx, j1 in 1:p.ny
            idx1 = (i1 - 1) * p.ny + j1
            x1 = (i1 - 1) * dx; y1 = (j1 - 1) * dy
            r2 = (x1 - x2)^2 + (y1 - y2)^2
            K[idx1, idx2] = exp(-r2 / (2 * ℓ2))
        end
    end
    @inbounds for i in 1:n; K[i, i] += 1e-8; end
    return Matrix(cholesky(Symmetric(K)).L)
end

function init(p::Params)
    truth_prior, background_prior = _build_prior_fields(p)
    sensors = _build_sensor_indices(p)
    L = _build_noise_factor(p)
    return Model(p, background_prior, truth_prior, sensors, L,
                 zeros(p.nx * p.ny, 1))
end

# ── noise + dynamics (verbatim from glacier_model.jl) ────────────────────
function apply_noise!(state::AbstractVector, m::Model, rng::AbstractRNG, σ::Real)
    z = view(m.noise_buffer, :, 1)
    @inbounds for i in eachindex(z); z[i] = randn(rng); end
    mul!(state, m.noise_factor, z, σ, 1.0)
    return state
end

function sample_initial_state!(state::AbstractVector, m::Model, rng::AbstractRNG)
    state .= m.beta_prior_mean
    apply_noise!(state, m, rng, m.p.init_std_beta)
    @inbounds for i in eachindex(state); state[i] = max(state[i], m.p.min_beta); end
    return state
end

function update_state_deterministic!(state::AbstractVector, m::Model)
    p = m.p; nx, ny = p.nx, p.ny
    dx = p.x_length / nx; ε = p.advection_epsilon
    β = reshape(state, ny, nx); β_new = similar(β)
    dt = p.time_step / p.n_integration_step
    if p.advection_type == "lax_wendroff"
        for _ in 1:p.n_integration_step
            @inbounds for j in 1:ny, i in 1:nx
                im = mod1(i - 1, nx); ip = mod1(i + 1, nx)
                v_i = 1.0 + ε * β[j, i]; α = v_i * dt / dx
                β_new[j, i] = β[j, i] -
                              0.5 * α     * (β[j, ip] - β[j, im]) +
                              0.5 * α * α * (β[j, ip] - 2*β[j, i] + β[j, im])
            end
            β, β_new = β_new, β
        end
    else
        is_nonlinear = p.advection_type == "nonlinear"
        for _ in 1:p.n_integration_step
            @inbounds for j in 1:ny, i in 1:nx
                im = mod1(i - 1, nx)
                dβdx = (β[j, i] - β[j, im]) / dx
                velocity = is_nonlinear ? (1 + ε * β[j, i]) : 1.0
                β_new[j, i] = β[j, i] - velocity * dt * dβdx
            end
            β, β_new = β_new, β
        end
    end
    state .= vec(β)
    return state
end

function update_state_stochastic!(state::AbstractVector, m::Model, rng::AbstractRNG)
    apply_noise!(state, m, rng, m.p.process_std_beta)
    @inbounds for i in eachindex(state); state[i] = max(state[i], m.p.min_beta); end
    return state
end

end # module IceExpDyn
