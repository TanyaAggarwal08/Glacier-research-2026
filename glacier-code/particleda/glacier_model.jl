module Glacier

using ParticleDA
using HDF5
using Random
using LinearAlgebra
using PDMats
using Distributions
using DelimitedFiles

Base.@kwdef struct GlacierModelParameters{T<:AbstractFloat}
    nx::Int = 40
    ny::Int = 40
    x_length::T = 160_000.0
    y_length::T = 160_000.0
    # Sensor placement: either a stride over flat indices (legacy/default)
    # OR an x,y CSV file. station_filename overrides sensor_stride when non-empty.
    sensor_stride::Int = 16
    station_filename::String = ""
    # β-space parameters (state = β directly, no log transform).
    # Defaults give same statistical behaviour as previous log-space (0.30, 0.007)
    # at β ≈ 1000: a 30 % initial spread and 0.7 % per-step process noise.
    init_std_beta::T = 150.0
    process_std_beta::T = 7.0
    # Lower clamp on β to keep the surrogate ux = 1000/β well-defined.
    # The default lets β range comfortably; only triggers in extreme noise tails.
    min_beta::T = 10.0
    # Observation space. "velocity" observes ux = 1000/β with additive Gaussian
    # noise of std obs_noise_std. "log_velocity" observes log(ux) = log(1000) -
    # log(β). "log_beta" observes log(β) directly. The two log modes differ
    # only by a sign and a constant, so they induce the same Gaussian likelihood
    # on relative β-misfit. A single σ in log space therefore buys the same
    # fractional accuracy at every β. In velocity space the same σ means 10 %
    # relative error at β = 1000 but 30 % at β = 3000.
    obs_space::String = "velocity"
    obs_noise_std::T = 0.10
    # Delta method: σ_log ≈ σ_v / ux = σ_v · β/1000, so σ_v = 0.10 at the
    # prior centre β = 2000 (ux = 0.5) is equivalent to σ_log ≈ 0.20.
    obs_noise_std_log::T = 0.20
    advection_epsilon::T = 5e-4
    n_integration_step::Int = 1
    time_step::T = 1.0
    # Spatial correlation length (m) for smooth noise. Both the initial-state
    # perturbation and the per-step process noise are drawn from a Gaussian
    # field with squared-exponential covariance, marginal std = the *_std_beta
    # parameter above, and correlation length set here. Setting ≤ 0 falls back
    # to iid noise (legacy behaviour).
    noise_length_scale::T = 30_000.0
    # Advection branch — "linear" (constant v=1) or "nonlinear" (v = 1 + ε·β).
    advection_type::String = "linear"
    # Prior family. "double_bump" keeps the legacy sinusoid. "pseudo_random_wave"
    # builds an Evensen-style separable wave field with a distinct truth and
    # background realisation.
    prior_mode::String = "double_bump"
    prior_center_beta::T = 2000.0
    prior_signal_scale_beta::T = 500.0
    background_std_beta::T = 500.0
    prior_max_wavenumber::Int = 2
    prior_truth_seed::Int = 11
    prior_background_seed::Int = 29
    # Ablation flag: when true, get_log_density returns 0 for every particle,
    # so weights stay uniform and the filter ignores observations entirely.
    # Truth generation and observation simulation are unaffected.
    disable_observations::Bool = false
end

struct GlacierModel{T<:AbstractFloat}
    parameters::GlacierModelParameters{T}
    beta_prior_mean::Vector{T}     # mean β field, flat, length nx*ny
    truth_prior_mean::Vector{T}    # truth initial field used by pseudo-random runs
    sensor_indices::Vector{Int}
    obs_cov::ScalMat{T}
    obs_buffer::Matrix{T}
    state_buffer::Matrix{T}
    # Cholesky factor L of the spatial covariance K (squared-exponential).
    # L is (nx*ny) × (nx*ny). Drawing z ~ N(0, I_n) and forming σ * L * z gives
    # a smooth Gaussian random field with marginal std σ and correlation length
    # noise_length_scale. Empty (0×0) when noise_length_scale ≤ 0 (iid mode).
    noise_factor::Matrix{T}
    noise_buffer::Matrix{T}        # scratch z (n × n_tasks)
end

# Build sensor flat indices either from a stations file (x,y in metres)
# or from the legacy sensor_stride.
function _build_sensor_indices(p::GlacierModelParameters)
    if p.station_filename != ""
        path = isabspath(p.station_filename) ? p.station_filename :
               realpath(p.station_filename)
        coords = readdlm(path, ',', Float64, '\n'; comments=true, comment_char='#')
        @assert size(coords, 2) == 2 "stations file must have 2 columns (x,y)"
        dx = p.x_length / p.nx
        dy = p.y_length / p.ny
        idxs = Int[]
        for row in 1:size(coords, 1)
            x_m, y_m = coords[row, 1], coords[row, 2]
            i = clamp(round(Int, x_m / dx) + 1, 1, p.nx)   # x column
            j = clamp(round(Int, y_m / dy) + 1, 1, p.ny)   # y row
            push!(idxs, (i - 1) * p.ny + j)                # column-major flat
        end
        # Deduplicate while preserving order
        seen = Set{Int}()
        unique_idxs = Int[]
        for idx in idxs
            if idx ∉ seen
                push!(seen, idx); push!(unique_idxs, idx)
            end
        end
        return unique_idxs
    else
        return collect(1:p.sensor_stride:(p.nx * p.ny))
    end
end

function _build_noise_factor(p::GlacierModelParameters{T}) where {T}
    n = p.nx * p.ny
    if p.noise_length_scale <= 0
        return zeros(T, 0, 0)
    end
    dx = p.x_length / p.nx
    dy = p.y_length / p.ny
    ℓ2 = p.noise_length_scale^2
    K = Matrix{T}(undef, n, n)
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
    # Jitter for numerical PD.
    @inbounds for i in 1:n
        K[i, i] += 1e-8
    end
    return Matrix(cholesky(Symmetric(K)).L)
end

function _double_bump_prior(p::GlacierModelParameters{T}) where {T}
    n_modes = 3
    ω = n_modes * 2π / p.x_length
    xs = range(0.0, p.x_length; length=p.nx)
    ys = range(0.0, p.y_length; length=p.ny)
    β_prior = [2000.0 + 2000.0 * sin(ω * xi) * sin(ω * yj) for yj in ys, xi in xs]
    return vec(β_prior)
end

function _normalised_pseudorandom_wave(
    p::GlacierModelParameters{T},
    rng::Random.AbstractRNG,
) where {T}
    nx, ny = p.nx, p.ny
    wave = zeros(T, ny, nx)
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
    σ = std(vec(wave))
    σ == 0 && (σ = one(T))
    wave ./= σ
    return vec(wave)
end

function _pseudorandom_truth_background(p::GlacierModelParameters{T}) where {T}
    rng_truth = MersenneTwister(p.prior_truth_seed)
    rng_background = MersenneTwister(p.prior_background_seed)

    truth_wave = _normalised_pseudorandom_wave(p, rng_truth)
    background_wave = _normalised_pseudorandom_wave(p, rng_background)

    truth_prior = p.prior_center_beta .+ p.prior_signal_scale_beta .* truth_wave
    background_prior = truth_prior .+ p.background_std_beta .* background_wave
    return truth_prior, background_prior
end

function _build_prior_fields(p::GlacierModelParameters{T}) where {T}
    if p.prior_mode == "pseudo_random_wave"
        return _pseudorandom_truth_background(p)
    end
    β_prior = _double_bump_prior(p)
    return β_prior, β_prior
end

const _OBS_SPACES = ("velocity", "log_velocity", "log_beta")

# The σ actually in force, given obs_space. Both obs_cov (used by the
# likelihood) and sample_observation_given_state! read it, so the twin
# experiment can never drift out of sync with the filter's assumed noise.
function _obs_sigma(p::GlacierModelParameters)
    return (p.obs_space == "log_velocity" || p.obs_space == "log_beta") ?
           p.obs_noise_std_log : p.obs_noise_std
end

function init(parameters_dict::Dict, n_tasks::Int=1)
    raw = get(parameters_dict, "glacier", Dict())
    user_input = (; (Symbol(k) => v for (k, v) in raw)...)
    p = GlacierModelParameters(; user_input...)
    @assert p.obs_space in _OBS_SPACES "obs_space must be one of $(_OBS_SPACES), got $(p.obs_space)"

    truth_prior_flat, β_prior_flat = _build_prior_fields(p)

    sensors = _build_sensor_indices(p)
    n_obs = length(sensors)
    obs_cov = ScalMat(n_obs, _obs_sigma(p)^2)

    obs_buffer = zeros(Float64, n_obs, n_tasks)
    state_buffer = zeros(Float64, p.nx * p.ny, n_tasks)
    L = _build_noise_factor(p)
    noise_buffer = zeros(Float64, p.nx * p.ny, n_tasks)

    return GlacierModel{Float64}(p, β_prior_flat, truth_prior_flat, sensors,
                                 obs_cov, obs_buffer, state_buffer, L,
                                 noise_buffer)
end

# Add a smooth or iid Gaussian perturbation of marginal std σ to `state`.
function _apply_noise!(state::AbstractVector, model::GlacierModel,
                       rng::Random.AbstractRNG, σ::Real, task_index::Integer)
    if size(model.noise_factor, 1) == 0
        @inbounds for i in eachindex(state)
            state[i] += σ * randn(rng)
        end
    else
        z = view(model.noise_buffer, :, task_index)
        @inbounds for i in eachindex(z)
            z[i] = randn(rng)
        end
        # state .+= σ * (L * z)
        mul!(state, model.noise_factor, z, σ, 1.0)
    end
    return state
end

ParticleDA.get_state_dimension(model::GlacierModel) =
    model.parameters.nx * model.parameters.ny

ParticleDA.get_observation_dimension(model::GlacierModel) = length(model.sensor_indices)

ParticleDA.get_state_eltype(::Type{<:GlacierModel{T}}) where {T} = T
ParticleDA.get_state_eltype(model::GlacierModel) = ParticleDA.get_state_eltype(typeof(model))

ParticleDA.get_observation_eltype(::Type{<:GlacierModel{T}}) where {T} = T
ParticleDA.get_observation_eltype(model::GlacierModel) =
    ParticleDA.get_observation_eltype(typeof(model))

ParticleDA.get_covariance_observation_noise(model::GlacierModel) = model.obs_cov

# Prior mean — state is now β directly (no log).
function ParticleDA.get_initial_state_mean!(
    state_mean::AbstractVector{T}, model::GlacierModel
) where {T<:Real}
    state_mean .= model.beta_prior_mean
    return state_mean
end

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T},
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
) where {T<:Real}
    ParticleDA.get_initial_state_mean!(state, model)
    _apply_noise!(state, model, rng, model.parameters.init_std_beta, task_index)
    floor_β = model.parameters.min_beta
    @inbounds for i in eachindex(state)
        state[i] = max(state[i], floor_β)          # keep β positive
    end
    return state
end

# Upwind advection on β.  State IS β (no log transform).
#
# dt = time_step / n_integration_step.  A one-shot CFL safety check warns if
# the chosen dt would violate the upwind stability ceiling.
const _CFL_WARNED = Ref(false)

function ParticleDA.update_state_deterministic!(
    state::AbstractVector,
    model::GlacierModel,
    time_index::Integer,
    task_index::Integer=1,
)
    p = model.parameters
    nx, ny = p.nx, p.ny
    dx = p.x_length / nx
    ε = p.advection_epsilon

    β = reshape(state, ny, nx)                      # alias the state directly
    β_new = similar(β)

    dt = p.time_step / p.n_integration_step

    # ── CFL check ────────────────────────────────────────────────────────
    if !_CFL_WARNED[]
        max_speed_now = p.advection_type == "nonlinear" ?
                        maximum(1 .+ ε .* β) : 1.0
        cfl_safe = 0.2 * dx / max_speed_now
        if dt > cfl_safe
            @warn "CFL violation likely" dt cfl_safe
        end
        _CFL_WARNED[] = true
    end

    # ── Advection inner loop ─────────────────────────────────────────────
    # Three branches share periodic BC, buffer swap, and finite-difference
    # discretisation. They differ in the stencil:
    #   "linear":       upwind, v = 1.        D_num = ½·v·Δx·(1−CFL) > 0
    #   "nonlinear":    upwind, v = 1 + ε·β.  Same diffusion plus β-feedback.
    #   "lax_wendroff": centred-diff + ½·CFL² correction.  D_num ≡ 0 to 2nd order,
    #                   but introduces dispersion (oscillations near sharp edges).
    # See glacier-notes/21_numerical_diffusion.md for the modified-equation
    # derivation and quantitative damping predictions.
    if p.advection_type == "lax_wendroff"
        for _ in 1:p.n_integration_step
            @inbounds for j in 1:ny
                for i in 1:nx
                    im = mod1(i - 1, nx)
                    ip = mod1(i + 1, nx)
                    v_i  = 1.0 + ε * β[j, i]   # local velocity (linear if ε ≈ 0)
                    α    = v_i * dt / dx
                    β_new[j, i] = β[j, i] -
                                  0.5 * α     * (β[j, ip] - β[j, im]) +
                                  0.5 * α * α * (β[j, ip] - 2*β[j, i] + β[j, im])
                end
            end
            β, β_new = β_new, β
        end
    else
        is_nonlinear = p.advection_type == "nonlinear"
        for _ in 1:p.n_integration_step
            @inbounds for j in 1:ny
                for i in 1:nx
                    im = mod1(i - 1, nx)
                    dβdx = (β[j, i] - β[j, im]) / dx
                    velocity = is_nonlinear ? (1 + ε * β[j, i]) : 1.0
                    β_new[j, i] = β[j, i] - velocity * dt * dβdx
                end
            end
            β, β_new = β_new, β
        end
    end

    # state already aliases the same storage as β through reshape, but we
    # explicitly copy back to be safe under the buffer-swap above.
    state .= vec(β)
    return state
end

function ParticleDA.update_state_stochastic!(
    state::AbstractVector,
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
)
    _apply_noise!(state, model, rng, model.parameters.process_std_beta, task_index)
    floor_β = model.parameters.min_beta
    @inbounds for i in eachindex(state)
        state[i] = max(state[i], floor_β)
    end
    return state
end

# Surrogate β -> ux: speed = 1000 / β.  State is β directly so no exp.
function surrogate_ux!(ux_flat::AbstractVector, state::AbstractVector,
                       floor_β::Real)
    @inbounds for i in eachindex(state)
        β_i = max(state[i], floor_β)
        ux_flat[i] = 1.0e3 / β_i
    end
    return ux_flat
end

function ParticleDA.get_observation_mean_given_state!(
    observation_mean::AbstractVector,
    state::AbstractVector,
    model::GlacierModel,
    task_index::Integer=1,
)
    ux_scratch = view(model.state_buffer, :, task_index)
    surrogate_ux!(ux_scratch, state, model.parameters.min_beta)
    obs_space = model.parameters.obs_space
    @inbounds for (k, idx) in enumerate(model.sensor_indices)
        if obs_space == "velocity"
            observation_mean[k] = ux_scratch[idx]
        elseif obs_space == "log_velocity"
            observation_mean[k] = log(ux_scratch[idx])
        else
            observation_mean[k] = log(max(state[idx], model.parameters.min_beta))
        end
    end
    return observation_mean
end

function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector,
    state::AbstractVector,
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
)
    ParticleDA.get_observation_mean_given_state!(observation, state, model, task_index)
    σ = _obs_sigma(model.parameters)
    @inbounds for k in eachindex(observation)
        observation[k] += σ * randn(rng)
    end
    return observation
end

function ParticleDA.get_log_density_observation_given_state(
    observation::AbstractVector,
    state::AbstractVector,
    model::GlacierModel,
    task_index::Integer=1,
)
    if model.parameters.disable_observations
        # Ablation mode: every particle returns the same log-likelihood, so
        # weights stay uniform and the filter never learns from observations.
        return zero(eltype(observation))
    end
    obs_mean = view(model.obs_buffer, :, task_index)
    ParticleDA.get_observation_mean_given_state!(obs_mean, state, model, task_index)
    return -invquad(model.obs_cov, observation .- obs_mean) / 2
end

function _write_parameters_group(group::HDF5.Group, params::GlacierModelParameters)
    for field in fieldnames(typeof(params))
        attributes(group)[string(field)] = getfield(params, field)
    end
end

function _sensor_xy(model::GlacierModel)
    p = model.parameters
    dx = p.x_length / p.nx
    dy = p.y_length / p.ny
    xs = Float64[]
    ys = Float64[]
    for idx in model.sensor_indices
        j = ((idx - 1) % p.ny) + 1   # row
        i = ((idx - 1) ÷ p.ny) + 1   # col
        push!(xs, (i - 1) * dx)
        push!(ys, (j - 1) * dy)
    end
    return xs, ys
end

function ParticleDA.write_model_metadata(file::HDF5.File, model::GlacierModel)
    p = model.parameters
    grid_x = collect(range(0.0, p.x_length; length=p.nx))
    grid_y = collect(range(0.0, p.y_length; length=p.ny))
    stations_x, stations_y = _sensor_xy(model)

    function _write_coords(g::HDF5.Group, x, y)
        for (name, val) in zip(("x", "y"), (x, y))
            ds, _ = create_dataset(g, name, val)
            ds[:] = val
            attributes(ds)["Description"] = "$name coordinate"
            attributes(ds)["Unit"] = "m"
        end
    end

    for (name, write_group) in [
        ("parameters", g -> _write_parameters_group(g, p)),
        ("grid_coordinates", g -> _write_coords(g, grid_x, grid_y)),
        ("station_coordinates", g -> _write_coords(g, stations_x, stations_y)),
    ]
        if !haskey(file, name)
            g = create_group(file, name)
            write_group(g)
        else
            @warn "Group $name already exists in $(file.filename)"
        end
    end

    if !haskey(file, "beta_prior")
        ds, _ = create_dataset(file, "beta_prior",
                               reshape(model.beta_prior_mean, p.ny, p.nx))
        ds[:, :] = reshape(model.beta_prior_mean, p.ny, p.nx)
        attributes(ds)["Description"] = "Prior mean beta field"
        attributes(ds)["Unit"] = "Pa s / m"
    end
    if !haskey(file, "beta_truth_prior")
        ds, _ = create_dataset(file, "beta_truth_prior",
                               reshape(model.truth_prior_mean, p.ny, p.nx))
        ds[:, :] = reshape(model.truth_prior_mean, p.ny, p.nx)
        attributes(ds)["Description"] = "Truth initial beta field"
        attributes(ds)["Unit"] = "Pa s / m"
    end
end

function ParticleDA.write_state(
    file::HDF5.File,
    state::AbstractVector{T},
    time_index::Int,
    group_name::String,
    model::GlacierModel,
) where {T}
    p = model.parameters
    subgroup_name = ParticleDA.time_index_to_hdf5_key(time_index)
    _, subgroup = ParticleDA.create_or_open_group(file, group_name, subgroup_name)

    # State IS β. Also derive log_beta for backward compatibility with old plots.
    beta = reshape(state, p.ny, p.nx)
    log_beta = log.(max.(beta, p.min_beta))

    for (name, field, unit, desc) in [
        ("beta", beta, "Pa s / m", "Basal-friction field (state)"),
        ("log_beta", log_beta, "log(Pa s / m)", "Log basal-friction (derived)"),
    ]
        if !haskey(subgroup, name)
            subgroup[name] = field
            a = attributes(subgroup[name])
            a["Description"] = desc
            a["Unit"] = unit
            a["Time index"] = time_index
            a["Time"] = time_index * p.time_step
        end
    end
end

end # module
