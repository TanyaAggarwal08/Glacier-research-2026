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
    init_std_beta::T = 300.0
    process_std_beta::T = 7.0
    # Lower clamp on β to keep the surrogate ux = 1000/β well-defined.
    # The default lets β range comfortably; only triggers in extreme noise tails.
    min_beta::T = 10.0
    obs_noise_std::T = 0.10
    advection_epsilon::T = 5e-4
    n_integration_step::Int = 1
    time_step::T = 1.0
    # Spatial correlation length (m) for smooth noise. Both the initial-state
    # perturbation and the per-step process noise are drawn from a Gaussian
    # field with squared-exponential covariance, marginal std = the *_std_beta
    # parameter above, and correlation length set here. Setting ≤ 0 falls back
    # to iid noise (legacy behaviour).
    noise_length_scale::T = 30_000.0
    # Ablation flag: when true, get_log_density returns 0 for every particle,
    # so weights stay uniform and the filter ignores observations entirely.
    # Truth generation and observation simulation are unaffected.
    disable_observations::Bool = false
end

struct GlacierModel{T<:AbstractFloat}
    parameters::GlacierModelParameters{T}
    beta_prior_mean::Vector{T}     # mean β field, flat, length nx*ny
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

function init(parameters_dict::Dict, n_tasks::Int=1)
    raw = get(parameters_dict, "glacier", Dict())
    user_input = (; (Symbol(k) => v for (k, v) in raw)...)
    p = GlacierModelParameters(; user_input...)

    ω = 2π / p.x_length
    xs = range(0.0, p.x_length; length=p.nx)
    ys = range(0.0, p.y_length; length=p.ny)
    # Prior mean field — same sinusoid as before, but stored directly (no log).
    β_prior = [1000.0 + 500.0 * sin(ω * xi) * sin(ω * yj) for yj in ys, xi in xs]
    β_prior_flat = vec(β_prior)

    sensors = _build_sensor_indices(p)
    n_obs = length(sensors)
    obs_cov = ScalMat(n_obs, p.obs_noise_std^2)

    obs_buffer = zeros(Float64, n_obs, n_tasks)
    state_buffer = zeros(Float64, p.nx * p.ny, n_tasks)
    L = _build_noise_factor(p)
    noise_buffer = zeros(Float64, p.nx * p.ny, n_tasks)

    return GlacierModel{Float64}(p, β_prior_flat, sensors, obs_cov,
                                 obs_buffer, state_buffer, L, noise_buffer)
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

    if !_CFL_WARNED[]
        max_speed_now = maximum(1 .+ ε .* β)
        cfl_safe = 0.2 * dx / max_speed_now
        if dt > cfl_safe
            @warn "CFL violation likely" dt cfl_safe
        end
        _CFL_WARNED[] = true
    end

    for _ in 1:p.n_integration_step
        @inbounds for j in 1:ny
            for i in 1:nx
                im = mod1(i - 1, nx)
                dβdx = (β[j, i] - β[j, im]) / dx
                velocity = 1 + ε * β[j, i]
                β_new[j, i] = β[j, i] - velocity * dt * dβdx
            end
        end
        β, β_new = β_new, β
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
    @inbounds for (k, idx) in enumerate(model.sensor_indices)
        observation_mean[k] = ux_scratch[idx]
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
    σ = model.parameters.obs_noise_std
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
