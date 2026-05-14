module Glacier

using ParticleDA
using HDF5
using Random
using LinearAlgebra
using PDMats
using Distributions

Base.@kwdef struct GlacierModelParameters{T<:AbstractFloat}
    nx::Int = 40
    ny::Int = 40
    x_length::T = 160_000.0
    y_length::T = 160_000.0
    sensor_stride::Int = 16
    init_std_theta::T = 0.05
    process_std_theta::T = 0.007
    obs_noise_std::T = 0.05
    advection_epsilon::T = 5e-4
    n_integration_step::Int = 1
    time_step::T = 1.0
end

struct GlacierModel{T<:AbstractFloat}
    parameters::GlacierModelParameters{T}
    beta_prior_logmean::Vector{T}
    sensor_indices::Vector{Int}
    obs_cov::ScalMat{T}
    obs_buffer::Matrix{T}
    state_buffer::Matrix{T}
end

function init(parameters_dict::Dict, n_tasks::Int=1)
    raw = get(parameters_dict, "glacier", Dict())
    user_input = (; (Symbol(k) => v for (k, v) in raw)...)
    p = GlacierModelParameters(; user_input...)

    ω = 2π / p.x_length
    xs = range(0.0, p.x_length; length=p.nx)
    ys = range(0.0, p.y_length; length=p.ny)
    β_prior = [1000.0 + 500.0 * sin(ω * xi) * sin(ω * yj) for yj in ys, xi in xs]
    θ₀ = log.(vec(β_prior))

    sensors = collect(1:p.sensor_stride:(p.nx * p.ny))
    n_obs = length(sensors)
    obs_cov = ScalMat(n_obs, p.obs_noise_std^2)

    obs_buffer = zeros(Float64, n_obs, n_tasks)
    state_buffer = zeros(Float64, p.nx * p.ny, n_tasks)

    return GlacierModel{Float64}(p, θ₀, sensors, obs_cov, obs_buffer, state_buffer)
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

function ParticleDA.get_initial_state_mean!(
    state_mean::AbstractVector{T}, model::GlacierModel
) where {T<:Real}
    state_mean .= model.beta_prior_logmean
    return state_mean
end

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T},
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
) where {T<:Real}
    ParticleDA.get_initial_state_mean!(state, model)
    σ = model.parameters.init_std_theta
    @inbounds for i in eachindex(state)
        state[i] += σ * randn(rng)
    end
    return state
end

# β = exp(state); upwind advection on β; state .= log(β_new).
# Mirrors glacier-code/testing_stage/particlefilteringwithiceflow.jl:146-172.
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

    β = reshape(exp.(state), ny, nx)
    β_new = similar(β)

    for _ in 1:p.n_integration_step
        max_speed = maximum(1 .+ ε .* β)
        dt = 0.2 * dx / max_speed
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

    state .= log.(vec(β))
    return state
end

function ParticleDA.update_state_stochastic!(
    state::AbstractVector,
    model::GlacierModel,
    rng::Random.AbstractRNG,
    task_index::Integer=1,
)
    σ = model.parameters.process_std_theta
    @inbounds for i in eachindex(state)
        state[i] += σ * randn(rng)
    end
    return state
end

# Surrogate β -> ux: speed = 1e3 / β (no y-component used for obs).
# Matches particlefilteringwithiceflow.jl:98-113.
function surrogate_ux!(ux_flat::AbstractVector, state::AbstractVector)
    @inbounds for i in eachindex(state)
        ux_flat[i] = 1.0e3 / (exp(state[i]) + 1.0e-6)
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
    surrogate_ux!(ux_scratch, state)
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
        ds, _ = create_dataset(file, "beta_prior", reshape(exp.(model.beta_prior_logmean), p.ny, p.nx))
        ds[:, :] = reshape(exp.(model.beta_prior_logmean), p.ny, p.nx)
        attributes(ds)["Description"] = "Prior mean beta field (used to centre log-beta state)"
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

    log_beta = reshape(state, p.ny, p.nx)
    beta = exp.(log_beta)

    for (name, field, unit, desc) in [
        ("log_beta", log_beta, "log(Pa s / m)", "Log basal-friction field"),
        ("beta", beta, "Pa s / m", "Basal-friction field"),
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
