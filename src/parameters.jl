struct Parameters
    values::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    errors::Vector{Float64}
    vary::BitVector
    indices::OrderedDict{String, Int}
end

Parameters() = Parameters(Float64[], Float64[], Float64[], Float64[], BitVector(), Dict{String, Int}())

function Base.push!(
        params::Parameters;
        name::String, value::Real,
        lower_bound::Real=-Inf, upper_bound::Real=Inf,
        vary::Bool=upper_bound != lower_bound
    )
    push!(params.values, value)
    push!(params.lower_bounds, lower_bound)
    push!(params.upper_bounds, upper_bound)
    push!(params.errors, NaN)
    push!(params.vary, vary)
    params.indices[name] = length(params.values)
    return params
end

Base.setindex!(params::Parameters, kwargs::NamedTuple, name::String) = push!(params; name, kwargs...)
Base.setindex!(params::Parameters, value::Real, name::String) = params[params.indices[name]] = value
Base.getindex(params::Parameters, name::String) = params.values[params.indices[name]]
Base.getindex(params::Parameters, index::Int) = params.values[index]

function set_value!(params::Parameters; name::String, value::Real)
    params.values[params.indices[name]] = value
    return params
end

function Base.show(io::IO, params::Parameters)
    for (i, name) in enumerate(keys(params.indices))
        if params.vary[i]
            println(io, " $(name) | Value = $(params.values[i]) | Bounds = [$(params.lower_bounds[i]), $(params.upper_bounds[i])]")
        else
            println(io, " $(name) | Value = $(params.values[i]) 🔒 | Bounds = [$(params.lower_bounds[i]), $(params.upper_bounds[i])]")
        end
    end
end