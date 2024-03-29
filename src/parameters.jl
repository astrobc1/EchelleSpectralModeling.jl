struct Parameters
    values::Vector{Float64}
    bounds::Vector{NTuple{2, Float64}}
    errors::Vector{Float64}
    vary::Vector{Bool}
    indices::OrderedDict{String, Int}
end

Parameters() = Parameters(Float64[], NTuple{2, Float64}[], Float64[], Bool[], OrderedDict{String, Int}())

function Base.push!(
        params::Parameters;
        name::String, value::Real,
        bounds::Tuple{<:Real, <:Real}=(-Inf, Inf),
        vary::Bool=bounds[1] != bounds[2]
    )
    push!(params.values, value)
    push!(params.bounds, bounds)
    push!(params.errors, NaN)
    push!(params.vary, vary)
    params.indices[name] = length(params.values)
    return params
end


function Base.setindex!(params::Parameters, kwargs::NamedTuple, name::String)
    if name in keys(params.indices)
        k = params.indices[name]
        if :value in keys(kwargs)
            params.values[k] = kwargs.value
        end
        if :bounds in keys(kwargs)
            params.bounds[k] = kwargs.bounds
        end
        if :errors in keys(kwargs)
            params.errors[k] = kwargs.error
        end
        if :vary in keys(kwargs)
            params.vary[k] = kwargs.vary
        end
    else
        push!(params; name, kwargs...)
    end
end


# function Base.setindex!(params::Parameters, value::Real, name::String)
#     if name in keys(params.indices)
#         params[params.indices[name]] = value
#     else
#         push!()
#     end
# end
Base.getindex(params::Parameters, name::String) = params.values[params.indices[name]]
Base.getindex(params::Parameters, index::Int) = params.values[index]


function set_value!(params::Parameters, name::String, value::Real)
    params.values[params.indices[name]] = value
    return params
end


function Base.show(io::IO, params::Parameters)
    for (i, name) in enumerate(keys(params.indices))
        lock = params.vary[i] ? " " : " 🔒 "
        println(io, " $(name) | Value = $(params.values[i])$(lock)| Bounds = ($(params.bounds[i][1]), $(params.bounds[i][2]))")
    end
end


function get_initial_value(bounds::NTuple{2, <:Real}; f::Real=0.51)
    lo, hi = bounds
    if lo == hi
        return lo
    elseif abs(lo) == hi
        r = hi - lo
        return lo + f * r
    else
        @assert lo < hi
        r = hi - lo
        return lo + 0.5 * r
    end
end