export PolyContinuum


struct PolyContinuum
    knots_λ::Vector{Float64}
    bounds::NTuple{2, Float64}
end


function PolyContinuum(;λ_range::Tuple{<:Real, <:Real}, deg::Int, bounds::Tuple{<:Real, <:Real})
    if deg > 0
        knots_λ = range(λ_range[1], stop=λ_range[2], length=deg + 1)
    else
        knots_λ = Float64[(λ_range[1] + λ_range[2]) / 2]
    end
    return PolyContinuum(Float64.(knots_λ), Float64.(bounds))
end


function build(continuum::PolyContinuum, λ::AbstractVector{<:Real}, params::Parameters, data::DataFrame)
    knots_spec = [params["c$i"] for i=1:length(continuum.knots_λ)]
    deg = length(continuum.knots_λ) - 1
    pfit = Polynomials.fit(ArnoldiFit, continuum.knots_λ, knots_spec, deg)
    y = pfit.(λ)
    return y
end


function get_initial_params!(params::Parameters, continuum::PolyContinuum, data::DataFrame)
    c0 = get_initial_value(continuum.bounds)
    for i=1:length(continuum.knots_λ)
        params["c$i"] = (value=c0, bounds=continuum.bounds)
    end
end