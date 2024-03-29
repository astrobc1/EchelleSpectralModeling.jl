export SplineContinuum


struct SplineContinuum
    knots_λ::Vector{Float64}
    bounds::NTuple{2, Float64}
end


function SplineContinuum(;λ_range::Tuple{<:Real, <:Real}, n_knots::Int, bounds::Tuple{<:Real, <:Real})
    if n_knots > 0
        knots_λ = range(λ_range[1], stop=λ_range[2], length=n_knots)
    else
        knots_λ = Float64[(λ_range[1] + λ_range[2]) / 2]
    end
    return SplineContinuum(Float64.(knots_λ), Float64.(bounds))
end


function build(continuum::SplineContinuum, λ::AbstractVector{<:Real}, params::Parameters, data::DataFrame)
    knots_spec = [params["c$i"] for i=1:length(continuum.knots_λ)]
    y = DataInterpolations.CubicSpline(knots_spec, continuum.knots_λ, extrapolate=true)(λ)
    return y
end


function get_initial_params!(params::Parameters, continuum::SplineContinuum, data::DataFrame)
    c0 = get_initial_value(continuum.bounds)
    for i=1:length(continuum.knots_λ)
        params["c$i"] = (value=c0, bounds=continuum.bounds)
    end
end