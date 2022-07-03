using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export SplineContinuum

"""
Container for a SplineContinuum model.

# Fields
- `n_splines::Int` The number of splines.
- `bounds::Vector{Float64}`: The bounds for each knot (in units of flux).
"""
struct SplineContinuum <: SpectralModelComponent
    n_splines::Int
    bounds::Vector{Float64}
end

"""
    SplineContinuum(;n_splines::Int, bounds::Vector{Float64})
Construct a SplineContinuum model component with `n_splines`.
"""
SplineContinuum(;n_splines::Int, bounds::Vector{Float64}) = SplineContinuum(n_splines, bounds)

"""
    get_init_parameters(m::SplineContinuum, data::SpecData1d, sregion::SpecRegion1d)
Gets the initial parameters for a SplineContinuum model component.
"""
function EchelleSpectralModeling.get_init_parameters(m::SplineContinuum, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    Δ = m.bounds[2] - m.bounds[1]
    v0 = m.bounds[1] + Δ / 2
    for i=1:m.n_splines+1
        pars["c$i"] = Parameter(value=v0, lower_bound=m.bounds[1], upper_bound=m.bounds[2])
    end
    return pars
end

function get_λ_lagrange_points(m::SplineContinuum, sregion)
    Δλ = sregion.λmax - sregion.λmin
    return collect(range(sregion.λmin - Δλ / 100, sregion.λmax + Δλ / 100, length=m.n_splines + 1))
end

"""
"""
function EchelleSpectralModeling.build(m::SplineContinuum, pars::Parameters, sregion::SpecRegion1d, λ_out)
    λs = get_λ_lagrange_points(m, sregion)
    knots = [pars["c$i"].value for i=1:m.n_splines+1]
    return build(m, λs, knots, λ_out)
end

"""
"""
function EchelleSpectralModeling.build(m::SplineContinuum, λs, knots, λ_out)
    continuum = maths.cspline_interp(λs, knots, λ_out)
    return continuum
end