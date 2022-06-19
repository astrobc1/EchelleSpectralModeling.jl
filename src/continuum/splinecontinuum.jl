using NaNStatistics
using Polynomials
using DataInterpolations
using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export SplineContinuum

struct SplineContinuum <: SpectralModelComponent
    n_splines::Int
    bounds::Vector{Float64}
end

SplineContinuum(;n_splines, bounds) = SplineContinuum(n_splines, bounds)

function EchelleSpectralModeling.get_init_parameters(m::SplineContinuum, data, sregion)
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

function EchelleSpectralModeling.build(m::SplineContinuum, pars::Parameters, sregion, λ_out)
    λs = get_λ_lagrange_points(m, sregion)
    knots = [pars["c$i"].value for i=1:m.n_splines+1]
    return build(m, λs, knots, λ_out)
end

function EchelleSpectralModeling.build(m::SplineContinuum, λs, knots, λ_out)
    continuum = DataInterpolations.CubicSpline(knots, λs).(λ_out)
    return continuum
end