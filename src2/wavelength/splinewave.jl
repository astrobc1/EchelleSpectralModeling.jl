using SpecialMatrices

using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

using Infiltrator

export SplineλSolution


struct SplineλSolution <: SpectralModelComponent
    n_splines::Int
    bounds::Vector{Float64}
end

"""
    SplineλSolution(;n_splines::Int, bounds::Vector{Float64})
Construct a SplineλSolution model component with `n_splines` (# of knots = `n_splines + 1`). Each knot (in units of wavelength) is bounded by `bounds`.
"""
SplineλSolution(;n_splines::Int, bounds::Vector{Float64}) = SplineλSolution(n_splines, bounds)

function get_pixel_lagrange_points(m::SplineλSolution, sregion)
    return Int.(round.(collect(range(sregion.pixmin+1, sregion.pixmax-1, length=m.n_splines + 1))))
end

function get_λ_lagrange_zero_points(m::SplineλSolution, sregion, λ_estimate)
    pixel_set_points = get_pixel_lagrange_points(m::SplineλSolution, sregion)
    λ_zero_points = λ_estimate[pixel_set_points]
    return λ_zero_points
end


function EchelleSpectralModeling.build(m::SplineλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    pixel_lagrange_points = get_pixel_lagrange_points(m, sregion)
    λ_lagrange_points = [pars["λ$i"].value for i=1:m.n_splines+1]
    return build(m, pixel_lagrange_points, λ_lagrange_points, length(data.data.flux))
end

function EchelleSpectralModeling.build(m::SplineλSolution, xs::AbstractVector, λs::AbstractVector, nx::Int)
    λ = maths.cspline_interp(float.(xs), λs, [1.0:nx;])
    return λ
end

function EchelleSpectralModeling.get_init_parameters(m::SplineλSolution, data::SpecData1d, sregion)
    pars = Parameters()
    λ_estimate = get_λsolution_estimate(data, sregion)
    λ_zero_points = get_λ_lagrange_zero_points(m, sregion, λ_estimate)
    for i=1:m.n_splines+1
        pname = "λ$i"
        pars[pname] = Parameter(value=λ_zero_points[i], lower_bound=λ_zero_points[i] + m.bounds[1], upper_bound=λ_zero_points[i] + m.bounds[2])
    end
    return pars
end