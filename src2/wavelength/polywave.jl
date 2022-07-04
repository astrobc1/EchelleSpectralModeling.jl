using SpecialMatrices

using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

using PyCall
using Infiltrator
using Statistics

export PolyλSolution

struct PolyλSolution <: SpectralModelComponent
    deg::Int
    bounds::Vector{Float64}
end

"""
    PolyλSolution(;deg::Int, bounds::Vector{Float64})
Construct a PolyλSolution model component of degree `deg`. The optimized parameters are set points opposed to coefficients. Each set point is(in units of wavelength) is bounded by `bounds`.
"""
PolyλSolution(;deg::Int, bounds::Vector{Float64}) = PolyλSolution(deg, bounds)

function get_pixel_lagrange_points(m::PolyλSolution, sregion)
    return Int.(round.(collect(range(sregion.pixmin+1, sregion.pixmax-1, length=m.deg + 1))))
end

function get_λ_lagrange_zero_points(m::PolyλSolution, sregion, λ_estimate)
    pixel_set_points = get_pixel_lagrange_points(m::PolyλSolution, sregion)
    λ_zero_points = λ_estimate[pixel_set_points]
    return λ_zero_points
end


function EchelleSpectralModeling.build(m::PolyλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    pixel_lagrange_points = get_pixel_lagrange_points(m, sregion)
    λ_lagrange_points = [pars["λ$i"].value for i=1:m.deg+1]
    return build(m, pixel_lagrange_points, λ_lagrange_points, length(data.data.flux))
end

function EchelleSpectralModeling.build(m::PolyλSolution, xs::AbstractVector, λs::AbstractVector, nx::Int)
    V = Vandermonde(xs)
    pcoeffs = V \ λs
    λ = Polynomial(pcoeffs).(1:nx)
    return λ
end

function EchelleSpectralModeling.get_init_parameters(m::PolyλSolution, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    λ_estimate = get_λsolution_estimate(data, sregion)
    λ_zero_points = get_λ_lagrange_zero_points(m, sregion, λ_estimate)
    for i=1:m.deg+1
        pars["λ$i"] = Parameter(value=λ_zero_points[i], lower_bound=λ_zero_points[i] + m.bounds[1], upper_bound=λ_zero_points[i] + m.bounds[2])
    end
    return pars
end