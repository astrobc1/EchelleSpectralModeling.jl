using Polynomials
using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export PolyContinuum

struct PolyContinuum <: SpectralModelComponent
    deg::Int
    coeffs_guess::Dict{Int, Vector{Float64}}
end

"""
    PolyContinuum(;deg::Int, coeffs_guess::Vector{Float64})
Construct a PolyContinuum model component of degree `deg`.
"""
PolyContinuum(;deg::Int, coeffs_guess::Vector{Float64}) = PolyContinuum(deg, coeffs_guess)

function EchelleSpectralModeling.get_init_parameters(m::PolyContinuum, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    for i=0:m.deg
        pars["c$i"] = Parameter(value=m.coeffs_guess[i][2], lower_bound=m.coeffs_guess[i][1], upper_bound=m.coeffs_guess[i][3])
    end
    return pars
end

"""
    build(m::PolyContinuum, pars::Parameters, sregion::SpecRegion1d, λ_out::AbstractVector{<:Real}; λ0=nothing)
Build the polynomial continuum model.
"""
function EchelleSpectralModeling.build(m::PolyContinuum, pars::Parameters, sregion::SpecRegion1d, λ_out; λ0=nothing)
    coeffs = [pars["c$i"].value for i=0:m.deg]
    return build(m, coeffs, λ_out, λ0=λ0)
end

"""
    build(m::PolyContinuum, coeffs::Vector, λ_out; λ0=nothing)
"""
function EchelleSpectralModeling.build(m::PolyContinuum, coeffs::Vector, λ_out::AbstractVector; λ0=nothing)
    if isnothing(λ0)
        n = length(λ_out)
        λ0 = λ_out[Int(floor(n / 2))]
    end
    y = Polynomial(coeffs).(λ_out .- λ0)
    return y
end

"""
    estimate_continuum(x, flux; med_filter_width, p=0.98)
Utility function to estimate the continuum from the 1d spectrum.
"""
function estimate_continuum(x, flux; med_filter_width, p=0.95, deg=6)
    c = maths.generalized_median_filter1d(flux, width=med_filter_width, p=p)
    good = findall(isfinite.(x) .&& isfinite.(c))
    p = @views Polynomials.fit(x[good], c[good], deg)
    return p
end