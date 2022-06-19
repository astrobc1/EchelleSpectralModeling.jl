using NaNStatistics
using Polynomials
using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export PolyContinuum

struct PolyContinuum <: SpectralModelComponent
    deg::Int
    coeffs_guess::Dict{Int, Vector{Float64}}
end

PolyContinuum(;deg, coeffs_guess) = PolyContinuum(deg, coeffs_guess)

function EchelleSpectralModeling.get_init_parameters(m::PolyContinuum, data, sregion)
    pars = Parameters()
    for i=0:m.deg
        pars["c$i"] = Parameter(value=m.coeffs_guess[i][2], lower_bound=m.coeffs_guess[i][1], upper_bound=m.coeffs_guess[i][3])
    end
    return pars
end

function EchelleSpectralModeling.build(m::PolyContinuum, pars::Parameters, sregion, λ_out; λ0=nothing)
    coeffs = [pars["c$i"].value for i=0:m.deg]
    return build(m, coeffs, λ_out, λ0=λ0)
end

function EchelleSpectralModeling.build(m::PolyContinuum, coeffs::Vector, λ_out; λ0=nothing)
    if isnothing(λ0)
        n = length(λ_out)
        λ0 = λ_out[Int(floor(n / 2))]
    end
    y = Polynomial(coeffs).(λ_out .- λ0)
    return y
end