using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling

export HermiteLSF, get_lsfkernel_λ_grid

struct HermiteLSF <: SpectralModelComponent
    deg::Int
    σ_guess::Vector{Float64}
    coeff_guess::Vector{Float64}
end

"""
    HermiteLSF(;deg::Int, σ_guess::Vector{Float64}, coeff_guess::Vector{Float64})
Construct a HermiteLSF model component of degree `n`, width `σ`, and coeffs. `n=0` corresponds to a pure Gaussian.
"""
HermiteLSF(;deg::Int, σ_guess::Vector{Float64}, coeff_guess::Vector{Float64}) = HermiteLSF(deg, σ_guess, coeff_guess)

function get_lsfkernel_λ_grid(m::HermiteLSF, δλ::Real)
    Δλ = 10 * m.σ_guess[3]
    n = Int(ceil(Δλ / δλ))
    if iseven(n)
        n += 1
    end
    λlsf = [Int(ceil(-n / 2)):Int(floor(n / 2));] .* δλ
    return λlsf
end

function EchelleSpectralModeling.build(m::HermiteLSF, pars::Parameters, templates::Dict)
    coeffs = [pars["a$k"].value for k=0:m.deg]
    λlsf = templates["λlsf"]
    return build(m, coeffs, λlsf)
end

function EchelleSpectralModeling.build(m::HermiteLSF, coeffs::Vector, λlsf::AbstractVector)
    σ = coeffs[1]
    herm = maths.hermfun(λlsf ./ σ, m.deg)
    kernel = herm[:, 1]
    if m.deg == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end
    for k=2:m.deg+1
        kernel .+= coeffs[k] .* herm[:, k]
    end
    kernel ./= sum(kernel)
    return kernel
end

function EchelleSpectralModeling.get_init_parameters(m::HermiteLSF, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    pars["a0"] = Parameter(value=m.σ_guess[2], lower_bound=m.σ_guess[1], upper_bound=m.σ_guess[3])
    for i=1:m.deg
        pars["a$i"] = Parameter(value=m.coeff_guess[2], lower_bound=m.coeff_guess[1], upper_bound=m.coeff_guess[3])
    end
    return pars
end