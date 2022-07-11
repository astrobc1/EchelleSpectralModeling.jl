module GasCells

using NPZ
using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export GasCell

struct GasCell <: SpectralModelComponent
    input_file::String
    vel_guess::Vector{Float64}
    τ_guess::Vector{Float64}
end

"""
    GasCell(;input_file::String, shift_guess::Vector{Float64}=[0.0, 0.0, 0.0], depth_guess::Vector{Float64}=[1.0, 1.0, 1.0])
Constructs a GasCell model component.
"""
GasCell(;input_file::String, vel_guess::Vector{Float64}=[0.0, 0.0, 0.0], τ_guess::Vector{Float64}=[1.0, 1.0, 1.0]) = GasCell(input_file, float.(vel_guess), float.(τ_guess))

function EchelleSpectralModeling.build(m::GasCell, pars::Parameters, templates::Dict)
    return build(m, templates["λ"], templates["gascell"], pars["τ_gascell"].value, pars["vel_gascell"].value)
end

function EchelleSpectralModeling.build(m::GasCell, λ, flux, τ=1, vel=0)
    if vel != 0
        return maths.doppler_shift_flux(λ, flux.^τ, vel)
    else
        return flux.^τ
    end
end

function EchelleSpectralModeling.get_init_parameters(m::GasCell, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    pars["vel_gascell"] = Parameter(value=m.vel_guess[2], lower_bound=m.vel_guess[1], upper_bound=m.vel_guess[3])
    pars["τ_gascell"] = Parameter(value=m.τ_guess[2], lower_bound=m.τ_guess[1], upper_bound=m.τ_guess[3])
    return pars
end

function EchelleSpectralModeling.load_template(m::GasCell, λ_out::AbstractVector)
    template_raw = npzread(m.input_file)
    template = maths.cspline_interp(template_raw["wavelength"], template_raw["flux"], λ_out)
    return template
end

end