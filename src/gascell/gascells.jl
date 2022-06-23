module GasCells

using NPZ
using CurveFitParameters

using EchelleBase
using EchelleSpectralModeling

export GasCell

struct GasCell <: SpectralModelComponent
    input_file::String
    shift_guess::Vector{Float64}
    depth_guess::Vector{Float64}
end

GasCell(;input_file, shift_guess=[0.0, 0.0, 0.0], depth_guess=[1.0, 1.0, 1.0]) = GasCell(input_file, shift_guess, depth_guess)

function EchelleSpectralModeling.build(m::GasCell, pars, templates)
    return build(m, templates["λ"], templates["gascell"], pars["gascell_depth"].value, pars["gascell_shift"].value)
end

function EchelleSpectralModeling.build(m::GasCell, λ, flux, τ, vel)
    return maths.doppler_shift_flux(λ, flux.^τ, vel)
end

function EchelleSpectralModeling.get_init_parameters(m::GasCell, data, sregion)
    pars = Parameters()
    pars["gascell_shift"] = Parameter(value=m.shift_guess[2], lower_bound=m.shift_guess[1], upper_bound=m.shift_guess[3])
    pars["gascell_depth"] = Parameter(value=m.depth_guess[2], lower_bound=m.depth_guess[1], upper_bound=m.depth_guess[3])
    return pars
end

function EchelleSpectralModeling.load_template(m::GasCell, λ_out)
    template_raw = npzread(m.input_file)
    template = maths.cspline_interp(template_raw["wavelength"], template_raw["flux"], λ_out)
    return template
end

end