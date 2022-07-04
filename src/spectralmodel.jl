export AbstractSpectralForwardModel, SpectralForwardModel, load_templates!, get_init_parameters, build

using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

using Infiltrator

"""
    AbstractSpectralForwardModel
Abstract type for a spectral forward model.
"""
abstract type AbstractSpectralForwardModel end

# Optional Model Component
const OptionalSpectralModelComponent = Union{SpectralModelComponent, Nothing}

"""
    SpectralForwardModel{S<:SpectralModelComponent, T<:SpectralModelComponent, G<:SpectralModelComponent, W<:SpectralModelComponent, P<:SpectralModelComponent, C<:SpectralModelComponent}
# Primary container for a spectral forward model which models the star, tellurics, the continuum, the wavelength solution, a gas cell (optional), and the line spread function. The wavelength solution may be static using APrioriλSolution.
# Fields:
- `star`::SpectralModelComponent` The stellar model.
- `tellurics::SpectralModelComponent` The telluric model. Optional.
- `gascell::SpectralModelComponent` The gas cell model. Optional.
- `λsolution::SpectralModelComponent` The wavelength solution model.
- `lsf::SpectralModelComponent` The line spread function model.
- `continuum::SpectralModelComponent` The continuum model.
- `sregion::SpecRegion1d` The spectral region for this model.
- `oversample::Int` The oversampleing factor of the model.
- `templates::Dict{String, Array{Float64}}` Contains any templates.
"""
struct SpectralForwardModel{S<:OptionalSpectralModelComponent, T<:OptionalSpectralModelComponent, G<:OptionalSpectralModelComponent, W<:OptionalSpectralModelComponent, P<:OptionalSpectralModelComponent, C<:OptionalSpectralModelComponent} <: AbstractSpectralForwardModel
    star::S
    tellurics::T
    gascell::G
    λsolution::W
    lsf::P
    continuum::C
    sregion::SpecRegion1d
    oversample::Int
    templates::Dict{String, Array{Float64}}
end


"""
    SpectralForwardModel(;star=nothing, tellurics=nothing, gascell=nothing, λsolution=nothing, lsf=nothing, continuum=nothing, sregion, oversample)
Construct a SpectralForwardModel object.
"""
function SpectralForwardModel(;star=nothing, tellurics=nothing, gascell=nothing, λsolution=nothing, lsf=nothing, continuum=nothing, sregion, oversample)
    return SpectralForwardModel(star, tellurics, gascell, λsolution, lsf, continuum, sregion, oversample, Dict{String, Array{Float64}}())
end

function get_model_grid_δλ(m::SpectralForwardModel, data::Vector{SpecData1d{S}}) where{S}
    if isnothing(m.sregion.pixmin)
        Δx = length(data[1].data.flux)
    else
        Δx = m.sregion.pixmax - m.sregion.pixmin
    end
    Δλ = m.sregion.λmax - m.sregion.λmin
    δλ = (Δλ / Δx) / m.oversample
    return δλ
end

function get_model_λ_grid(m::SpectralForwardModel, data::Vector{SpecData1d{S}}; pad::Real=2)  where{S}
    δλ = get_model_grid_δλ(m, data)
    λ = [m.sregion.λmin-pad:δλ:m.sregion.λmax+pad;]
    return λ
end

"""
    build(m::SpectralForwardModel, pars::Parameters, data; interp=true)
Builds the model for a given set of parameters and given observation.
"""
function build(m::SpectralForwardModel, pars::Parameters, data::SpecData1d; interp=true)
        
    # Get model wave grid
    λhr = m.templates["λ"]
        
    # Init a model
    model_flux = ones(length(λhr))

    # Star
    if !isnothing(m.star)
        model_flux .*= build(m.star, pars, m.templates)
    end
    
    # Gas Cell
    if !isnothing(m.gascell)
        model_flux .*= build(m.gascell, pars, m.templates)
    end

    # All tellurics
    if !isnothing(m.tellurics)
        model_flux .*= build(m.tellurics, pars, m.templates)
    end
        
    # Convolve
    begin
        lags = [-500:0.1734:500;]
        ccf = maths.cross_correlate_doppler(λhr, maths.convolve1d(model_flux, kernel), λhr, model_flux, lags)
        plot(lags, ccf)
    end
    if !isnothing(m.lsf)
        kernel = build(m.lsf, pars, m.templates)
        model_flux .= maths.convolve1d(model_flux, kernel)
        model_flux ./= maths.weighted_median(model_flux, p=0.999)
    end

    # Continuum
    if !isnothing(m.continuum)
        model_flux .*= build(m.continuum, pars, m.sregion, λhr)
    end

    # Generate the wavelength solution of the data
    if !isnothing(m.λsolution)
        λ_data = build(m.λsolution, data, pars, m.sregion)
    end

    # Interpolate high res model onto data grid
    if interp
        model_flux = maths.cspline_interp(λhr, model_flux, λ_data)
        out = (λ_data, model_flux)
    else
        out = (λhr, model_flux)
    end
    
    # Return
    return out

end

"""
    get_init_parameters(m::SpectralForwardModel, data)
Gets the initial parameters for a given observation.
"""
function get_init_parameters(m::SpectralForwardModel, data::SpecData1d)
    pars = Parameters()
    if !isnothing(m.star)
        merge!(pars, get_init_parameters(m.star, data, m.sregion))
    end
    if !isnothing(m.tellurics)
        merge!(pars, get_init_parameters(m.tellurics, data, m.sregion))
    end
    if !isnothing(m.gascell)
        merge!(pars, get_init_parameters(m.gascell, data, m.sregion))
    end
    if !isnothing(m.λsolution)
        merge!(pars, get_init_parameters(m.λsolution, data, m.sregion))
    end
    if !isnothing(m.continuum)
        merge!(pars, get_init_parameters(m.continuum, data, m.sregion))
    end
    if !isnothing(m.lsf)
        merge!(pars, get_init_parameters(m.lsf, data, m.sregion))
    end
    return pars
end

"""
    load_templates!(m::SpectralForwardModel, data)
Loads in the stellar, tellurics, and gas cell templates (if any).
"""
function load_templates!(m::SpectralForwardModel, data::Vector{SpecData1d{S}}) where {S}
    m.templates["λ"] = get_model_λ_grid(m, data)
    if !isnothing(m.star)
        m.templates["star"] = load_template(m.star, m.templates["λ"])
    end
    if !isnothing(m.tellurics)
        m.templates["tellurics"] = load_template(m.tellurics, m.templates["λ"])
    end
    if !isnothing(m.gascell)
        m.templates["gascell"] = load_template(m.gascell, m.templates["λ"])
    end
    if !isnothing(m.lsf)
        δλ = m.templates["λ"][3] - m.templates["λ"][2]
        m.templates["λlsf"] = get_lsfkernel_λ_grid(m.lsf, δλ)
    end
end