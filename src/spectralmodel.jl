export AbstractSpectralForwardModel, SpectralForwardModel, load_templates!, get_init_parameters, build

using EchelleBase
using EchelleSpectralModeling

abstract type AbstractSpectralForwardModel end

struct SpectralForwardModel{S, T, G, W, P, C} <: AbstractSpectralForwardModel
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

function SpectralForwardModel(;star=nothing, tellurics=nothing, gascell=nothing, λsolution=nothing, lsf=nothing, continuum=nothing, sregion, oversample)
    return SpectralForwardModel(star, tellurics, gascell, λsolution, lsf, continuum, sregion, oversample, Dict{String, Array{Float64}}())
end

function get_model_λ_spacing(m::SpectralForwardModel, data)
    if isnothing(m.sregion.pixmin)
        Δx = length(data[1].data.flux)
    else
        Δx = m.sregion.pixmax - m.sregion.pixmin
    end
    Δλ = m.sregion.λmax - m.sregion.λmin
    δλ = (Δλ / Δx) / m.oversample
    return δλ
end

function get_model_λ_grid(m::SpectralForwardModel, data; pad=2)
    δλ = get_model_λ_spacing(m, data)
    λ = [m.sregion.λmin-pad:δλ:m.sregion.λmax+pad;]
    return λ
end

function build(m::SpectralForwardModel, pars::Parameters, data; interp=true)
        
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

function get_init_parameters(m::SpectralForwardModel, data)
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

function get_varied_parameters(pars)
    vp = Parameters()
    for par ∈ values(pars)
        if par.lower_bound == par.upper_bound
            vp[par.name] = par
        end
    end
    return vp
end

function load_templates(m::SpectralForwardModel, data)
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
        m.templates["λrel"] = get_kernel_λ(m.lsf, δλ)
    end
end