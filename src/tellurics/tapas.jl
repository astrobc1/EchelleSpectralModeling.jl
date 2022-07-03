using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using NaNStatistics
using NPZ

export TAPASTellurics, has_water_features, has_airmass_features, get_mask

struct TAPASTellurics <: SpectralModelComponent
    input_file::String
    min_feature_flux::Float64
    vel_guess::Vector{Float64}
    τ_water_guess::Vector{Float64}
    τ_airmass_guess::Vector{Float64}
end

"""
    TAPASTellurics(;input_file::String, min_feature_flux=0.02, Δv_guess=[-100, 10, 100], τ_water_guess=[0.1, 1.1, 4], τ_airmass_guess=[0.8, 1.1, 3])
Construct a TAPASTellurics model component. The templates are stored within `input_file`. C02, N2O, O2, O3, and CH4 are combined into a single template and utilize the parameter `airmass_depth`. H2O is kept separate and utilizes `water_depth`. If there are no features less than `min_feature_flux`, the parameters for this chunk are fixed for the appropriate template.
"""
function TAPASTellurics(;input_file::String, min_feature_flux::Real=0.02, vel_guess::Vector{<:Real}=[-100, 10, 100], τ_water_guess::Vector{<:Real}=[0.1, 1.1, 4], τ_airmass_guess::Vector{<:Real}=[0.8, 1.1, 3])
    return TAPASTellurics(input_file, min_feature_flux, vel_guess, τ_water_guess, τ_airmass_guess)
end

function EchelleSpectralModeling.build(m::TAPASTellurics, pars::Parameters, templates::Dict)
    return build(m, templates["λ"], templates["tellurics"][:, 1], templates["tellurics"][:, 2], pars["τ_water"].value, pars["τ_airmass"].value, pars["vel_tel"].value)
end

function EchelleSpectralModeling.build(m::TAPASTellurics, λ::Vector{<:Real}, flux_water::Vector{<:Real}, flux_airmass::Vector{<:Real}, τ_water::Real, τ_airmass::Real, vel::Real)
    flux = flux_water.^τ_water .* flux_airmass.^τ_airmass
    return maths.doppler_shift_flux(λ, flux, vel)
end

function has_water_features(m::TAPASTellurics, templates::Dict, kernel=nothing)
    flux = templates["tellurics"][:, 1]
    if !isnothing(kernel)
        flux .= maths.convolve1d(flux, kernel)
        flux ./= nanmaximum(flux)
    end
    return any(flux .< 1 - m.min_feature_flux)
end
function has_airmass_features(m::TAPASTellurics, templates::Dict, kernel=nothing)
    flux = templates["tellurics"][:, 2]
    if !isnothing(kernel)
        flux .= maths.convolve1d(flux, kernel)
        flux ./= nanmaximum(flux)
    end
    return any(flux .< 1 - m.min_feature_flux)
end

function load_tapas_templates(input_file::String, λ_out)

    # Raw file
    template_raw = npzread(input_file)

    # Wavelength
    λ = template_raw["wavelength"]
    λi, λf = λ_out[1], λ_out[end]
    good = findall((λ .> λi) .&& (λ .< λf))
    λ = λ[good]
    
    # Water
    flux_water = template_raw["water"][good]
    nx = length(λ_out)
    templates = ones(nx, 2)
    templates[:, 1] .= maths.cspline_interp(λ, flux_water, λ_out)
    templates[:, 1] ./= maths.weighted_median(templates[:, 1], p=0.999)
    
    # Remaining do in a loop
    for species ∈ keys(template_raw)
        if species ∉ ["water", "wavelength"]
            flux = template_raw[species][good]
            templates[:, 2] .*= maths.cspline_interp(λ, flux, λ_out)
        end
    end
    templates[:, 2] ./= maths.weighted_median(templates[:, 2], p=0.999)
    return templates
end

function EchelleSpectralModeling.load_template(m::TAPASTellurics, λ_out::AbstractVector)
    return load_tapas_templates(m.input_file, λ_out)
end

function EchelleSpectralModeling.get_init_parameters(m::TAPASTellurics, data::SpecData1d, sregion::SpecRegion1d)
    pars = Parameters()
    pars["τ_water"] = Parameter(value=m.τ_water_guess[2], lower_bound=m.τ_water_guess[1], upper_bound=m.τ_water_guess[3])
    pars["τ_airmass"] = Parameter(value=m.τ_airmass_guess[2], lower_bound=m.τ_airmass_guess[1], upper_bound=m.τ_airmass_guess[3])
    pars["vel_tel"] = Parameter(value=m.vel_guess[2], lower_bound=m.vel_guess[1], upper_bound=m.vel_guess[3])
    return pars
end


function get_mask(m::TAPASTellurics, pars::Parameters, templates::Dict, kernel=nothing, λ_out=nothing)
    tell_flux = build(m, pars, templates)
    if !isnothing(kernel)
        tell_flux .= maths.convolve1d(tell_flux, kernel)
        tell_flux ./= maximum(tell_flux)
    end
    if isnothing(λ_out)
        λ_out = templates["λ"]
    end
    tell_flux = maths.cspline_interp(templates["λ"], tell_flux, λ_out)
    mask = zeros(length(tell_flux))
    good = findall(tell_flux .> m.min_feature_flux)
    mask[good] .= 1
    return mask
end