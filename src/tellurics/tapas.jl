using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using NPZ

using NaNStatistics

export TAPASTellurics, has_water_features, has_airmass_features, get_mask

struct TAPASTellurics <: SpectralModelComponent
    input_file::String
    min_feature_depth::Float64
    vel_guess::Vector{Float64}
    water_depth_guess::Vector{Float64}
    airmass_depth_guess::Vector{Float64}
end


function TAPASTellurics(;input_file, min_feature_depth=0.02, vel_guess=[-100, 10, 100], water_depth_guess=[0.1, 1.1, 4], airmass_depth_guess=[0.8, 1.1, 3])
    return TAPASTellurics(input_file, min_feature_depth, vel_guess, water_depth_guess, airmass_depth_guess)
end


function EchelleSpectralModeling.build(m::TAPASTellurics, pars, templates)
    return build(m, templates["λ"], templates["tellurics"][:, 1], templates["tellurics"][:, 2], pars["water_depth"].value, pars["airmass_depth"].value, pars["vel_tel"].value)
end

function EchelleSpectralModeling.build(m::TAPASTellurics, λ, flux_water, flux_airmass, τ_water, τ_airmass, vel)
    flux = flux_water.^τ_water .* flux_airmass.^τ_airmass
    return maths.doppler_shift_flux(λ, flux, vel)
end

function has_water_features(m::TAPASTellurics, templates, kernel=nothing)
    flux = templates["tellurics"][:, 1]
    if !isnothing(kernel)
        flux .= maths.convolve1d(flux, kernel)
        flux ./= nanmaximum(flux)
    end
    return any(flux .< 1 - m.min_feature_depth)
end
function has_airmass_features(m::TAPASTellurics, templates, kernel=nothing)
    flux = templates["tellurics"][:, 2]
    if !isnothing(kernel)
        flux .= maths.convolve1d(flux, kernel)
        flux ./= nanmaximum(flux)
    end
    return any(flux .< 1 - m.min_feature_depth)
end

function _load_template(input_file::String, λ_out)

    # Raw file
    template_raw = npzread(input_file)

    # Wavelength
    λ = template_raw["wavelength"]
    λi, λf = λ_out[1], λ_out[end]
    good = findall((λ .> λi) .& (λ .< λf))
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

function EchelleSpectralModeling.load_template(m::TAPASTellurics, λ_out)
    return _load_template(m.input_file, λ_out)
end

function EchelleSpectralModeling.get_init_parameters(m::TAPASTellurics, data, sregion)
    pars = Parameters()
    pars["water_depth"] = Parameter(value=m.water_depth_guess[2], lower_bound=m.water_depth_guess[1], upper_bound=m.water_depth_guess[3])
    pars["airmass_depth"] = Parameter(value=m.airmass_depth_guess[2], lower_bound=m.airmass_depth_guess[1], upper_bound=m.airmass_depth_guess[3])
    pars["vel_tel"] = Parameter(value=m.vel_guess[2], lower_bound=m.vel_guess[1], upper_bound=m.vel_guess[3])
    return pars
end

function get_mask(m::TAPASTellurics, pars, templates, kernel=nothing, λ_out=nothing)
    tell_flux = build(m, pars, templates)
    if !isnothing(kernel)
        tell_flux .= maths.convole1d(tell_flux, kernel)
        tell_flux ./= nanmaximum(tell_flux)
    end
    if isnothing(λ_out)
        λ_out = templates["λ"]
    end
    tell_flux = maths.cspline_interp(templates["λ"], tell_flux, λ_out)
    mask = zeros(length(tell_flux))
    good = findall(tell_flux .> m.min_feature_depth)
    mask[good] .= 1
    return mask
end