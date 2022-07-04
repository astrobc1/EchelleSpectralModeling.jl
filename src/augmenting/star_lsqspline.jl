using EchelleBase
using EchelleSpectralModeling
using NaNStatistics
using Infiltrator
using PyCall

export LSQCSplineStar

struct LSQCSplineStar <: TemplateAugmenter
    oversample::Float64
end

function fit_lsqcspline(λ::AbstractVector, flux::AbstractVector, weights::AbstractVector, knots::AbstractVector)
    scipyinterp = pyimport("scipy.interpolate")
    cspline_fit = scipyinterp.LSQUnivariateSpline(λ, flux, knots, w=weights, k=3, ext=1)
    return cspline_fit
end

# Updates the current lsq object
function augment_star!(model::SpectralForwardModel, data::Vector{SpecData1d{S}}, opt_results::Vector, augmenter::LSQCSplineStar) where {S}

    # Master λ grid
    λ = model.templates["λ"]
    
    # Temp storage arrays
    nx = length(λ)
    n_data = length(data)
    λ_coherent_flat = Float64[]
    residuals_flat = Float64[]
    weights_flat = Float64[]
    sizehint!(residuals_flat, n_data * nx)
    sizehint!(weights_flat, n_data * nx)

    # Loop over spectra
    for i=1:n_data

        if !isfinite(opt_results[i].fbest)
            continue
        end

        try

            # Best fit pars
            pars = opt_results[i].pbest
        
            # Generate the low res model
            data_λ, model_lr = build(model, pars, data[i])
            
            # Residuals
            residuals_lr = data[i].data.flux .- model_lr

            # Shift to coherent reference frame
            if isnothing(model.star.input_file)
                vel = data[i].header["bc_vel"]
            else
                vel = -1 * pars["vel_star"].value
            end
            λ_coherent_flat = vcat(λ_coherent_flat, maths.doppler_shift_λ(data_λ , vel))
            residuals_flat = vcat(residuals_flat, residuals_lr)
            weights_lr = 1 ./ data[i].data.fluxerr.^2
            weights_flat = vcat(weights_flat, weights_lr)
        
        catch
            @warn "Failed to include [$(data)] in template augmentation!"
        end
    end
    
    # Remove bad vals
    bad = findall(.~isfinite.(λ_coherent_flat) .|| .~isfinite.(residuals_flat) .|| (residuals_flat .== 0) .|| (weights_flat .<= 0) .|| .~isfinite.(weights_flat))
    deleteat!(λ_coherent_flat, bad)
    deleteat!(residuals_flat, bad)
    deleteat!(weights_flat, bad)

    # Sort
    ss = sortperm(λ_coherent_flat)
    λ_coherent_flat .= λ_coherent_flat[ss]
    residuals_flat .= residuals_flat[ss]
    weights_flat .= weights_flat[ss]

    # Knots
    data_λ0 = build(model.λsolution, data[1], opt_results[1].pbest, model.sregion)
    δλ = nanmedian(diff(data_λ0)) / augmenter.oversample
    λi, λf = nanminimum(λ_coherent_flat), nanmaximum(λ_coherent_flat)
    knots = [λi+δλ:δλ:λf-δλ;]

    # Fit
    cspline_star = fit_lsqcspline(λ_coherent_flat, residuals_flat, weights_flat, knots)

    # Augment the template
    new_flux = star_flux .+ cspline_star(λ)
    
    # Force the max to be less than given thresh.
    bad = findall(new_flux .> 1)
    new_flux[bad] .= 1

    # Update the template
    model.templates["star"] .= new_flux

    # Return new star
    return new_flux

end