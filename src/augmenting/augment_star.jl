using EchelleBase
using EchelleSpectralModeling
using NaNStatistics
using Infiltrator

export augment_star!

"""
    augment_star!(ensemble::IterativeSpectralRVEnsembleProblem, opt_results)
Augment the stellar template variable by computing the weighted median of the residuals in the barycentric frame. Weights are proportional to 1 / rms^2.
"""
function augment_star!(ensemble::IterativeSpectralRVEnsembleProblem, opt_results)

    # Unpack the current stellar template
    star_λ = ensemble.model.templates["λ"]
    star_flux = copy(ensemble.model.templates["star"])
    
    # Storage arrays
    nx = length(star_λ)
    residuals = zeros(nx, length(ensemble))
    weights = zeros(nx, length(ensemble))

    # Loop over spectra
    for i=1:length(ensemble)

        if !isfinite(opt_results[i].fbest)
            continue
        end

        try

            # Best fit pars
            pars = opt_results[i].pbest
        
            # Generate the low res model
            data_λ, model_lr = build(ensemble.model, pars, ensemble.data[i])
            
            # Residuals
            residuals_lr = ensemble.data[i].data.flux .- model_lr

            # Shift to a coherent frame
            if isnothing(ensemble.model.star.input_file)
                vel = ensemble.data[i].header["bc_vel"]
            else
                vel = -1 * pars["vel_star"].value
            end
            λ_star_rest = maths.doppler_shift_λ(data_λ , vel)
            residuals[:, i] .= maths.cspline_interp(λ_star_rest, residuals_lr, star_λ)
            good = findall(isfinite.(residuals_lr) .&& (ensemble.data[i].data.mask .> 0))
            if length(good) == 0
                continue
            end
            rms = (nansum(residuals_lr[good].^2) / length(good))^.5
            weights_lr = ensemble.data[i].data.mask ./ rms^2
            bad = findall(.~isfinite.(weights_lr))
            weights_lr[bad] .= 0
            
            # Interpolate to a high res grid
            weights_hr = maths.lin_interp(λ_star_rest, weights_lr, star_λ)
            bad = findall((weights_hr .< 0) .|| .~isfinite.(weights_hr))
            weights_hr[bad] .= 0
            weights[:, i] .= weights_hr
        
        catch
            nothing
        end

    end
    
    # Sync
    bad = findall(.~isfinite.(residuals) .|| (weights .<= 0))
    residuals[bad] .= NaN
    weights[bad] .= 0

    # Combine
    residuals_median = zeros(nx)
    for ix=1:nx
        @views ww, rr = weights[ix, :], residuals[ix, :]
        if maths.nansum(ww) > 0
            good = findall((ww .> 0) .&& isfinite.(ww))
            if length(good) == 0
                residuals_median[ix] = 0
            elseif length(good) == 1
                residuals_median[ix] = rr[good[1]]
            else
                residuals_median[ix] = maths.weighted_median(rr, w=ww)
            end
        else
            residuals_median[ix] = 0
        end
    end

    # Change any nans to zero just in case
    bad = findall(.~isfinite.(residuals_median))
    residuals_median[bad] .= 0

    # Augment the template
    new_flux = star_flux .+ residuals_median
    
    # Force the max to be less than given thresh.
    bad = findall(new_flux .> 1)
    new_flux[bad] .= 1

    # Update the template
    ensemble.model.templates["star"] .= new_flux

end