

function augment_stellar_template!(
        model::SpectralForwardModel, data::Vector{DataFrame}, opt_results::Vector;
        smooth_width::Int=0,
        max_val::Real=1.0,
        remove_contaminants::Bool=false
    )

    # Unpack the current high res stellar template
    star_spec = isnothing(model.star.template) ? ones(length(model.λ)) : model.star.template

    # Get coherent residuals
    λ_coherent, residuals_mat, weights_mat = get_coherent_residuals(model, data, opt_results; upsample=true, remove_contaminants)

    # Sync
    bad = findall(@. ~isfinite(residuals_mat) || (weights_mat <= 0) || ~isfinite(weights_mat))
    residuals_mat[bad] .= NaN
    weights_mat[bad] .= 0

    # Combine
    residuals_mean = fill(NaN, size(residuals_mat, 1))
    residuals_err = fill(NaN, size(residuals_mat, 1))
    for ix in eachindex(residuals_mean)
        @views ww, rr1 = weights_mat[ix, :], residuals_mat[ix, :]
        if nansum(ww) > 0
            good = findall(@. (ww > 0) && isfinite(ww))
            n_good = length(good)
            if n_good == 1
                residuals_mean[ix] = rr1[good[1]]
            elseif n_good > 1
                residuals_mean[ix] = robust_mean(rr1, ww, nσ=4)
                residuals_err[ix] = robust_stddev(rr1, ww, nσ=4)
            end
        end
    end

    # Sanity
    bad = findall(.~isfinite.(residuals_mean))
    residuals_mean[bad] .= NaN
    residuals_err[bad] .= NaN

    # Smooth, convert to model pixels
    if !isnothing(smooth_width) && smooth_width > 0
        width = Int(round(smooth_width))
        if !isodd(width)
            width += 1
        end
        good = findall(isfinite.(residuals_mean))
        residuals_mean[good] .= @views savitzky_golay(residuals_mean[good], width, 3).y
        good = findall(isfinite.(residuals_err))
        residuals_err[good] .= @views savitzky_golay(residuals_err[good], width, 3).y
    end

    # Check after smoothing
    bad = findall(.~isfinite.(residuals_mean))
    residuals_mean[bad] .= NaN
    residuals_err[bad] .= NaN

    # Augment the template
    star_spec_new = star_spec .+ residuals_mean
    
    # Clamp
    clamp!(star_spec_new, 1E-5, max_val)

    # Update
    model.star.template = star_spec_new

end


function get_coherent_residuals(
        model::SpectralForwardModel, data::Vector{DataFrame}, opt_results::Vector;
        upsample::Bool=true, remove_contaminants::Bool=false,
    )

    # Coherent wavelength grid
    if upsample
        λ_coherent = model.λ
    else
        k, _ = get_best_rms(opt_results)
        λ_coherent = build(model.λsolution, opt_results[k].pbest, data[k])
    end

    # Coherent wavelength grid size
    nx = length(λ_coherent)
    n_spec = length(data)

    # Storage arrays
    residuals_mat = zeros(nx, n_spec)
    weights_mat = zeros(nx, n_spec)

    # Median RV
    if model.star.from_flat
        rvμ = 0
    else
        rvμ = nanmedian([!isnothing(r) ? r.pbest["vel_star"] : NaN for r in opt_results])
    end

    # Loop over spectra
    for i in eachindex(data)

        # Skip if bad
        if isnothing(opt_results[i])
            continue
        end

        # Best fit params
        params = opt_results[i].pbest

        # Data wls
        data_λ, model_lr, _ = build(model, params, data[i])

        # Generate residuals
        if remove_contaminants
            _, model_lr, _ = build(model, params, data[i])
            residuals_lr = data[i].spec ./ model_lr .- 1
        else
            residuals_lr = data[i].spec .- model_lr
        end
        
        # Good vals
        good = findall(isfinite.(residuals_lr))

        # Skip if no good vals
        if length(good) == 0
            continue
        end

        # Doppler shift low res wavelength grid
        bc_vel = metadata(data[i], "bc_vel")
        βstar = rvμ / SPEED_OF_LIGHT_MPS
        βbc = bc_vel / SPEED_OF_LIGHT_MPS
        tstar = sqrt((1 + βstar) / (1 - βstar))
        tbc = sqrt((1 + βbc) / (1 - βbc))
        λ_star_rest_lr = @. data_λ * tbc / tstar

        # Interpolate onto High res grid
        residuals_mat[:, i] .= interp1d(λ_star_rest_lr, residuals_lr, λ_coherent, extrapolate=true)

        # Initialize a mask
        mask = ones(length(data[i].spec))
        bad = findall(@. ~isfinite(data[i].spec))
        mask[bad] .= 0

        # Weights
        snrs = quantile_filter(data[i].spec ./ data[i].specerr, window=5)
        weights_lr = mask .* snrs.^2

        # Nan -> 0
        weights_lr[.~isfinite.(weights_lr)] .= 0

        # Interpolate onto high res grid
        weights_mat[:, i] .= LinearInterpolation(λ_star_rest_lr, weights_lr, extrapolate=false)(λ_coherent)
        
    end

    # Fix negative weights
    clamp!(weights_mat, 0, Inf)

    # Return
    return λ_coherent, residuals_mat, weights_mat

end


function get_best_rms(opt_results::Vector)
    rmss = [!isnothing(r) ? r.rms : NaN for r in opt_results]
    k = nanargmin(rmss)
    return k, rmss[k]
end