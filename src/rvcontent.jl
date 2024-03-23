export compute_rv_content

function compute_rv_content(model::SpectralForwardModel, params::Parameters; snr::Real=1)

    # Data wave grid
    data_λ = build(model.λsolution, model.templates, params, data)

    # Need mean of data spec for scaling
    data_spec_mean = nanmean(data["spec"])

    # Model wave grid
    λhr = model.templates["λ"]

    # Star spec on model data wave grid
    star_spec = build(model.star, model.templates, params, data)

    # Convolve stellar spec
    if !isnothing(model.lsf)
        star_spec .= convolve_spectrum(model.lsf, model.templates, params, star_spec, data)
    end
    
    # Interpolate star spec onto data grid
    star_spec = interp1d(λhr, star_spec, data_λ, left=NaN, right=NaN, method="cubic")

    # Gas cell spec on model wave grid for kth observation
    if !isnothing(model.gascell)
        gas_spec = build(model.gascell, model.templates, params, data)
        if !isnothing(model.lsf)
            gas_spec .= convolve_spectrum(model.lsf, model.templates, params, gas_spec, data)
        end
        
        # Interpolate gas cell spec onto data grid
        gas_spec = interp1d(λhr, gas_spec, data_λ, left=:const, right=:const, method="cubic")
    
    else
        gas_spec = nothing
    end

    # Telluric spec on model wave grid for kth observation
    if !isnothing(model.tellurics)
        tell_spec = build(model.tellurics, model.templates, params, data)
        if !isnothing(model.lsf)
            tell_spec .= convolve_spectrum(model.lsf, model.templates, params, tell_spec, data)
        end

        # Interpolate telluric spec onto data grid
        tell_spec = interp1d(λhr, tell_spec, data_λ, left=:const, right=:const, method="cubic")
    
    else
        tell_spec = nothing
    end

    # Find good pixels
    good = findall(@. isfinite(data_λ) && isfinite(star_spec))

    # Create a spline for the stellar spec to compute derivatives
    cspline_star = DataInterpolations.CubicSpline(star_spec[good], data_λ[good])

    # Stores rv content for star
    rvc_per_pix_star = fill(NaN, length(data_λ))

    # Create a spline for the gas cell spec to compute derivatives
    if !isnothing(gas_spec)

        # Find good pixels
        good = findall(@. isfinite(data_λ) && isfinite(gas_spec))

        cspline_gas = DataInterpolations.CubicSpline(gas_spec[good], data_λ[good])

        # Stores rv content for gas cell
        rvc_per_pix_gas = fill(NaN, length(data_λ))
        
    end

    # Continuum
    if !isnothing(model.continuum)
        continuum = build(model.continuum, model.templates, params, data)
        continuum = interp1d(model.templates["λ"], continuum, data_λ, left=NaN, right=NaN, method="cubic")
    end

    # Loop over pixels
    for i ∈ eachindex(data_λ)

        if !isfinite(data_λ[i])
            continue
        end

        # Skip if this pixel is not used
        if model.sregion.mask_mode == :λ
            if !(model.sregion.λrange[1] < data_λ[i] < model.sregion.λrange[2])
                continue
            end
        else
            if !(data_λ[model.sregion.pixrange[1]] < data_λ[i] < data_λ[model.sregion.pixrange[2]])
                continue
            end
        end

        # Stellar flux and derivative
        Ai = star_spec[i]
        dAi_dw_star = DataInterpolations.derivative(cspline_star, data_λ[i])

        # Gas, tell, and continuum contribution to flux
        if !isnothing(model.gascell)
           Ai *= gas_spec[i]
           dAi_dw_star *= gas_spec[i]
        end
        if !isnothing(model.tellurics)
           Ai *= tell_spec[i]
           dAi_dw_star *= tell_spec[i]
        end
        if !isnothing(model.continuum)
            Ai *= continuum[i]
            dAi_dw_star *= continuum[i]
        end

        # Make sure slope is finite
        if !isfinite(dAi_dw_star) || !isfinite(Ai)
            continue
        end

        # Scale to S/N
        if data_mean_scale
            Ai /= data_spec_mean
            dAi_dw_star /= data_spec_mean
        end
        Ai *= snr^2
        dAi_dw_star *= snr^2

        # Compute stellar rv content
        rvc_per_pix_star[i] = SPEED_OF_LIGHT_MPS * sqrt(Ai) / (data_λ[i] * abs(dAi_dw_star))

        # Now do same for gas cell
        if !isnothing(model.gascell)
            dAi_dw_gas = DataInterpolations.derivative(cspline_gas, data_λ[i])
            dAi_dw_gas *= star_spec[i]
            
            if !isnothing(model.tellurics)
                dAi_dw_gas *= tell_spec[i]
            end

            # Continuum
            if !isnothing(model.continuum)
                dAi_dw_gas *= continuum[i]
            end

            # Rescaling
            if data_mean_scale
                dAi_dw_gas /= data_spec_mean
            end

            # Scale to S/N
            dAi_dw_gas *= snr^2

            # Compute gas cell rv content
            rvc_per_pix_gas[i] = SPEED_OF_LIGHT_MPS * sqrt(Ai) / (data_λ[i] * abs(dAi_dw_gas))
        end
    end

    
    # Full RV Content per pixel
    if !isnothing(model.gascell)
        rvc_per_pix = sqrt.(rvc_per_pix_star.^2 .+ rvc_per_pix_gas.^2)
    else
        rvc_per_pix = rvc_per_pix_star
    end

    # Full RV Content
    rvc_tot = nansum(1 ./ rvc_per_pix.^2).^-0.5

    # Return
    return rvc_per_pix, rvc_tot
end