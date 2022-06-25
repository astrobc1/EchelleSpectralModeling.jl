export compute_rv_content

"""
    compute_rv_content(model::SpectralForwardModel, pars::Parameters, data::SpecData1d; snr=1)
Computes the stellar rv content at each pixel and co-added across all pixels, accounting for the gas cell flux and corresponding calibration errors, as well as the telluric flux.
"""
function compute_rv_content(model::SpectralForwardModel, pars::Parameters, data::SpecData1d; include_gascell_content=true, include_telluric_content=true, snr=1)

    # Data wave grid
    data_λ = build(model.λsolution, data, pars, model.sregion)

    # Model wave grid
    λhr = model.templates["λ"]

    # Star flux on model data wave grid
    star_flux = build(model.star, pars, model.templates)

    # Convolve stellar flux
    if !isnothing(model.lsf)
        kernel = build(model.lsf, pars, model.templates)
        star_flux .= maths.convolve1d(star_flux, kernel)
    end
    
    # Interpolate star flux onto data grid
    star_flux = maths.cspline_interp(λhr, star_flux, data_λ)

    # Gas cell flux on model wave grid for kth observation
    if !isnothing(model.gascell)
        gas_flux = build(model.gascell, pars, model.templates)
        if !isnothing(model.lsf)
            gas_flux .= maths.convolve1d(gas_flux, kernel)
        end
        
        # Interpolate gas cell flux onto data grid
        gas_flux = maths.cspline_interp(λhr, gas_flux, data_λ)
    
    else
        gas_flux = nothing
    end

    # Telluric flux on model wave grid for kth observation
    if !isnothing(model.tellurics)
        tell_flux = build(model.tellurics, pars, model.templates)
        if !isnothing(model.lsf)
            tell_flux .= maths.convolve1d(tell_flux, kernel)
        end

        # Interpolate telluric flux onto data grid
        tell_flux = maths.cspline_interp(λhr, tell_flux, data_λ)
    
    else
        tell_flux = nothing
    end

    # Find good pixels
    good = findall(isfinite.(data_λ) .&& isfinite.(star_flux))

    # Create a spline for the stellar flux to compute derivatives
    cspline_star = maths.CubicSpline(star_flux[good], data_λ[good])

    # Stores rv content for star
    rvc_per_pix_star = fill(NaN, length(data_λ))

    # Create a spline for the gas cell flux to compute derivatives
    if !isnothing(gas_flux)

        # Find good pixels
        good = findall(isfinite.(data_λ) .&& isfinite.(gas_flux))

        cspline_gas = maths.CubicSpline(gas_flux[good], data_λ[good])

        # Stores rv content for gas cell
        rvc_per_pix_gas = fill(NaN, length(data_λ))
    end

    # Loop over pixels
    for i=1:length(data_λ)

        # Skip if this pixel is not used
        if !isfinite(data_λ[i])
            continue
        end

        # Compute stellar flux at this wavelength
        Ai = star_flux[i]

        # Include gas and tell flux
        if !isnothing(gas_flux)
           Ai *= gas_flux[i]
        end
        if !isnothing(tell_flux)
           Ai *= tell_flux[i]
        end

        # Scale to S/N (assumes gain = 1)
        Ai = Ai * snr^2

        # Compute derivative of stellar flux and gas flux
        dAi_dw_star = derivative(cspline_star, data_λ[i])
        if !isnothing(gas_flux)
            dAi_dw_star *= gas_flux[i]
        end
        if !isnothing(tell_flux)
            dAi_dw_star *= tell_flux[i]
        end

        # Make sure slope is finite
        if !isfinite(dAi_dw_star)
            continue
        end

        # Scale to S/N
        dAi_dw_star *= snr^2

        # Compute stellar rv content
        rvc_per_pix_star[i] = SPEED_OF_LIGHT_MPS * sqrt(Ai) / (data_λ[i] * abs(dAi_dw_star))

        # Compute derivative of gas cell flux
        if !isnothing(gas_flux)
            dAi_dw_gas = derivative(cspline_gas, data_λ[i])
            dAi_dw_gas *= star_flux[i]
            
            if !isnothing(tell_flux)
                dAi_dw_gas *= tell_flux[i]
            end

            # Scale to S/N
            dAi_dw_gas *= snr^2

            # Compute gas cell rv content
            rvc_per_pix_gas[i] = SPEED_OF_LIGHT_MPS * sqrt(Ai) / (data_λ[i] * abs(dAi_dw_gas))
        end
    end

    
    # Full RV Content per pixel
    if !isnothing(gas_flux)
        rvc_per_pix = sqrt.(rvc_per_pix_star.^2 .+ rvc_per_pix_gas.^2)
    else
        rvc_per_pix = rvc_per_pix_star
    end

    # Full RV Content
    rvc_tot = nansum(1 ./ rvc_per_pix.^2).^-0.5

    # Return
    return rvc_per_pix, rvc_tot
end