using EchelleBase
using EchelleSpectralModeling
using DataInterpolations

export bin_rvs_single_order

const SPEED_OF_LIGHT_MPS = 299792458.0

####################################
#### CROSS-CORRELATION ROUTINES ####
####################################

# function brute_force_ccf(p0, spectral_model, iter_index, vel_window=400_000)
    
#     # Copy init params
#     pars = copy.deepcopy(p0)
    
#     # Get current star vel
#     v0 = p0[spectral_model.star.par_names[0]].value
    
#     # Make coarse and fine vel grids
#     vel_step_coarse = 200
#     vels_coarse = np.arange(v0 - vel_window / 2, v0 + vel_window / 2, vel_step_coarse)

#     # Stores the rms as a function of velocity
#     rmss_coarse = np.full(vels_coarse.size, dtype=np.float64, fill_value=np.nan)
    
#     # Starting weights are bad pixels
#     weights_init = np.copy(spectral_model.data.mask)

#     # Wavelength grid for the data
#     wave_data = spectral_model.wls.build(pars)

#     # Compute RV info content
#     rvc_per_pix, _ = compute_rv_content(p0, spectral_model, snr=100) # S/N here doesn't matter

#     # Weights are 1 / rv info^2
#     star_weights_init = 1 / rvc_per_pix**2

#     # Data flux
#     data_flux = np.copy(spectral_model.data.flux)
    
#     # Compute RMS for coarse vels
#     for i in range(vels_coarse.size):
        
#         # Set the RV parameter to the current step
#         pars[spectral_model.star.par_names[0]].value = vels_coarse[i]
        
#         # Build the model
#         _, model_lr = spectral_model.build(pars)
        
#         # Shift the stellar weights instead of recomputing the rv content.
#         _, star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_coarse[i], wave_out=wave_data)
        
#         # Final weights
#         weights = weights_init * star_weights_shifted
#         bad = np.where(weights < 0)[0]
#         weights[bad] = 0
#         good = np.where(weights > 0)[0]
#         if good.size == 0:
#             continue
        
#         # Compute the RMS
#         rmss_coarse[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

#     # Extract the best coarse rv
#     M = np.nanargmin(rmss_coarse)
#     xcorr_rv_init = vels_coarse[M]

#     # Determine the uncertainty from the coarse ccf
#     try:
#         n = np.nansum(spectral_model.data.mask)
#         xcorr_rv_stddev, skew = compute_ccf_moments(vels_coarse, rmss_coarse)
#         n_used = np.nansum(spectral_model.data.mask)
#         xcorr_rv_unc = xcorr_rv_stddev / np.sqrt(n_used)
#     except:
#         return np.nan, np.nan, np.nan

#     # Define the fine vels
#     vel_step_fine = 2
#     vel_window_fine = 1000  # For now
#     vels_fine = np.arange(xcorr_rv_init - vel_window_fine / 2, xcorr_rv_init + vel_window_fine / 2, vel_step_fine)
#     rmss_fine = np.full(vels_fine.size, fill_value=np.nan)
    
#     # Now do a finer CCF
#     for i in range(vels_fine.size):
        
#         # Set the RV parameter to the current step
#         pars[spectral_model.star.par_names[0]].value = vels_fine[i]
        
#         # Build the model
#         _, model_lr = spectral_model.build(pars)
        
#         # Shift the stellar weights instead of recomputing the rv content.
#         _, star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_fine[i], wave_out=wave_data)
        
#         # Final weights
#         weights = weights_init * star_weights_shifted
#         bad = np.where(weights < 0)[0]
#         weights[bad] = 0
#         good = np.where(weights > 0)[0]
#         if good.size == 0:
#             continue
        
#         # Compute the RMS
#         rmss_fine[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

#     # Fit (M-2, M-1, ..., M+1, M+2) with parabola to determine true minimum
#     # Extract the best coarse rv
#     M = np.nanargmin(rmss_fine)
#     use = np.arange(M - 2, M + 3, 1).astype(int)
#     try:
#         pfit = np.polyfit(vels_fine[use], rmss_fine[use], 2)
#         xcorr_rv = -0.5 * pfit[1] / pfit[0] + spectral_model.data.bc_vel
#     except:
#         xcorr_rv = np.nan

#     return xcorr_rv, xcorr_rv_unc, skew

# def compute_ccf_moments(vels, rmss):
#     p0 = [1.0, vels[np.nanargmin(rmss)], 5000, 10, 0.1] # amp, mean, sigma, alpha (~skewness), offset
#     bounds = [(0.8, 1.2), (p0[1] - 2000, p0[1] + 2000), (100, 1E5), (-100, 100), (-0.5, 0.5)]
#     opt_result = scipy.optimize.minimize(fit_ccf_skewnorm, x0=p0, bounds=bounds, args=(vels, rmss), method="Nelder-Mead")
#     ccf_stddev = opt_result.x[2]
#     alpha = opt_result.x[3]
#     delta = alpha / np.sqrt(1 + alpha**2)
#     skewness = (4 - np.pi) / 2 * (delta * np.sqrt(2 / np.pi))**3 / (1 - 2 * delta**2 / np.pi)**1.5
#     return ccf_stddev, skewness

#########################
#### RV INFO CONTENT ####
#########################

function compute_rv_content(model::SpectralForwardModel, pars::Parameters; snr=100)

    # Data wave grid
    data_λ = build(model.λsolution, pars)

    # Model wave grid
    λhr = model.templates["λhr"]

    # Star flux on model data wave grid
    star_flux = build(model.star, pars, model.templates)

    # Convolve stellar flux
    if !isnothing(model.lsf)
        kernel = build(model.lsf, pars)
        star_flux = maths.convolve1d(star_flux, kernel)
    end
    
    # Interpolate star flux onto data grid
    star_flux = maths.cspline_interp(λhr, star_flux, data_λ)

    # Gas cell flux on model wave grid for kth observation
    if !isnothing(model.gascell)
        gas_flux = build(model.gascell, pars, model.templates["gascell"], λhr)
        if !isnothing(model.lsf)
            gas_flux = maths.convolve1d(gas_flux, kernel)
        end
        
        # Interpolate gas cell flux onto data grid
        gas_flux = pcmath.cspline_interp(model_wave, gas_flux, data_λ)
    
    else
        gas_flux = nothing
    end

    # Telluric flux on model wave grid for kth observation
    if !isnothing(model.tellurics)
        tell_flux = build(model.tellurics, pars, model.templates["tellurics"], λhr)
        if !isnothing(model.lsf)
            tell_flux = maths.convolve1d(tell_flux, kernel)
        end

        # Interpolate telluric flux onto data grid
        tell_flux = maths.cspline_interp(λhr, tell_flux, data_λ)
    
    else
        tell_flux = nothing
    end

    # Find good pixels
    good = findall(isfinite.(data_λ) .&& isfinite.(star_flux))

    # Create a spline for the stellar flux to compute derivatives
    cspline_star = DataInterpolations.CubicSpline(star_flux[good], data_λ[good])

    # Stores rv content for star
    rvc_per_pix_star = fill(NaN, length(data_λ))

    # Create a spline for the gas cell flux to compute derivatives
    if !isnothing(gas_flux)

        # Find good pixels
        good = findall(isfinite.(data_λ) .&& isfinite.(gas_flux))

        cspline_gas = DataInterpolations.CubicSpline(gas_flux[good], data_λ[good])

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
        dAi_dw_star = derivative(cspline_star)(data_λ[i])
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
            dAi_dw_gas = derivative(cspline_gas)(data_λ[i])
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


#######################
#### CO-ADDING RVS ####
#######################

function combine_relative_rvs(bjds::Vector{Float64}, rvs::Matrix{Float64}, weights::Matrix{Float64}, indices::Vector{Float64})

    # Numbers
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)

    # Align chunks
    rvli, wli = align_chunks(rvs, weights)
    
    # Output arrays
    rvs_single_out = fill(NaN, n_spec)
    unc_single_out = fill(NaN, n_spec)
    t_binned_out = fill(NaN, n_bins)
    rvs_binned_out = fill(NaN, n_bins)
    unc_binned_out = fill(NaN, n_bins)
    bad = findall(.~isfinite.(wli))
    wli[bad] = 0
        
    # Per-observation RVs
    for i=1:n_spec
        rvs_single_out[i] = maths.weighted_mean(rvli[:, i], wli[:, i])
    end
        
    # Per-night RVs
    for i=1:n_bins
        f, l = indices[i]
        rr = collect(Iterator.flatten(rvli[:, f:l]))
        ww = collect(Iterator.flatten(wli[:, f:l]))
        bad = findall(.~isfinite(rr))
        ww[bad] = 0
        rvs_binned_out[i] = maths.weighted_mean(rr, ww)
        unc_binned_out[i] = maths.weighted_stddev(rr, ww)
        t_binned_out[i] = nanmean(bjds[f:l])
    end
    
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
end

function bin_rvs_single_order(rvs::Vector{Float64}, weights::Vector{Float64}, indices::Vector{Float64})

    # The number of spectra and nights
    n_spec = length(rvs)
    n_bins = len(indices)
    
    # Initialize the binned rvs and uncertainties
    rvs_binned = fill(n_bins, NaN)
    unc_binned = fill(n_bins, NaN)
    
    # Bin
    for i=1:n_bins
        f, l = indices[i]
        rr = @view rvs[f:l+1]
        ww = @view weights[f:l+1]
        rvs_binned[i], unc_binned[i] = maths.weighted_combine(rr, ww, yerr, err_type="empirical")
    end
            
    return rvs_binned, unc_binned
end

function align_chunks(rvs::Matrix{Float64}, weights::Matrix{Float64})

    n_chunks, n_spec = size(rvs)
    
    # Determine differences and weights tensors
    rvlij = fill(NaN, (n_chunks, n_spec, n_spec))
    wlij = fill(NaN, (n_chunks, n_spec, n_spec))
    wli = fill(NaN, (n_chunks, n_spec))
    for l=1:n_chunks
        for i=1:n_spec
            wli[l, i] = weights[l, i]
            for j=1:n_spec
                rvlij[l, i, j] = rvs[l, i] - rvs[l, j]
                wlij[l, i, j] = sqrt(weights[l, i] * weights[l, j])
            end
        end
    end

    # Average over differences
    rvli = fill(NaN, (n_chunks, n_spec))
    for l=1:n_chunks
        for i=1:n_spec
            rvli[l, i] = maths.weighted_mean(rvlij[l, i, :], wlij[l, i, :])
        end
    end

    return rvli, wli
end

function bin_jds(jds::Vector{Float64}; sep=0.5, utc_offset=-8)
    
    # Number of spectra
    n_obs_tot = length(jds)

    # Keep track of previous night's last index
    prev_i = 0

    # Calculate mean JD date and number of observations per night for binned
    # Assume that observations are in separate bins if noon passes or if Δt > sep
    jds_binned = Float64[]
    n_obs_binned = Float64[]
    indices_binned = Vector{Float64}[]
    if n_obs_tot == 1
        push!(jds_binned, jds[1])
        push!(n_obs_binned, 1)
        push!(indices_binned, [1, 1])
    else
        for i=1:n_obs_tot-1
            t_noon = ceil(jds[i] + utc_offset / 24) - utc_offset / 24
            if jds[i+1] > t_noon | jds[i+1] - jds[i] > sep
                jd_avg = mean(jds[prev_i:i])
                push!(jds_binned, jd_avg)
                push!(indices_binned, [prev_i, i])
                prev_i = i + 1
            end
        end
        push!(jds_binned, mean(jds[prev_i:end]))
        push!(indices_binned, [prev_i, n_obs_tot - 1])
    end

    return jds_binned, indices_binned
end