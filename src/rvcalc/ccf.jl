using CurveFitParameters
using EchelleBase
using EchelleSpectralModeling

using Infiltrator

"""
    brute_force_ccf(model::SpectralForwardModel, data::SpecData1d, p0::Parameters; vel_window=400_000, vel_step=10)
Calculates a brute force ccf (really a reduced chi-square curve) by stepping through velocity space.
"""
function brute_force_ccf(model::SpectralForwardModel, data::SpecData1d, p0::Parameters; vel_window_coarse=200_000, vel_step_coarse=100, vel_step_fine=10, vel_window_fine=1000)
    
    # Copy init params
    pars = deepcopy(p0)
    
    # Get nominal star vel
    v0 = p0["vel_star"].value
    
    # Make coarse and fine vel grids
    vels_coarse = (v0 - vel_window_coarse / 2):vel_step_coarse:(v0 + vel_window_coarse / 2)

    # Stores the rms as a function of velocity
    redχ2_coarse = fill(NaN, length(vels_coarse))

    # Wavelength grid for the data
    λ_data = build(model.λsolution, data, pars, model.sregion)

    # Data flux, error, mask
    data_flux = data.data.flux
    data_mask = data.data.mask
    data_fluxerr = data.data.fluxerr

    # Dof
    ν = sum(data_mask) - num_varied(pars)
    
    # Compute RMS for coarse vels
    for i=1:length(vels_coarse)

        # Vel
        vel = vels_coarse[i]
        
        # Set the RV parameter to the current step
        pars["vel_star"].value = vel
        
        # Build the model
        _, model_lr = build(model, pars, data)
        
        # Compute the RMS
        redχ2_coarse[i] = maths.redχ2loss(data_flux .- model_lr, data_fluxerr, data_mask; flag_worst=30, remove_edges=6, ν=ν)

    end

    # Extract the best coarse rv
    M = maths.nanargminimum(redχ2_coarse)
    xcorr_rv_init = vels_coarse[M]

    # Define the fine vels
    vels_fine = (xcorr_rv_init - vel_window_fine / 2):vel_step_fine:(xcorr_rv_init + vel_window_fine / 2)
    redχ2_fine = fill(NaN, length(vels_fine))
    
    # Now do a finer CCF
    for i=1:length(vels_fine)
        
        # Vel
        vel = vels_fine[i]
        
        # Set the RV parameter to the current step
        pars["vel_star"].value = vel
        
        # Build the model
        _, model_lr = build(model, pars, data)
        
        # Compute the RMS
        redχ2_fine[i] = maths.redχ2loss(data_flux .- model_lr, data_fluxerr, data_mask; flag_worst=30, remove_edges=6, ν=ν)
    end

    
    # Extract the best fine rv
    M = maths.nanargminimum(redχ2_fine)

    # Fit with Parabola (Eqs. 10 and 11 from SERVAL https://arxiv.org/abs/1710.10114)
    xcorr_rv, xcorr_rv_unc = NaN, NaN
    try
        xcorr_rv = vels_fine[M] - (vel_step_fine / 2) * (redχ2_fine[M+1] - redχ2_fine[M-1]) / (redχ2_fine[M-1] - 2 * redχ2_fine[M] + redχ2_fine[M+1])
        n_good = sum(data_mask)
        xcorr_rv_unc = sqrt((2 * vel_step_fine^2) / (redχ2_fine[M-1] - 2 * redχ2_fine[M] + redχ2_fine[M+1])) / sqrt(n_good)
    catch
        xcorr_rv = NaN
        xcorr_rv_unc = NaN
    end

    # Return rv and error
    return xcorr_rv, xcorr_rv_unc

end