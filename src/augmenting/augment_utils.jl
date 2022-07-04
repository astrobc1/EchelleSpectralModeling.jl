using CurveFitParameters
using EchelleBase

export estimate_initial_stellar_template

using Infiltrator

"""
    estimate_initial_stellar_template(model::SpectralForwardModel, data::Vector{SpecData1d{S}}, p0s::Vector{Parameters}; continuum_poly_deg::Union{Int, Nothing}=nothing, continuum_med_filter_width::Int=101) where{S}
Estimates the stellar template from the data itself. This works best when the spectrum is dominated by stellar features, telluric features are sparse, and **there is no gas cell**. The wavelength grid should also be known a priori. The template is estimated by first dividing out the convolved telluric model (scaled according to the airmass of the observation) as well as an optional estimate to the continuum. The resulting modified observations are then shifted to the barycenter frame and a median reduction provides an estimate of the initial stellar template. This template is then upsampled onto the the high-resolution model grid.
- `continuum_poly_deg_estimate::Bool`: If this value is not `nothing`, a polynomial of degree `continuum_poly_deg_estimate` will be estimated and removed from each spectrum individually before estimating the initial stellar template.
"""
function estimate_initial_stellar_template(model::SpectralForwardModel, data::Vector{SpecData1d{S}}, p0s::Vector{Parameters}; continuum_poly_deg::Union{Int, Nothing}=nothing, continuum_med_filter_width::Int=101) where{S}

    # Numbers
    nx = length(data[1].data.flux)
    n_data = length(data)

    # Store modified data on coherent grid
    data_corrected = fill(NaN, (nx, n_data))

    λ = model.templates["λ"]

    # LSF kernel
    if !isnothing(model.lsf)
        kernel = build(model.lsf, p0s[1], model.templates)
    else
        kernel = nothing
    end

    # Base telluric model
    if !isnothing(model.tellurics)
        tell_flux = build(model.tellurics, p0s[1], model.templates)
        if !isnothing(kernel)
            tell_flux .= maths.convolve1d(tell_flux, kernel)
        end
    else
        tell_flux = nothing
    end

    # Default wavelength solution
    data_λ0 = build(model.λsolution, data[1], p0s[1], model.sregion)

    # Loop over data
    for i=1:n_data

        # Correct telluric flux in lab frame
        if !isnothing(tell_flux)
            data_λ = build(model.λsolution, data[i], p0s[i], model.sregion)
            _tell_flux = maths.cspline_interp(λ, tell_flux, data_λ)
            try
                _tell_flux .^= parse_airmass(data[i])
            catch
                nothing
            end
            vel = data[i].header["bc_vel"]
            data_λ_shifted = maths.doppler_shift_λ(data_λ, vel)
            data_corrected[:, i] .= maths.lin_interp(data_λ_shifted, data[i].data.flux ./ _tell_flux, data_λ0)
        end

         # Remove continuum
        if !isnothing(continuum_poly_deg)
            data_corrected[:, i] ./= Continuum.estimate_continuum(1:nx, data_corrected[:, i], med_filter_width=continuum_med_filter_width, deg=continuum_poly_deg).(1:nx)
        end
        data_corrected[:, i] .= maths.median_filter1d(data_corrected[:, i], 3)
    end

    # Median
    star_flux = nanmedian(data_corrected, dim=2)

    # Upsample
    star_flux = maths.lin_interp(data_λ0, star_flux, λ)

    # Smooth
    smooth_width = 7 * nanmedian(diff(data_λ0))
    star_flux .= maths.poly_filter(λ, star_flux, width=smooth_width, deg=3)

    # Return
    return star_flux

end


