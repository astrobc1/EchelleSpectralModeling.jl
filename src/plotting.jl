using PyPlot
using NaNStatistics

using EchelleBase
using EchelleSpectralModeling

export plot_spectrum_fit, plot_rvs

"""
    plot_spectrum_fit(data::SpecData1d, model::SpectralForwardModel, pars::Parameters, iteration::Int, output_path::String)
Plots the data and model.
"""
function plot_spectrum_fit(data::SpecData1d, model::SpectralForwardModel, pars::Parameters, iteration::Int, output_path::String)

    # Build the model
    data_λ, model_flux = build(model, pars, data)

    # The high res wave grid
    λhr = model.templates["λ"]
    
    # The residuals for this iteration
    residuals = data.data.flux .- model_flux

    # Copy the mask
    mask = copy(data.data.mask)
    
    # Ensure known bad pixels are nans in the residuals
    residuals[mask .== 0] .= NaN
    
    # Left and right padding
    good = findall(isfinite.(data_λ) .&& (mask .== 1))
    pad = 0.01 * (maximum(data_λ[good]) - minimum(data_λ[good]))

    pygui(false)
    
    # Plot data
    figure(figsize=(10, 3), dpi=200)
    plot(data_λ, data.data.flux, color=COLORS_HEX_GADFLY[1], lw=1, label="Data")
    
    # Plot model
    plot(data_λ, model_flux, color=COLORS_HEX_GADFLY[3], lw=1, label="Model")
    
    # Zero line and -0.2 line
    z = zeros(length(data_λ))
    plot(data_λ, z, color=(89/255, 23/255, 130/255), lw=0.8, ls=":")
    plot(data_λ, z .- 0.2, color=(89/255, 23/255, 130/255), lw=0.8, ls=":")
    
    # Residuals
    plot(data_λ, residuals, color=COLORS_HEX_GADFLY[2], lw=1, label="Residuals")
    
    # LSF
    if !isnothing(model.lsf)
        kernel = build(model.lsf, pars, model.templates)
    end
    
    # Star
    if !isnothing(model.star)
        
        # Initial star
        # if !isnothing(model.star.input_file) && iteration != 1
        #     star_flux = build(model.star, pars, model.templates)
        #     star_flux = maths.doppler_shift_flux(λhr, star_flux, pars["vel_star"].value)
        #     if !isnothing(model.lsf)
        #         star_flux .= maths.convolve1d(star_flux, kernel)
        #     end
        #     plot(λhr, star_flux .- 1.2, label="Initial Star", lw=1, color=(235/255, 98/255, 52/255, 0.4))
        # end
        
        # Current star
        star_flux = build(model.star, pars, model.templates)
        if !isnothing(model.lsf)
            star_flux .= maths.convolve1d(star_flux, kernel)
        end
        plot(λhr, star_flux .- 1.2, label="Current Star", lw=1, color=(0,0,0,0.8))
    end
    
    # Tellurics
    if !isnothing(model.tellurics)
        tell_flux = build(model.tellurics, pars, model.templates)
        if !isnothing(model.lsf)
            tell_flux .= maths.convolve1d(tell_flux, kernel)
        end
        plot(λhr, tell_flux .- 1.2, label="Tellurics", lw=1, color=(137/255, 96/255, 248/255, 0.5))
    end
    
    # Gas Cell
    if !isnothing(model.gascell)
        gas_flux = build(model.gascell, pars, model.templates)
        if !isnothing(model.lsf)
            gas_flux .= maths.convolve1d(gas_flux, kernel)
        end
        plot(λhr, gas_flux .- 1.2, label="Gas Cell", lw=1, color=(129/255, 190/255, 129/255, 0.5))
    end

    # X and Y limits
    xlim(model.sregion.λmin - pad, model.sregion.λmax + pad)
    ylim(-1.3, 1.2)
        
    # Legend
    legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    # X and Y axis labels
    xlabel("Wavelength [nm]", fontsize=10)
    ylabel("Norm. flux", fontsize=10)

    # The title
    title("$(replace(model.star.star_name, "_" => " ")), $(label(model.sregion)), Iteration $(iteration)", fontsize=10)
    
    # Save figure
    fname = output_path * label(model.sregion) * PATHSEP * "Fits" * PATHSEP * "$(splitext(basename(data.fname))[1])_$(label(model.sregion))_iter$(iteration).png"
    tight_layout()
    savefig(fname)
    plt.close()

end

"""
    plot_rvs(ensemble::IterativeSpectralRVEnsembleProblem, rvs::Dict, iteration::Int, output_path::String; time_offset::Real=2450000)
Plots the rvs for a given iteration and saves the figure.
"""
function plot_rvs(ensemble::IterativeSpectralRVEnsembleProblem, rvs::Dict, iteration::Int, output_path::String; time_offset::Real=2450000)

    pygui(false)

    # Plot the rvs, binned rvs, xcorr rvs, xcorr binned rvs
    figure(figsize=(8, 4), dpi=200)
    
    # Individual Forward Model
    plot(rvs["bjds"] .- time_offset, rvs["rvsfwm"][:, iteration] ./ 1E3, marker="o", linewidth=0, color=(0.1, 0.8, 0.1), alpha=0.7, label="Forward model")

    # Individual XC
    if "rvsxc" ∈ keys(rvs)
        errorbar(rvs["bjds"] .- time_offset, rvs["rvsxc"][:, iteration] ./ 1E3, yerr=rvs["rvsxcerr"][:, iteration] ./ 1E3, marker="X", linewidth=0, elinewidth=1, color="black", alpha=0.5, label="XC")
    end
    
    # Plot labels
    title("$(replace(ensemble.model.star.star_name, "_" => " ")), $(label(ensemble.model.sregion)), Iteration $iteration")
    ax = plt.gca()
    ax.ticklabel_format(useOffset=false, style="plain")
    plt.xlabel("BJDTDB - $(time_offset)")
    plt.ylabel("RV [km/s]")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    fname = output_path * label(ensemble.model.sregion) * PATHSEP * "RVs" * PATHSEP * "rvs_$(label(ensemble.model.sregion))_iter$(iteration).png"
    savefig(fname)
    plt.close()
end
