using PyPlot

export plot_spectral_fit, plot_rvs

function plot_spectral_fit(
        data::DataFrame, model::SpectralForwardModel, opt_result::NamedTuple, iteration::Int, output_path::String
    )

    # Parameters
    params = opt_result.pbest

    # Disable pygui
    pygui(false)

    # Make figures
    fig, axarr = plt.subplots(figsize=(12, 6), nrows=2, ncols=1, sharex=true, sharey=false, squeeze=true, dpi=150)

    # Build the model
    data_λ, model_spec, _ = build(model, params, data)

    # Plot data
    axarr[1].plot(data_λ, data.spec, color=COLORS_GADFLY_HEX[1], lw=1.5, label="Data")
    
    # The residuals for this iteration
    residuals = data.spec .- model_spec

    # Left and right padding
    good = findall(@. isfinite(residuals) && isfinite(data_λ))
    λi, λf = data_λ[good[1]], data_λ[good[end]]
    pad = 0.005 * (maximum(data_λ[good]) - minimum(data_λ[good]))
    
    # Plot model
    axarr[1].plot(data_λ[good], model_spec[good], color=COLORS_GADFLY_HEX[3], lw=1.5, label="Model")

    # Zero lines
    axarr[1].plot(data_λ, zeros(size(data_λ)), color=(66/255, 16/255, 97/255), lw=0.8, ls=":", alpha=0.5)
    axarr[1].plot(data_λ, fill(-0.2, size(data_λ)), color=(66/255, 16/255, 97/255), lw=0.8, ls=":", alpha=0.5)
    axarr[2].plot(data_λ, zeros(size(data_λ)), color=(66/255, 16/255, 97/255), lw=0.8, ls=":", alpha=0.5)

    # Residuals
    axarr[1].plot(data_λ, residuals, color=COLORS_GADFLY_HEX[2], lw=1.5, label="Residuals")
    axarr[2].plot(data_λ, residuals, color=COLORS_GADFLY_HEX[2], lw=1.5)
    
    # Star
    if !isnothing(model.star)
        star_spec = build(model.star, model.templates, params, data)
        if !isnothing(model.lsf)
            star_spec .= convolve_spectrum(model.lsf, star_spec, model.templates, params, data)[1]
            axarr[1].plot(model.templates["λ"], star_spec .- 1.2, lw=1.5, color=(0,0,0,0.8), label="Star")
        end
    end

    # Tellurics
    if !isnothing(model.tellurics)
        tell_spec = build(model.tellurics, model.templates, params, data)
        if !isnothing(model.lsf)
            tell_spec .= convolve_spectrum(model.lsf, tell_spec, model.templates, params, data)[1]
            axarr[1].plot(model.templates["λ"], tell_spec .- 1.2, lw=1.5, color=(137/255, 96/255, 248/255, 0.5), label="Tellurics")
        end
    end
    
    # Gas Cell
    if !isnothing(model.gascell)
        gas_spec = build(model.gascell, model.templates, params, data)
        if !isnothing(model.lsf)
            gas_spec .= convolve_spectrum(model.lsf, gas_spec, model.templates, params, data)[1]
            axarr[1].plot(model.templates["λ"], gas_spec .- 1.2, lw=1.5, color=(129/255, 190/255, 129/255, 0.5), label="Gas Cell")
        end
    end

    # X Lims
    axarr[1].set_xlim(λi - pad, λf + pad)
    axarr[2].set_xlim(λi - pad, λf + pad)

    # Y Lims
    axarr[1].set_ylim(-1.25, 1.2)

    # Y ticks for data/model + templates
    axarr[1].set_yticks(-1:0.5:1)

    # Y Lims for residuals
    residuals_smooth = quantile_filter(residuals, window=3)
    yi, yf = nanminimum(residuals_smooth), nanmaximum(residuals_smooth)
    dy = yf - yi
    yi -= dy / 3
    yf += dy / 3
    yi = max(-1, yi)
    yf = min(1, yf)
    axarr[2].set_ylim(yi, yf)
    
    # Legend
    axarr[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), prop=Dict("weight"=>"bold", "size"=>12))
    
    # X and Y axis labels
    axarr[2].set_xlabel("Wavelength [nm]", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.5, "Norm Spec", fontsize=14, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center")


    # Tick label font size
    # Bold axis tick labels
    axarr[1].tick_params(labelsize=14)
    axarr[2].tick_params(labelsize=14)

    # The title
    fig.suptitle("$(basename(metadata(data, "filename"))), $(replace(model.star.name, "_" => " ")), $(split(output_path, PATHSEP)[end-1]), Iteration $(iteration)", fontsize=14, fontweight="bold")
    
    # Save figure
    fname = output_path * PATHSEP * "Fits" * PATHSEP * "$(splitext(basename(metadata(data, "filename")))[1])_iter$(iteration).png"
    #tight_layout()
    subplots_adjust(left=0.07, bottom=0.15, right=0.87, top=0.9, wspace=nothing, hspace=0.05)
    fig.savefig(fname)
    close(fig)

end


function plot_rvs(
        model::SpectralForwardModel, rvs::Dict,
        iteration::Int, output_path::String;
        t0::Real=2450000,
    )

    pygui(false)
    figure(figsize=(8, 4), dpi=200)
    
    # Individual Forward Model
    errorbar(rvs["bjds"] .- t0, rvs["rvsfwm"][:, iteration] ./ 1E3, yerr=rvs["rvsfwmerr"][:, iteration] ./ 1E3, marker="o", linewidth=0, elinewidth=1, color=(0.1, 0.8, 0.1), alpha=0.7, label="Forward model")


    # Individual XC
    if "rvsxc" ∈ keys(rvs)
        errorbar(rvs["bjds"] .- t0, rvs["rvsxc"][:, iteration] ./ 1E3, yerr=rvs["rvsxcerr"][:, iteration] ./ 1E3, marker="X", linewidth=0, elinewidth=1, color="black", alpha=0.5, label="XC")
    end
    
    # Plot labels
    title("$(replace(model.star.name, "_" => " ")), $(split(output_path, PATHSEP)[end-1]), Iteration $iteration")
    ax = plt.gca()
    ax.ticklabel_format(useOffset=false, style="plain")
    xlabel("BJD - $(t0)")
    ylabel("RV [km/s]")
    legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    # Tight layout
    tight_layout()
    
    # Save
    fname = output_path * "RVs" * PATHSEP * "rvs_iter$(iteration).png"
    savefig(fname)
    plt.close()
end