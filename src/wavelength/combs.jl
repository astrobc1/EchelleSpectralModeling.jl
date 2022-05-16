using EchelleBase
using NaNStatistics
using Polynomials
using Peaks
using Infiltrator
using PyPlot
using ModelingToolkit, GalacticOptim, Optim

function estimate_peak_spacing(xi, xf, λi, λf, λ_estimate, ν0, Δν)
    integers, lfc_centers_λ_theoretical = gen_theoretical_peaks(λi, λf, ν0, Δν)
    xarr = [1:length(λ_estimate);]
    peak_separations = Float64[]
    xx = Float64[]
    for i=1:length(lfc_centers_λ_theoretical)-1
        k1 = argmin(abs.(lfc_centers_λ_theoretical[i] .- λ_estimate))
        k2 = argmin(abs.(lfc_centers_λ_theoretical[i+1] .- λ_estimate))
        push!(peak_separations, abs(xarr[k1] - xarr[k2]))
        push!(xx, (xarr[k1] + xarr[k2]) / 2)
    end
    pfit = Polynomials.fit(xx, peak_separations, 1)
    return pfit
end

function get_peaks(λ_estimate, lfc_flux, ν0, Δν, xrange; σ_guess=[0.2, 1.4, 3.0], μ_bounds=[-1, 1])

    # Generate theoretical LFC peaks
    good = findall(isfinite.(λ_estimate) .&& isfinite.(lfc_flux))
    xi, xf = minimum(good), maximum(good)
    λi, λf = λ_estimate[xi], λ_estimate[xf]
    lfc_peak_integers, lfc_centers_λ_theoretical = gen_theoretical_peaks(λi - 10, λf + 10, ν0, Δν)
    peak_spacing = estimate_peak_spacing(xi, xf, λi, λf, λ_estimate, ν0, Δν)
    min_peak_spacing = min(peak_spacing(xi), peak_spacing(xf))
    
    # Number of pixels
    nx = length(λ_estimate)
    xarr = [1:nx;]
    pfit_estimate = Polynomials.fit(xarr[good], λ_estimate[good], 4)

    # Background
    background = estimate_background(λ_estimate, lfc_flux, ν0, Δν)
    lfc_flux_no_bg = lfc_flux .- background
    lfc_peak_max = maths.weighted_median(lfc_flux_no_bg, p=0.99)
    
    # Continuum
    continuum = estimate_continuum(λ_estimate, lfc_flux .- background, ν0, Δν)
    lfc_flux_norm = (lfc_flux .- background) ./ continuum

    # Estimate peaks in pixel space (just indices)
    peaks, _ = findmaxima(lfc_flux_norm[xrange[1]:xrange[2]], Int(round(0.8*min_peak_spacing)), strict=false)
    peaks, _ = peakproms(peaks, lfc_flux_norm[xrange[1]:xrange[2]]; strict=false, minprom=0.75, maxprom=nothing)
    peaks .+= xrange[1] .- 1
    sort!(peaks)
    peaks = peaks[2:end-1]

    # Only consider peaks with enough flux
    good_peaks_int = Int64[]
    for peak ∈ peaks
        if lfc_flux_no_bg[peak] >= 0.1 * lfc_peak_max
            push!(good_peaks_int, peak)
        end
    end

    # Further refine peaks based on centroids
    good_peaks = zeros(length(good_peaks_int))
    for i=1:length(good_peaks_int)
        use = findall((xarr .>= floor(good_peaks_int[i] - peak_spacing(good_peaks_int[i]) / 2)) .&& (xarr .< ceil(good_peaks_int[i] + peak_spacing(good_peaks_int[i]) / 2)))
        xx, yy = xarr[use], lfc_flux_no_bg[use]
        good_peaks[i] = maths.weighted_mean(xx, yy)
    end

    # Once more
    for i=1:length(good_peaks)
        use = findall((xarr .>= floor(good_peaks[i] - peak_spacing(good_peaks_int[i]) / 2)) .&& (xarr .< ceil(good_peaks_int[i] + peak_spacing(good_peaks[i]) / 2)))
        xx, yy = xarr[use], lfc_flux_no_bg[use]
        good_peaks[i] = maths.weighted_mean(xx, yy)
    end

    # Fit each peak with a Gaussian
    lfc_centers_pix = fill(NaN, length(good_peaks))
    amplitudes = fill(NaN, length(good_peaks))
    rms = fill(NaN, length(good_peaks))
    σs = fill(NaN, length(good_peaks))
    offsets = fill(NaN, length(good_peaks))
    for i=1:length(good_peaks)

        # Region to consider
        use = findall((xarr .>= floor(good_peaks[i] - peak_spacing(good_peaks[i]) / 2)) .&& (xarr .< ceil(good_peaks[i] + peak_spacing(good_peaks[i]) / 2)))

        # Crop data
        xx, yy = xarr[use], lfc_flux_no_bg[use]

        # Normalize lfc flux to max
        yy .-= nanminimum(yy)
        yy ./= nanmaximum(yy)
        
        # System
        @variables A μ σ B
        @parameters x[1:length(xx)] y[1:length(xx)]
        loss = begin
            model = maths.gauss(collect(x), A, μ, σ) .+ B
            sqrt(sum((collect(y) .- model).^2) / length(x))
        end
        @named sys = OptimizationSystem(loss, [A, μ, σ, B], [x, y])

        # Pars and bounds
        u0 = [A => 1.0, μ => good_peaks[i], σ => σ_guess[2], B => 0.1]
        lb = [A => 0.7, μ => good_peaks[i] + μ_bounds[1], σ => σ_guess[1], B => -0.5]
        ub = [A => 1.3, μ => good_peaks[i] + μ_bounds[2], σ => σ_guess[3], B => 0.5]
        p = [x => xx, y => yy]

        prob = OptimizationProblem(sys, u0, p, grad=false, hess=false)

        # Fit
        ubest = solve(prob, NelderMead(), maxiters=1000)

        # Results
        amplitudes[i] = ubest[1]
        lfc_centers_pix[i] = ubest[2]
        σs[i] = ubest[3]
        offsets[i] = ubest[4]
        rms[i] = maths.rmsloss(maths.gauss(xx, amplitudes[i], lfc_centers_pix[i], σs[i]) .+ offsets[i], yy)

        #@infiltrate
        #begin
        #    plot(xx, yy);
        #    plot(xx, maths.gauss(xx, amplitudes[i], lfc_centers_pix[i], σs[i]) .+ offsets[i]);
        #end
    end

    # Determine which LFC spot matches each peak
    lfc_centers_λ = Float64[]
    peak_integers = Float64[]
    for i=1:length(lfc_centers_pix)
        diffs = abs.(pfit_estimate(lfc_centers_pix[i]) .- lfc_centers_λ_theoretical)
        k = argmin(diffs)
        push!(lfc_centers_λ, lfc_centers_λ_theoretical[k])
        push!(peak_integers, lfc_peak_integers[k])
    end
    return lfc_centers_pix, lfc_centers_λ, peak_integers, amplitudes, σs, rms, offsets
end

function gen_theoretical_peaks(λi, λf, ν0, Δν)

    # Generate the frequency grid
    n_left, n_right = 10_000, 10_000
    lfc_centers_freq_theoretical = [ν0 - n_left * Δν:Δν:ν0 + (n_right + 1) * Δν;]
    integers = [-n_left:n_right + 1;]

    # Convert to wavelength
    lfc_centers_λ_theoretical = 299792458.0 ./ lfc_centers_freq_theoretical
    reverse!(lfc_centers_λ_theoretical)
    lfc_centers_λ_theoretical .*= 1E9
    reverse!(integers)

    # Only peaks within the bounds
    good = findall((lfc_centers_λ_theoretical .> λi - 0.1) .&& (lfc_centers_λ_theoretical .< λf + 0.1))
    lfc_centers_λ_theoretical = lfc_centers_λ_theoretical[good]
    integers = integers[good]

    return integers, lfc_centers_λ_theoretical
end

function estimate_background(lfc_λ, lfc_flux, ν0, Δν)
    good = findall(isfinite.(lfc_λ) .&& isfinite.(lfc_λ))
    xi, xf = minimum(good), maximum(good)
    λi, λf = lfc_λ[xi], lfc_λ[xf]
    peak_spacing = estimate_peak_spacing(xi, xf, λi, λf, lfc_λ, ν0, Δν)
    min_peak_spacing = min(peak_spacing(xi), peak_spacing(xf))
    background = maths.generalized_median_filter1d(lfc_flux, width=Int(round(2 * min_peak_spacing)), p=0.01)
    return background
end

function estimate_continuum(lfc_λ, lfc_flux, ν0, Δν)
    good = findall(isfinite.(lfc_λ) .&& isfinite.(lfc_λ))
    xi, xf = minimum(good), maximum(good)
    λi, λf = lfc_λ[xi], lfc_λ[xf]
    peak_spacing = estimate_peak_spacing(xi, xf, λi, λf, lfc_λ, ν0, Δν)
    min_peak_spacing = min(peak_spacing(xi), peak_spacing(xf))
    continuum = maths.generalized_median_filter1d(lfc_flux, width=Int(round(2 * min_peak_spacing)), p=0.99)
    return continuum
end