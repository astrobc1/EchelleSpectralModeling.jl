using EchelleBase
using NaNStatistics
using Polynomials
using Infiltrator
using PyPlot
using LsqFit

SPEED_OF_LIGHT_MPS = 299792458.0

function get_peak_spacing(λ, λi, λf, ν0, Δν; deg=1)

    # Get peaks
    integers, centers_λ = gen_theoretical_peaks(λi, λf, ν0, Δν)

    # Pixel grid
    xarr = 1:length(λ)

    # Storage arrays
    peak_separations = Float64[]
    x = Float64[]

    # Compute peak separation for each spot
    for i=1:length(centers_λ)-1
        k1 = argmin(abs.(centers_λ[i] .- λ))
        k2 = argmin(abs.(centers_λ[i+1] .- λ))
        push!(peak_separations, abs(xarr[k1] - xarr[k2]))
        push!(x, (xarr[k1] + xarr[k2]) / 2)
    end

    # Make a polynomial, assume linear across order
    pfit = Polynomials.fit(x, peak_separations, deg)

    # Return
    return pfit
end

function get_peaks(λ, flux, ν0, Δν; σ_guess=[0.2, 1.4, 3.0], μ_bounds=[-1, 1])

    # Generate theoretical peaks
    nx = length(λ)
    xarr = [1:nx;]
    good = findall(isfinite.(λ) .&& isfinite.(flux))
    xi, xf = minimum(good), maximum(good)
    λi, λf = λ[xi], λ[xf]
    integers, centers_λ = gen_theoretical_peaks(λi, λf, ν0, Δν)

    # Ignore first and last peak
    integers, centers_λ = integers[2:end-1], centers_λ[2:end-1]
    n_peaks = length(centers_λ)

    # Peak spacing
    peak_spacing = get_peak_spacing(λ, λi, λf, ν0, Δν)
    min_peak_spacing = minimum(peak_spacing.(1:nx))

    # Store pixel peaks
    centers_pixels = Float64[xarr[maths.nanargminimum(abs.(centers_λ[i] .- λ))] for i=1:n_peaks]

    # First iteratively refine based on centroid just to get the window right
    for i=1:10
        for j=1:n_peaks

            # Window
            use = findall((xarr .>= floor(centers_pixels[j] - peak_spacing(centers_pixels[j]) / 2)) .&& (xarr .<= ceil(centers_pixels[j] + peak_spacing(centers_pixels[j]) / 2)))
            xx, yy = @views xarr[use], flux[use]

            # Centroid
            centers_pixels[j] = maths.weighted_mean(xx, yy)
        end
    end

    # Fit results
    amplitudes = fill(NaN, n_peaks)
    σs = fill(NaN, n_peaks)
    offsets = fill(NaN, n_peaks)
    slopes = fill(NaN, n_peaks)
    rms = fill(NaN, n_peaks)

    # Fit peaks
    for i=1:n_peaks

        # Window
        use = findall((xarr .>= floor(centers_pixels[i] - peak_spacing(centers_pixels[i]) / 2)) .&& (xarr .<= ceil(centers_pixels[i] + peak_spacing(centers_pixels[i]) / 2)))
        xx, yy = xarr[use], flux[use]

        # Remove approx baseline
        yy .-= nanminimum(yy)
        peak_val = nanmaximum(yy)

        # Pars and bounds
        p0 = [peak_val, centers_pixels[i], σ_guess[2], 0.1 * peak_val, 0.01 * peak_val]
        lb = [0.7 * peak_val, centers_pixels[i] + μ_bounds[1], σ_guess[1], -0.5 * peak_val, -0.1 * peak_val]
        ub = [1.3 * peak_val, centers_pixels[i] + μ_bounds[2], σ_guess[3], 0.5 * peak_val, 0.1 * peak_val]

        # Model
        model = (_, pars) -> begin
            return maths.gauss(xx, pars[1], pars[2], pars[3]) .+ pars[4] .+ pars[5] .* (xx .- nanmean(xx))
        end

        # Fit
        try
            result = LsqFit.curve_fit(model, xx, yy, p0, lower=lb, upper=ub)
            pbest = result.param
            amplitudes[i] = pbest[1]
            centers_pixels[i] = pbest[2]
            σs[i] = pbest[3]
            offsets[i] = pbest[4]
            slopes[i] = pbest[5]
            rms[i] = maths.rmsloss(model(xx, pbest), yy)
        catch
            nothing
        end
    end

    # Return
    return centers_pixels, centers_λ, integers, amplitudes, σs, rms, offsets, slopes
end



function gen_theoretical_peaks(λi::Real, λf::Real, ν0::Real, Δν::Real, n_left=10_000, n_right=10_000)

    # Generate the frequency grid
    centers_ν = [ν0 - n_left * Δν:Δν:ν0 + (n_right + 1) * Δν;]
    integers = [-n_left:n_right;]

    # Convert to wavelength
    centers_λ = SPEED_OF_LIGHT_MPS ./ centers_ν
    reverse!(centers_λ)
    centers_λ .*= 1E9 # meters to nm
    reverse!(integers)

    # Only peaks within the bounds
    good = findall((centers_λ .> λi) .&& (centers_λ .< λf))
    centers_λ = centers_λ[good]
    integers = integers[good]

    # Return
    return integers, centers_λ
end

function estimate_background(λ, flux, ν0, Δν)

    # Good data
    nx = length(λ)
    good = findall(isfinite.(λ) .&& isfinite.(flux))
    xi, xf = minimum(good), maximum(good)
    λi, λf = λ[xi], λ[xf]

    # Estimate peak spacing
    peak_spacing = get_peak_spacing(λ, λi, λf, ν0, Δν)
    min_peak_spacing = minimum(peak_spacing.(1:nx))

    # Get background using minima
    background = maths.generalized_median_filter1d(flux, width=Int(round(2 * min_peak_spacing)), p=0.01)

    # Smooth
    background .= maths.poly_filter([1:nx;], background, width=3 * (peak_spacing(xi) + peak_spacing(xf)) / 2, deg=2)

    # Return
    return background
end

function estimate_continuum(λ, flux, ν0, Δν)

    # Good data
    nx = length(λ)
    good = findall(isfinite.(λ) .&& isfinite.(flux))
    xi, xf = minimum(good), maximum(good)
    λi, λf = λ[xi], λ[xf]

    # Estimate peak spacing
    peak_spacing = get_peak_spacing(λ, λi, λf, ν0, Δν)
    min_peak_spacing = minimum(peak_spacing.(1:nx))

    # Get background using minima
    continuum = maths.generalized_median_filter1d(flux, width=Int(round(2 * min_peak_spacing)), p=0.99)

    # Smooth
    continuum .= maths.poly_filter([1:nx;], background, width=6 * (peak_spacing(xi) + peak_spacing(xf)) / 2, deg=2)

    # Return
    return continuum
end