function get_knots(pixel_centers, orders, n_splines_inter_order, n_splines_intra_order)
    xmin, xmax = nanminimum(pixel_centers), nanmaximum(pixel_centers)
    Δx = xmax - xmin
    δx = Δx / (n_splines_intra_order + 1)
    knots_pixels = [xmin:δx:xmax;]
    omin, omax = nanminimum(orders), nanmaximum(orders)
    Δm = omax - omin
    δm = Δm / (n_splines_inter_order + 1)
    knots_orders = [omin:δm:omax;]
    return knots_pixels, knots_orders
end

function fit_peaks_bs2d(pixel_centers, orders, λ_centers, weights, deg_inter_order, deg_intra_order, n_splines_inter_order, n_splines_intra_order, n_iterations=1, max_vel_cut=200)

    # Running weights
    weights_running = copy(weights)

    # Knots
    knots_pixels, knots_orders = get_knots(pixel_centers, orders, n_splines_inter_order, n_splines_intra_order)

    # Load scipy
    scipyinterp = pyimport("scipy.interpolate")
    spl = nothing

    for i=1:n_iterations

        # Get good data
        good = findall(isfinite.(weights_running) .&& (weights_running .> 0) .&& isfinite.(pixel_centers) .&& isfinite.(λ_centers))

        # Lsq
        spl = scipyinterp.LSQBivariateSpline(pixel_centers[good], orders[good], λ_centers[good], knots_pixels, knots_orders, w=weights_running[good], kx=deg_intra_order, ky=deg_inter_order)

        # Flag
        model_best = [spl(pixel_centers[j], orders[j])[1] for j=1:length(pixel_centers)]
        residuals = λ_centers .- model_best
        residuals .= maths.δλ2δv.(residuals, λ_centers)
        useσ = findall(weights_running .> 0)
        bad = findall(abs.(residuals) .> min(3 * maths.robust_σ(residuals[useσ]), max_vel_cut) .&& weights_running .> 0)
        if length(bad) == 0
           break
        end
        weights_running[bad] .= 0
    end

    # Good peaks
    good_peaks = findall(isfinite.(weights_running) .&& (weights_running .> 0) .&& isfinite.(pixel_centers) .&& isfinite.(λ_centers))

    # Return
    return spl, good_peaks
end

function build_λsolution_bs2d(spl, nx, orders)
    n_orders = length(orders)
    λ = fill(NaN, (n_orders, nx))
    for i=1:nx
	    for j=1:n_orders
            λ[j, i] = spl(i, orders[j])[1]
        end
    end
    return λ
end