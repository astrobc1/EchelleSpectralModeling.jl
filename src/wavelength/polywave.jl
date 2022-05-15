using SpecialMatrices

using EchelleBase
using EchelleSpectralModeling

using PyCall
using SciPy
using Statistics

export PolyλSolution

struct PolyλSolution <: SpectralModelComponent
    deg::Int
    bounds::Vector{Float64}
end

PolyλSolution(;deg, bounds) = PolyλSolution(deg, bounds)

function get_pixel_lagrange_points(m::PolyλSolution, sregion)
    return Int.(round.(collect(range(sregion.pixmin+1, sregion.pixmax-1, length=m.deg + 1))))
end

function get_λ_lagrange_zero_points(m::PolyλSolution, sregion, λ_estimate)
    pixel_set_points = get_pixel_lagrange_points(m::PolyλSolution, sregion)
    λ_zero_points = λ_estimate[pixel_set_points]
    return λ_zero_points
end


function EchelleSpectralModeling.build(m::PolyλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    pixel_lagrange_points = get_pixel_lagrange_points(m, sregion)
    λ_lagrange_points = [pars["λ$i"].value for i=1:m.deg+1]
    return build(m, pixel_lagrange_points, λ_lagrange_points, length(data.data.flux))
end

function EchelleSpectralModeling.build(m::PolyλSolution, xs::AbstractVector, λs::AbstractVector, nx::Int)
    V = Vandermonde(xs)
    pcoeffs = V \ λs
    λ = Polynomial(pcoeffs).(1:nx)
    return λ
end

function EchelleSpectralModeling.get_init_parameters(m::PolyλSolution, data::SpecData1d, sregion)
    pars = Parameters()
    λ_estimate = get_λsolution_estimate(data, sregion)
    λ_zero_points = get_λ_lagrange_zero_points(m, sregion, λ_estimate)
    for i=1:m.deg+1
        pname = "λ$i"
        pars[pname] = Parameter(value=λ_zero_points[i], lower_bound=λ_zero_points[i] + m.bounds[1], upper_bound=λ_zero_points[i] + m.bounds[2])
    end
    return pars

end

## 2d CC polynomial
function build_λsolution_chebyval2d(pixels, orders, max_pixel, max_order, coeffs)
    nx = length(pixels)
    nm = length(orders)
    λ = fill(NaN, (nm, nx))
    for i=1:nx
        for m=1:nm
            order = orders[m]
            pixel = pixels[i]
            λ[m, i] = maths.chebyval2d(pixel / max_pixel, order / max_order, coeffs) / order
        end
    end
    return λ
end

function build_λsolution_chebyval2d_flat(pixels, orders, max_pixel, max_order, coeffs)
    n = length(pixels)
    λ = zeros(n)
    for i=1:n
        λ[i] = maths.chebyval2d(pixels[i] / max_pixel, orders[i] / max_order, coeffs) / orders[i]
    end
    return λ
end

function fit_peaks_cc2d(pixel_centers, λ_centers, weights, orders, max_pixel, max_order, nx, deg_inter_order, deg_intra_order, n_iterations=1)

    # Pars and bounds
    loss = (coeffs, _weights) -> begin
        #_coeffs = reshape(coeffs, (deg_inter_order+1, deg_intra_order+1))
        numpy = pyimport("numpy")
        model = build_λsolution_chebyval2d_flat(pixel_centers, orders, max_pixel, max_order, numpy.reshape(coeffs, (deg_inter_order+1, deg_intra_order+1)))
        wres = _weights .* (λ_centers .- model)
        return wres
    end

    u0 = ones((deg_inter_order + 1) * (deg_intra_order + 1)) / 100
    weights_running = copy(weights)
    coeffs_best = copy(u0)
    numpy = pyimport("numpy")

    for i=1:n_iterations
        result = SciPy.optimize.least_squares(loss, coeffs_best, method="lm", args=(weights_running,), max_nfev=200 * length(u0))
        coeffs_best .= result["x"]
        model_best = build_λsolution_chebyval2d_flat(pixel_centers, orders, max_pixel, max_order, numpy.reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1)))
        residuals = λ_centers .- model_best
        residuals .= maths.δλ2δv(residuals, model_best)
        bad = findall(abs.(residuals) .> 3 * std(residuals))
        if length(bad) == 0
            break
        end
        weights_running[bad] .= 0
    end

    # Return
    coeffs_best = numpy.reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1))
    return coeffs_best
end