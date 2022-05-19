using SpecialMatrices

using EchelleBase
using EchelleSpectralModeling

using PyCall
using LsqFit
using Infiltrator
using ForwardDiff
using Flux
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
    m, n = size(coeffs)
    λ = fill(NaN, (nm, nx))
    for i=1:nx
        for o=1:nm
            s = 0.0
            for j=1:n
                for k=1:m
                    s += coeffs[k, j] * maths.chebval(pixels[i] / max_pixel, j-1) * maths.chebval(orders[o] / max_order, k-1) / orders[o]
                end
            end
            λ[o, i] = s
        end
    end
    return λ
end

function build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, coeffs, orders)
    nx = length(chebs_pixels)
    λ = zeros(nx)
    m, n = size(coeffs)
    for i=1:nx
        s = 0.0
        for j=1:n
            for k=1:m
                s += coeffs[k, j] * chebs_pixels[i][j] * chebs_orders[i][k] / orders[i]
            end
        end
        λ[i] = s
    end
    return λ
end

function fit_peaks_cc2d(pixel_centers, orders, λ_centers, weights, max_pixel, max_order, nx, deg_inter_order, deg_intra_order, n_iterations=1)

    # Chebs
    chebs_pixels, chebs_orders = maths.get_chebvals(pixel_centers, orders, max_pixel, max_order, deg_intra_order, deg_inter_order)

    # Initial params and weights
    p0 = ones((deg_inter_order + 1) * (deg_intra_order + 1)) / 100
    weights_running = copy(weights)
    coeffs_best = copy(p0)

    # Load scipy
    scipyopt = pyimport("scipy.optimize")

    for i=1:n_iterations
        
        # Loss
        loss = (coeffs) -> begin
            _model = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs, (deg_inter_order+1, deg_intra_order+1)), orders)
            return weights_running .* (λ_centers .- _model)
        end

        # Lsq
        result = scipyopt.least_squares(loss, p0, max_nfev=400 * length(coeffs_best), method="lm")
        coeffs_best = result["x"]

        # Flag
        model_best = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1)), orders)
        residuals = maths.δλ2δv(λ_centers .- model_best, λ_centers)
        bad = findall(abs.(residuals) .> 3 * maths.robust_σ(residuals))
        if length(bad) == 0
            break
        end
        weights_running[bad] .= 0
    end

    # Return
    coeffs_best = reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1))
    return coeffs_best
end