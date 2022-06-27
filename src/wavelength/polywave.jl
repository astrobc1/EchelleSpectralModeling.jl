using SpecialMatrices

using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

using PyCall
using Infiltrator
using Statistics

export PolyλSolution

struct PolyλSolution <: SpectralModelComponent
    deg::Int
    bounds::Vector{Float64}
end

"""
    PolyλSolution(;deg::Int, bounds::Vector{Float64})
Construct a PolyλSolution model component of degree `deg`. The optimized parameters are set points opposed to coefficients. Each set point is(in units of wavelength) is bounded by `bounds`.
"""
PolyλSolution(;deg::Int, bounds::Vector{Float64}) = PolyλSolution(deg, bounds)

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

function fit_peaks_cheb2d(pixel_centers, orders, λ_centers, weights, max_pixel, max_order, nx, deg_inter_order, deg_intra_order, n_iterations=1, max_vel_cut=200)

    # Initial params and weights
    p0 = ones((deg_inter_order + 1) * (deg_intra_order + 1)) / 100
    pixel_centers_running = copy(pixel_centers)
    λ_centers_running = copy(λ_centers)
    weights_running = copy(weights)
    coeffs_best = copy(p0)

    # Load scipy
    scipyopt = pyimport("scipy.optimize")

    for i=1:n_iterations

        # Update bad weights
        bad = findall(.~isfinite.(weights_running) .|| (weights_running .== 0) .|| .~isfinite.(pixel_centers) .|| .~isfinite.(λ_centers))
        weights_running[bad] .= 0
        pixel_centers_running[bad] .= 0
        λ_centers_running[bad] .= 0

        # Chebs
        chebs_pixels, chebs_orders = get_chebvals(pixel_centers_running, orders, max_pixel, max_order, deg_intra_order, deg_inter_order)
        
        # Loss
        loss = (coeffs) -> begin
            _model = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs, (deg_inter_order+1, deg_intra_order+1)), orders)
            wres = weights_running .* (λ_centers_running .- _model)
            #@show nansum(wres.^2) / nansum(weights_running)
            return wres
        end

        # Lsq
        result = scipyopt.least_squares(loss, p0, max_nfev=800 * length(coeffs_best), method="lm")
        coeffs_best .= result["x"]

        # Flag
        model_best = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1)), orders)
        residuals = maths.δλ2δv.(λ_centers .- model_best, λ_centers)
        σuse = findall(isfinite.(residuals) .&& (abs.(residuals) .> 0))
        bad = findall(abs.(residuals) .> min(3 * maths.robust_σ(residuals[σuse]), max_vel_cut))
        if length(bad) == 0
           break
        end
        weights_running[bad] .= 0
    end

    good_peaks = findall(weights_running .> 0)
    coeffs_best = reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1))

    # Return
    return coeffs_best, good_peaks
end

"""
get_chebvals(pixels, orders, max_pixel::Real, max_order::Real, deg_intra_order::Int, deg_inter_order::Int)
A standard median filter where x_out[i, j] = median(x[i-w2:i+w2, j-w2:j+w2]) where w2 = ceil(width / 2).
"""
function get_chebvals(pixels, orders, max_pixel::Real, max_order::Real, deg_intra_order::Int, deg_inter_order::Int)
    chebs_pixels = Vector{Float64}[]
    chebs_orders = Vector{Float64}[]
    @assert length(pixels) == length(orders)
    for i=1:length(pixels)
        push!(chebs_pixels, maths.get_chebvals(pixels[i] / max_pixel, deg_intra_order))
        push!(chebs_orders, maths.get_chebvals(orders[i] / max_order, deg_inter_order))
    end
    return chebs_pixels, chebs_orders
end