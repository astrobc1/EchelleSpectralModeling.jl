export GaussHermiteLSF


struct GaussHermiteLSF
    λkernel::Vector{Float64}
    degh::Int
    knots_poly::Vector{Float64}
    chunks::Vector{NTuple{2, Int}}
    n_pad::Int
    σ_bounds::NTuple{2, Float64}
    coeff_bounds::NTuple{2, Float64}
end


function GaussHermiteLSF(λ::Vector{Float64}; λ_range::Tuple{<:Real, <:Real}, degλ::Int=0, degh::Int=0, n_chunks::Int=1, σ_bounds::Tuple{<:Real, <:Real}, coeff_bounds::Tuple{<:Real, <:Real})
    δλ = nanmedian(diff(λ))
    λkernel = get_lsfkernel_λgrid(σ_bounds[2] * 2.355, δλ)
    if n_chunks == 1
        n_pad = 0
        chunks = [(1, length(λ))]
        knots_poly = Float64[nanmean(λ_range)]
    else
        chunks, n_pad = get_lsf_chunks(λ, λkernel, λ_range, n_chunks)
        knots_poly = collect(range(λ_range[1], stop=λ_range[2], length=degλ+1)) 
    end
    return GaussHermiteLSF(λkernel, degh, knots_poly, chunks, n_pad, Float64.(σ_bounds), Float64.(coeff_bounds))
end


function build(lsf::GaussHermiteLSF, λ::Vector{<:Real}, params::Parameters, data::DataFrame)
    n_chunks = length(lsf.chunks)
    coeffs = fill(NaN, n_chunks, lsf.degh + 1)
    degλ = length(lsf.knots_poly) - 1
    if n_chunks > 1
        degλ = length(lsf.knots_poly) - 1
        xc = λ[Int.(round.(nanmean.(lsf.chunks)))]
    end
    for k=0:lsf.degh
        y = [params["a_$(i)_$k"] for i=1:degλ+1]
        if n_chunks > 1
            pfit = Polynomials.fit(ArnoldiFit, lsf.knots_poly, y, degλ)
            coeffs[:, k+1] .= pfit.(xc)
        else
            coeffs[:, k+1] .= y
        end
    end
    kernels = Vector{Vector{Float64}}(undef, n_chunks)
    for i=1:n_chunks
        kernels[i] = build(lsf, coeffs[i, :], lsf.λkernel)
    end
    return kernels
end


function build(lsf::GaussHermiteLSF, coeffs::Vector{<:Real}, λkernel::Vector{<:Real}; zero_centroid::Bool=lsf.degh > 0)
    σ = coeffs[1]
    herm = gauss_hermite(λkernel ./ σ, lsf.degh)
    kernel = herm[:, 1]
    if lsf.degh == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end
    for k=2:lsf.degh+1
        kernel .+= coeffs[k] .* herm[:, k]
    end
    if zero_centroid
        λcen = sum(abs.(kernel) .* λkernel) ./ sum(abs.(kernel))
        return build(lsf, coeffs, λkernel .+ λcen, zero_centroid=false)
    end
    kernel ./= sum(kernel)
    return kernel
end


function get_initial_params!(params::Parameters, lsf::GaussHermiteLSF, data::DataFrame)
    a0 = get_initial_value(lsf.σ_bounds)
    ak = get_initial_value(lsf.coeff_bounds)
    degλ = length(lsf.knots_poly) - 1
    # i indexes polynomial set point (1, ..., n_points)
    # k indexes lsf coeff (0, ..., deg-1)
    for i=1:degλ+1
        params["a_$(i)_0"] = (value=a0, bounds=lsf.σ_bounds)
        for k=1:lsf.degh
            params["a_$(i)_$k"] = (value=ak, bounds=lsf.coeff_bounds)
        end
    end
end


function get_lsf_chunks(λ::Vector{<:Real}, λkernel::Vector{<:Real}, λ_range::Tuple{<:Real, <:Real}, n_chunks::Int)
    λi, λf = λ_range
    λi -= 0.2
    λf += 0.2
    xi, xf = nanargmin(abs.(λ .- λi)), nanargmin(abs.(λ .- λf))
    nx = xf - xi + 1
    δλ = nanmedian(diff(λ))
    Δλ = λkernel[end] - λkernel[1]
    n_pad = Int(ceil(Δλ / δλ / 2))
    chunks = NTuple{2, Int}[]
    chunk_overlap = Int(round(2.5 * n_pad))
    chunk_width = Int(round(nx / n_chunks + chunk_overlap))
    push!(chunks, (xi, Int(round(min(xi + chunk_width, xf)))))
    if chunks[1][2] == xf
        return chunks
    end
    for i=2:nx
	    _xi = chunks[i-1][2] - chunk_overlap
        _xf = Int(floor(min(_xi + chunk_width, xf)))
        push!(chunks, (_xi, _xf))
        if _xf == xf
            break
        end
    end
    if (chunks[end][2] - chunks[end][1]) <= chunk_width / 2
        deleteat!(chunks, lastindex(chunks))
        deleteat!(chunks, lastindex(chunks))
        _xi = chunks[end][2] - chunk_overlap
        push!(chunks, (_xi, xf))
    end
    chunks[1] = (1, chunks[1][2])
    chunks[end] = (chunks[end][1], length(λ))
    return chunks, n_pad
end


function convolve_spectrum(lsf::GaussHermiteLSF, model_spec::AbstractVector{<:Real}, λ::Vector{<:Real}, params::Parameters, data::DataFrame)
    kernels = build(lsf, λ, params, data)
    n_chunks = length(lsf.chunks)
    model_specc = fill(NaN, length(model_spec), n_chunks)
    for i=1:n_chunks
        xi, xf = lsf.chunks[i]
        model_specc[xi:xf, i] .= convolve1d(view(model_spec, xi:xf), kernels[i])
        k1 = xi + lsf.n_pad - 1
        k2 = xf - lsf.n_pad + 1
        model_specc[1:k1, i] .= NaN
        model_specc[k2:end, i] .= NaN
    end
    model_specc = nanmean(model_specc, dim=2)
    return model_specc, kernels
end


function check_positive(lsf::GaussHermiteLSF, kernels::Vector{<:Vector{<:Real}})
    if lsf.degh > 0
        for k in kernels
            for kk in k
                if kk < 0
                    return false
                end
            end
        end
    else
        return true
    end
end


function gauss_hermite(x::AbstractVector{<:Real}, deg::Int)
    herms = fill(NaN, length(x), deg + 1)
    herms[:, 1] .= π^-0.25 .* exp.(-0.5 .* x.^2)
    if deg == 0
        return vec(herms)
    elseif deg == 1
        herms[:, 2] .= @views sqrt(2) .* herms[:, 1] .* x
        return herms
    else
        herms[:, 2] .= @views sqrt(2) .* herms[:, 1] .* x
        for k=3:deg+1
            herms[:, k] .= @views sqrt(2 / (k - 1)) .* (x .* herms[:, k-1] .- sqrt((k - 2) / 2) .* herms[:, k-2])
        end
        return herms
    end
end