
export GaussHermiteLSF


struct GaussHermiteLSF
    λkernel::Vector{Float64}
    degλ::Int
    degh::Int
    knots_poly::Union{Vector{Float64}, Nothing}
    chunks_convolve::Union{Vector{NTuple{2, Float64}}, Nothing}
    n_pad::Int
    σ_bounds::NTuple{2, Float64}
    coeff_bounds::NTuple{2, Float64}
end


function GaussHermiteLSF(;λ::Vector{Float64}, λ_range::NTuple{2, <:Real}, degλ::Int=0, degh::Int=0, n_chunks::Int=1, σ_bounds::NTuple{2, <:Real}, coeff_bounds::NTuple{2, <:Real})
    δλ = nanmedian(diff(λ))
    λkernel = get_lsfkernel_λgrid(σ_bounds[2] * 2.355, δλ)
    if n_chunks == 1
        n_pad = 0
        chunks_convolve = nothing
        knots_poly = nothing
    else
        chunks_convolve, n_pad = get_lsf_chunks(λ, λkernel, λ_range, n_chunks)
        knots_poly = collect(range(λ_range[1], λ_range[2], length=lsf.degλ+1))
    end
    return GaussHermiteLSF(λkernel, degλ, degh, chunks_poly, chunks_convolve, n_pad, Float64.(σ_bounds), Float64.(coeff_bounds))
end


function build(lsf::GaussHermiteLSF, λ::Vector{<:Real}, params::Parameters, data::DataFrame)
    coeffs = fill(NaN, lsf.n_chunks, lsf.degh + 1)
    if lsf.n_chunks > 1
        xc = λ[Int.(round.(nanmean.(templates["lsf_chunks_convolve"])))]
    end
    for k=0:lsf.degh
        y = [params["a_$(i)_$k"] for i=1:lsf.degλ+1]
        if lsf.n_chunks > 1
            pfit = Polynomials.fit(ArnoldiFit, templates["lsf_chunks_poly"], y, lsf.degλ)
            coeffs[:, k+1] .= pfit.(xc)
        else
            coeffs[:, k+1] .= y
        end
    end
    kernels = Vector{Vector{Float64}}(undef, lsf.n_chunks)
    for i=1:lsf.n_chunks
        kernels[i] = build(lsf, coeffs[i, :], templates["λlsf"])
    end
    return kernels
end


function build(lsf::GaussHermiteLSF, coeffs::Vector{<:Real}, λlsf::Vector{<:Real}; zero_centroid::Bool=lsf.degh > 0)
    σ = coeffs[1]
    herm = gauss_hermite(λlsf ./ σ, lsf.degh)
    kernel = herm[:, 1]
    if lsf.degh == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end
    for k=2:lsf.degh+1
        kernel .+= coeffs[k] .* herm[:, k]
    end
    if zero_centroid
        λcen = sum(abs.(kernel) .* λlsf) ./ sum(abs.(kernel))
        return build(lsf, coeffs, λlsf .+ λcen, zero_centroid=false)
    end
    kernel ./= sum(kernel)
    return kernel
end


function get_initial_params!(lsf::GaussHermiteLSF, params::Parameters, λ::Vector{<:Real}, data::DataFrame)
    templates["lsf_chunks_convolve"], templates["lsf_distrust"] = generate_chunks(lsf, templates["λ"], data)
    λi, λf = get_data_λ_bounds(data)
    λi -= 0.2
    λf += 0.2
    if lsf.degλ > 0
        templates["lsf_chunks_poly"] = collect(range(λi, λf, length=lsf.degλ+1))
    end
    a0 = nanmean(lsf.σ_bounds)
    ak = nanmean(lsf.coeff_bounds)
    if ak == 0 && lsf.coeff_bounds[1] != lsf.coeff_bounds[2]
        ak = lsf.coeff_bounds[1] + 0.55 * (lsf.coeff_bounds[2] - lsf.coeff_bounds[1])
    end
    for i in eachindex(data)
        # j indexes polynomial set point (1, ..., n_points)
        # k indexes lsf coeff (0, ..., deg-1)
        for j=1:lsf.degλ+1
            push!(params[i]; name="a_$(j)_0", value=a0, lower_bound=lsf.σ_bounds[1], upper_bound=lsf.σ_bounds[2])
            for k=1:lsf.degh
                params[i]["a_$(j)_$k"] = (value=ak, lower_bound=lsf.coeff_bounds[1], upper_bound=lsf.coeff_bounds[2])
            end
        end
    end
end

function get_lsf_chunks(λ::Vector{<:Real}, λkernel::Vector{<:Real}, λ_range::NTuple{2, <:Real}, n_chunks::Int)
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

function convolve_spectrum(lsf::GaussHermiteLSF, model_spec::AbstractVector{<:Real}, params::Parameters, data::DataFrame)
    kernels = build(lsf, params, data)
    model_specc = fill(NaN, length(model_spec), lsf.n_chunks)
    for i=1:lsf.n_chunks
        xi, xf = lsf.chunks_convolve[i]
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

