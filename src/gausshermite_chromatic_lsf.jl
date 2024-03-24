
export GaussHermiteChunkedLSF


struct GaussHermiteChunkedLSF
    degλ::Int
    degh::Int
    n_chunks::Int
    σ_bounds::Vector{Float64}
    coeff_bounds::Vector{Float64}
end

GaussHermiteChunkedLSF(;degλ::Int=0, degh::Int=0, n_chunks::Int=1, σ_bounds::Vector{<:Real}, coeff_bounds::Vector{<:Real}) = GaussHermiteChunkedLSF(degλ, degh, n_chunks, Float64.(σ_bounds), Float64.(coeff_bounds))

function build(lsf::GaussHermiteChunkedLSF, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    coeffs = fill(NaN, lsf.n_chunks, lsf.degh+1)
    #@infiltrate
    if lsf.n_chunks > 1
        xc = templates["λ"][Int.(round.(nanmean.(templates["lsf_chunks_convolve"])))]
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


function build(lsf::GaussHermiteChunkedLSF, coeffs::Vector{<:Real}, λlsf::AbstractVector{<:Real}; zero_centroid::Bool=lsf.degh > 0)
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


function initialize!(lsf::GaussHermiteChunkedLSF, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    δλ = nanmedian(diff(templates["λ"]))
    templates["λlsf"] = get_lsfkernel_λgrid(lsf.σ_bounds[2] * 2.355, δλ)
    templates["lsf_chunks_convolve"], templates["lsf_distrust"] = generate_chunks(lsf, templates["λ"], data)
    λi, λf = get_data_λ_bounds(data)
    λi -= 0.2
    λf += 0.2
    templates["lsf_chunks_poly"] = collect(range(λi, λf, length=lsf.degλ+1))
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

function generate_chunks(lsf::GaussHermiteChunkedLSF, λ, data)
    λi, λf = get_data_λ_bounds(data)
    λi -= 0.2
    λf += 0.2
    xi, xf = nanargmin(abs.(λ .- λi)), nanargmin(abs.(λ .- λf))
    nx = xf - xi + 1
    δλ = λ[3] - λ[2]
    Δλ = 10 * lsf.σ_bounds[2]
    n_distrust = Int(ceil(Δλ / δλ / 2))
    chunks = Tuple{Int, Int}[]
    chunk_overlap = Int(round(2.5 * n_distrust))
    chunk_width = Int(round(nx / lsf.n_chunks + chunk_overlap))
    push!(chunks, (xi, Int(round(min(xi + chunk_width, xf)))))
    if chunks[1][2] == xf
        return chunks, 0
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
    return chunks, n_distrust
end

function convolve_spectrum(lsf::GaussHermiteChunkedLSF, model_spec::AbstractVector{<:Real}, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    kernels = build(lsf, templates, params, data)
    model_specc = fill(NaN, length(model_spec), lsf.n_chunks)
    for j=1:lsf.n_chunks
        xi, xf = templates["lsf_chunks_convolve"][j]
        model_specc[xi:xf, j] .= convolve1d(view(model_spec, xi:xf), kernels[j])
        k1 = xi + templates["lsf_distrust"] - 1
        k2 = xf - templates["lsf_distrust"] + 1
        model_specc[1:k1, j] .= NaN
        model_specc[k2:end, j] .= NaN
    end
    model_specc = nanmean(model_specc, dim=2)
    return model_specc, kernels
end


function check_positive(lsf::GaussHermiteChunkedLSF, kernels::Vector{<:Vector{<:Real}})
    if lsf.degh > 0
        return all(all.((x) -> x .> 0, kernels))
    else
        return true
    end
end

