function convolve1d(x::Vector{<:Real}, k::Vector{<:Real})
    nx = length(x)
    nk = length(k)
    n_pad = Int(floor(nk / 2))
    out = fill(NaN, nx)
    
    # Left values
    @turbo for i=1:n_pad
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            iii = max(ii, 1)
            s += ifelse(1 <= ii <= nx, x[iii] * k[j], NaN)
        end
        out[i] = s
    end

    # Middle values
    @turbo for i=n_pad+1:nx-n_pad
        s = 0.0
        for j=1:nk
            s += x[i - n_pad + j - 1] * k[j]
        end
        out[i] = s
    end

    # Right values
    @turbo for i=nx-n_pad+1:nx
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            iii = min(ii, nx)
            s += ifelse(1 <= ii <= nx, x[iii] * k[j], NaN)
        end
        out[i] = s
    end

    # Return out
    return out

end


function get_lsfkernel_λgrid(fwhm::Real, δλ::Real; nσ::Int=10)
    Δλ = nσ * fwhm / 2.355
    n = Int(ceil(Δλ / δλ))
    if iseven(n)
        n += 1
    end
    n2 = n / 2
    λrel = [Int(ceil(-n2)):Int(floor(n2));] .* δλ
    return λrel
end

function get_toy_kernel(λ, R)
    fwhm = nanmean(λ) / R
    σ = fwhm / 2.355
    δλ = λ[3] - λ[2]
    λrel = get_lsfkernel_λgrid(fwhm, δλ)
    kernel = gauss.(λrel, 1, 0, σ)
    kernel ./= sum(kernel)
    return kernel
end


function toy_convolve(λ, spec; R=100_000)
    kernel = get_toy_kernel(λ, R)
    specc = convolve1d(spec, kernel)
    return specc
end


convolve_spectrum(lsf::Nothing, model_spec::AbstractVector{<:Real}, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame) = model_spec

function convolve_spectrum(lsf::Any, model_spec::AbstractVector{<:Real}, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    kernel = build(lsf, templates, params, data)
    model_specc = convolve1d(model_spec, kernel)
    return model_specc, kernel
end