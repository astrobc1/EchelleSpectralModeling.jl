
export GaussHermiteLSF


struct GaussHermiteLSF
    deg::Int
    σ_bounds::Vector{Float64}
    coeff_bounds::Vector{Float64}
end

GaussHermiteLSF(;deg::Int=0, σ_bounds::Vector{<:Real}, coeff_bounds::Vector{<:Real}) = GaussHermiteLSF(deg, Float64.(σ_bounds), Float64.(coeff_bounds))

function build(lsf::GaussHermiteLSF, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    coeffs = [params["a$i"] for i=0:lsf.deg]
    return build(lsf, coeffs, templates["λlsf"])
end


function build(lsf::GaussHermiteLSF, coeffs::Vector{<:Real}, λlsf::AbstractVector{<:Real}; zero_centroid::Bool=lsf.deg > 0)
    σ = coeffs[1]
    herm = gauss_hermite(λlsf ./ σ, lsf.deg)
    kernel = herm[:, 1]
    if lsf.deg == 0  # just a Gaussian
        return kernel ./ sum(kernel)
    end
    for k=2:lsf.deg+1
        kernel .+= coeffs[k] .* herm[:, k]
    end
    if zero_centroid
        λcen = sum(abs.(kernel) .* λlsf) ./ sum(abs.(kernel))
        return build(lsf, coeffs, λlsf .+ λcen, zero_centroid=false)
    end
    kernel ./= sum(kernel)
    return kernel
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


function initialize!(lsf::GaussHermiteLSF, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    δλ = nanmedian(diff(templates["λ"]))
    templates["λlsf"] = get_lsfkernel_λgrid(lsf.σ_bounds[2] * 2.355, δλ)
    a0 = nanmean(lsf.σ_bounds)
    ak = nanmean(lsf.coeff_bounds)
    if ak == 0
        ak = lsf.coeff_bounds[1] + 0.55 * (lsf.coeff_bounds[2] - lsf.coeff_bounds[1])
    end
    for i in eachindex(data)
        push!(params[i]; name="a0", value=a0, lower_bound=lsf.σ_bounds[1], upper_bound=lsf.σ_bounds[2])
        for k=1:lsf.deg
            params[i]["a$k"] = (value=ak, lower_bound=lsf.coeff_bounds[1], upper_bound=lsf.coeff_bounds[2])
        end
    end
end


enforce_positivity(lsf::GaussHermiteLSF) = lsf.deg > 0