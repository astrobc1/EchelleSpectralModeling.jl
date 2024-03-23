export estimate_continuum

function estimate_continuum(x::AbstractVector{<:Real}, spec::Vector{<:Real}; width::Real, deg::Int)
    spec_smooth = quantile_filter1d(spec, width=5, q=0.5)
    y = quantile_filter1d(spec_smooth; width, q=1)
    good = findall(isfinite.(y) .&& isfinite.(x))
    pfit = @views Polynomials.fit(ArnoldiFit, x[good], y[good], deg)
    return pfit.(x)
end
