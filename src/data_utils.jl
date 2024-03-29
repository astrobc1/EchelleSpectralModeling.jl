export get_data_λ_bounds, get_data_pixel_bounds

function get_data_λ_bounds(data::Vector{DataFrame})
    λi, λf = Inf, -Inf
    for (i, d) ∈ enumerate(data)
        good = findall(isfinite.(d.spec))
        if length(good) > 0
            _λi = d.λ[good[1]]
            _λf = d.λ[good[end]]
            if _λi < λi
                λi = _λi
            end
            if _λf > λf
                λf = _λf
            end
        end
    end
    @assert λf > λi
    return λi, λf
end


function get_data_pixel_bounds(data::Vector{DataFrame})
    xi, xf = Inf, 1
    for d ∈ data
        good = findall(isfinite.(d.spec))
        if length(good) > 0
            _xi = good[1]
            _xf = good[end]
            if _xi < xi
                xi = _xi
            end
            if _xf > xf
                xf = _xf
            end
        end
    end
    @assert xf > xi
    return xi, xf
end