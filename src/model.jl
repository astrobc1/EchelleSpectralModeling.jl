export SpectralForwardModel, initialize!, build


struct SpectralForwardModel{W, L, C, S, T, G}
    λsolution::W
    lsf::L
    continuum::C
    star::S
    tellurics::T
    gascell::G
    oversample::Float64
    templates::Dict{String, <:Any}
end


function SpectralForwardModel(;
        λsolution=nothing, lsf=nothing, continuum=nothing,
        star=nothing, tellurics=nothing, gascell=nothing,
        oversample::Real=8
    )
    return SpectralForwardModel(λsolution, lsf, continuum, star, tellurics, gascell, Float64(oversample), Dict{String, Any}())
end

function initialize!(model::SpectralForwardModel, data::Vector{DataFrame})
    model.templates["λ"] = get_model_λ_grid(model, data)
    params = [Parameters() for _ in data]
    initialize!(model.λsolution, model.templates, params, data)
    initialize!(model.lsf, model.templates, params, data)
    initialize!(model.continuum, model.templates, params, data)
    initialize!(model.star, model.templates, params, data)
    initialize!(model.tellurics, model.templates, params, data)
    initialize!(model.gascell, model.templates, params, data)
    return params
end


function build(model::SpectralForwardModel, params::Parameters, data::DataFrame; drop::Tuple=())

    # Initialize model
    model_spec = ones(length(model.templates["λ"]))

    # Star
    if !isnothing(model.star) && "star" ∉ drop
        model_spec .*= build(model.star, model.templates, params, data)
    end
    
    # Gas Cell
    if !isnothing(model.gascell) && "gascell" ∉ drop
        model_spec .*= build(model.gascell, model.templates, params, data)
    end

    # All tellurics
    if !isnothing(model.tellurics) && "tellurics" ∉ drop
        model_spec .*= build(model.tellurics, model.templates, params, data)
    end

    # Convolve
    if !isnothing(model.lsf) && "lsf" ∉ drop
        r = convolve_spectrum(model.lsf, model_spec, model.templates, params, data)
        model_spec .= r[1]
        kernel = r[2]
    else
        kernel = nothing
    end

    # Continuum
    if !isnothing(model.continuum) && "continuum" ∉ drop
        model_spec .*= build(model.continuum, model.templates, params, data)
    end

    # Generate the wavelength solution of the data
    if !isnothing(model.λsolution)
        data_λ = build(model.λsolution, model.templates, params, data)
        model_spec = interp1d(model.templates["λ"], model_spec, data_λ, extrapolate=true)
        out = (data_λ, model_spec, kernel)
    else
        out = (model.templates["λ"], model_spec, kernel)
    end

    # Return
    return out

end


function get_model_grid_δλ(model::SpectralForwardModel, data::Vector{DataFrame})
    λi, λf = get_data_λ_bounds(data)
    Δλ = λf - λi
    xi, xf = get_data_pixel_bounds(data)
    Δx = xf - xi
    δλ = (Δλ / Δx) / model.oversample
    return δλ
end


function get_model_λ_grid(model::SpectralForwardModel, data::Vector{DataFrame}; pad::Real=1)
    λi, λf = get_data_λ_bounds(data)
    δλ = get_model_grid_δλ(model, data)
    λ = [(λi-pad):δλ:(λf+pad);]
    return λ
end


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