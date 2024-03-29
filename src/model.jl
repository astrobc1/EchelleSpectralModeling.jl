export SpectralForwardModel, initialize!, build, get_model_λ_grid, get_initial_params


struct SpectralForwardModel{W, L, C, S, T, G}
    λ::Vector{Float64}
    λsolution::W
    lsf::L
    continuum::C
    star::S
    tellurics::T
    gascell::G
end


function SpectralForwardModel(λ::Vector{Float64};
        λsolution=nothing, lsf=nothing, continuum=nothing,
        star=nothing, tellurics=nothing, gascell=nothing
    )
    return SpectralForwardModel(λ, λsolution, lsf, continuum, star, tellurics, gascell)
end

get_initial_params!(params::Parameters, ::Nothing, ::Any) = params

function get_initial_params(data::Vector{DataFrame}, model::SpectralForwardModel)
    params = [Parameters() for _ in data]
    for (p, d) in zip(params, data)
        get_initial_params!(p, model.λsolution, d)
        get_initial_params!(p, model.lsf, d)
        get_initial_params!(p, model.continuum, d)
        get_initial_params!(p, model.star, d)
        get_initial_params!(p, model.tellurics, d)
        get_initial_params!(p, model.gascell, d)
    end
    return params
end


function build(model::SpectralForwardModel, params::Parameters, data::DataFrame; drop::Tuple=())

    # Initialize model of ones
    model_spec = ones(length(model.λ))

    # Star
    if !isnothing(model.star) && "star" ∉ drop
        model_spec .*= build(model.star, model.λ, params, data)
    end
    
    # Gas Cell
    if !isnothing(model.gascell) && "gascell" ∉ drop
        model_spec .*= build(model.gascell, model.λ, params, data)
    end

    # All tellurics
    if !isnothing(model.tellurics) && "tellurics" ∉ drop
        model_spec .*= build(model.tellurics, model.λ, params, data)
    end

    # Convolve
    if !isnothing(model.lsf) && "lsf" ∉ drop
        r = convolve_spectrum(model.lsf, model_spec, model.λ, params, data)
        model_spec .= r[1]
        kernel = r[2]
    else
        kernel = nothing
    end

    # Continuum
    if !isnothing(model.continuum) && "continuum" ∉ drop
        model_spec .*= build(model.continuum, model.λ, params, data)
    end

    # Generate the wavelength solution of the data
    if !isnothing(model.λsolution)
        data_λ = build(model.λsolution, params, data)
        model_spec_lr = interp1d(model.λ, model_spec, data_λ, extrapolate=false)
        out = (data_λ, model_spec_lr, kernel)
    else
        out = (model.λ, model_spec, kernel)
    end

    # Return
    return out

end


function get_model_grid_δλ(data::Vector{DataFrame}; oversample::Real=4)
    λi, λf = get_data_λ_bounds(data)
    Δλ = λf - λi
    xi, xf = get_data_pixel_bounds(data)
    Δx = xf - xi
    δλ = (Δλ / Δx) / oversample
    return δλ
end


function get_model_λ_grid(data::Vector{DataFrame}; pad::Real=1, oversample::Real=4)
    λi, λf = get_data_λ_bounds(data)
    δλ = get_model_grid_δλ(data; oversample)
    λ = collect((λi-pad):δλ:(λf+pad))
    return λ
end


convolve_spectrum(lsf::Nothing, model_spec::AbstractVector{<:Real}, λ::Vector{<:Real}, params::Parameters, data::DataFrame) = model_spec


function convolve_spectrum(lsf::Any, model_spec::AbstractVector{<:Real}, λ::Vector{<:Real}, params::Parameters, data::DataFrame)
    kernel = build(lsf, λ, params, data)
    model_specc = convolve1d(model_spec, kernel)
    return model_specc, kernel
end