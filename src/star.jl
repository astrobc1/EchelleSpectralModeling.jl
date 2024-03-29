export AugmentedStar

mutable struct AugmentedStar
    template::Union{Vector{Float64}, Nothing}
    from_flat::Bool
    name::String
    rv_abs::Float64
    vel_bounds::NTuple{2, Float64}
end


AugmentedStar(template::Union{Vector{Float64}, Nothing}=nothing; name::String, vel_bounds::Tuple{<:Real, <:Real}, rv_abs::Real=0) = AugmentedStar(template, isnothing(template), name, Float64(rv_abs), Float64.(vel_bounds))


function build(star::AugmentedStar, λ::AbstractVector{<:Real}, params::Parameters, data::DataFrame)
    if !isnothing(star.template)
        bc_vel = metadata(data, "bc_vel")
        vel_star = params["vel_star"]
        βstar = vel_star / SPEED_OF_LIGHT_MPS
        βbc = bc_vel / SPEED_OF_LIGHT_MPS
        tstar = sqrt((1 + βstar) / (1 - βstar))
        tbc = sqrt((1 + βbc) / (1 - βbc))
        λ′ = λ .* tstar ./ tbc
        s = interp1d(λ′, star.template, λ, extrapolate=false)
    else
        s = ones(length(λ))
    end
    return s
end


function get_initial_params!(params::Parameters, star::AugmentedStar, data::DataFrame)
    if star.from_flat
        params["vel_star"] = (value=0, bounds=(0, 0))
    else
        vel0 = get_initial_value(star.rv_abs .+ star.vel_bounds)
        params["vel_star"] = (value=vel0, bounds=star.rv_abs .+ star.vel_bounds)
    end
end


function activate_star!(params::Parameters, star::AugmentedStar)
    vel0 = get_initial_value(star.vel_bounds)
    params["vel_star"] = (value=vel0, bounds=star.vel_bounds, vary=star.vel_bounds[1] != star.vel_bounds[2])
end