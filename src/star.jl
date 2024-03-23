export AugmentedStar

struct AugmentedStar{F}
    filename::F
    vel_bounds::Vector{Float64}
    rv_abs::Float64
    name::String
end

function AugmentedStar(;filename::Union{String, Nothing}=nothing, vel_bounds::Vector{<:Real}, rv_abs::Real=0, name::String)
    return AugmentedStar(filename, Float64.(vel_bounds), Float64(rv_abs), name)
end

function build(::AugmentedStar, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    bc_vel = metadata(data, "bc_vel")
    vel_star = params["vel_star"]
    zstar = vel_star / SPEED_OF_LIGHT_MPS
    zbc = bc_vel / SPEED_OF_LIGHT_MPS
    tstar = sqrt((1 + zstar) / (1 - zstar))
    tbc = sqrt((1 + zbc) / (1 - zbc))
    λ′ = @. templates["λ"] * tstar / tbc
    s = interp1d(λ′, templates["star"], templates["λ"], extrapolate=false)
    return s
end


function initialize!(star::AugmentedStar, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    if !isnothing(star.filename)
        d = readdlm(star.filename, ',', comments=true, comment_char='#')
        λ_raw, s = @views d[:, 1], d[:, 2]
        good = findall(templates["λ"][1] - 1 .<= λ_raw .<= templates["λ"][end] + 1)
        s = interp1d(λ_raw[good], s[good], templates["λ"], extrapolate=false)
        s ./= nanmaximum(s)
        templates["star"] = s
        if star.rv_abs == 0
            vel0 = star.vel_bounds[1] + 0.51 * (star.vel_bounds[2] - star.vel_bounds[1])
            lb, ub = star.vel_bounds
        else
            vel0 = star.rv_abs
            lb, ub = vel0 .+ star.vel_bounds
        end
        for i in eachindex(data)
            push!(params[i]; name="vel_star", value=vel0, lower_bound=lb, upper_bound=ub)
        end
    else
        templates["star"] = ones(length(templates["λ"]))
        vel0 = star.vel_bounds[1] + 0.501 * (star.vel_bounds[2] - star.vel_bounds[1])
        for i in eachindex(data)
            params[i]["vel_star"] = (;value=vel0, lower_bound=star.vel_bounds[1], upper_bound=star.vel_bounds[2], vary=false)
        end
    end
end