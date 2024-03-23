export TAPASTellurics

struct TAPASTellurics
    filename::String
    vel_bounds::Vector{Float64}
    τ_water_bounds::Vector{Float64}
    τ_dry_bounds::Vector{Float64}
end

function TAPASTellurics(;filename::String, vel_bounds::Vector{<:Real}=[0, 0], τ_water_bounds::Vector{<:Real}=[1, 1], τ_dry_bounds::Vector{<:Real}=[1, 1])
    return TAPASTellurics(filename, Float64.(vel_bounds), Float64.(τ_water_bounds), Float64.(τ_dry_bounds))
end

function build(::TAPASTellurics, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    s = templates["tell_water"] .^ params["τ_water"] .* templates["tell_dry"] .^ params["τ_dry"]
    vel = params["vel_tell"]
    if vel != 0
        λ′ = templates["λ"] .* (1 .+ vel ./ SPEED_OF_LIGHT_MPS)
        s = interp1d(λ′, s, templates["λ"]; extrapolate=true)
    end
    return s
end

function initialize!(tellurics::TAPASTellurics, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    file = jldopen(tellurics.filename)
    λ_raw = file["wave"]
    good = findall(templates["λ"][1] - 1 .<= λ_raw .<= templates["λ"][end] + 1)
    s = interp1d(λ_raw[good], file["water"][good], templates["λ"], extrapolate=false)
    s ./= nanmaximum(s)
    templates["tell_water"] = s
    s = ones(length(templates["λ"]))
    for key in keys(file)
        if key ∉ ("water", "wave")
            s .*= interp1d(λ_raw[good], file[key][good], templates["λ"], extrapolate=false)
        end
    end
    s ./= nanmaximum(s)
    templates["tell_dry"] = s
    close(file)
    vel0 = nanmean(tellurics.vel_bounds)
    if vel0 == 0 && tellurics.vel_bounds[2] != tellurics.vel_bounds[1]
        vel0 = tellurics.vel_bounds[1] + 0.51 * (tellurics.vel_bounds[2] - tellurics.vel_bounds[1])
    end
    τ_water0 = nanmean(tellurics.τ_water_bounds)
    τ_dry0 = nanmean(tellurics.τ_dry_bounds)
    for i in eachindex(data)
        params[i]["vel_tell"] = (value=vel0, lower_bound=tellurics.vel_bounds[1], upper_bound=tellurics.vel_bounds[2])
        params[i]["τ_water"] = (value=τ_water0, lower_bound=tellurics.τ_water_bounds[1], upper_bound=tellurics.τ_water_bounds[2])
        params[i]["τ_dry"] = (value=τ_dry0, lower_bound=tellurics.τ_dry_bounds[1], upper_bound=tellurics.τ_dry_bounds[2])
    end
end