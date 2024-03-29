export TAPASTellurics, read_tapas

struct TAPASTellurics
    templates::@NamedTuple{water::Vector{Float64}, dry::Vector{Float64}}
    vel_bounds::NTuple{2, Float64}
    τ_water_bounds::NTuple{2, Float64}
    τ_dry_bounds::NTuple{2, Float64}
end


TAPASTellurics(templates::@NamedTuple{water::Vector{Float64}, dry::Vector{Float64}}; vel_bounds::Tuple{<:Real, <:Real}=(0, 0), τ_water_bounds::Tuple{<:Real, <:Real}=(1, 1), τ_dry_bounds::Tuple{<:Real, <:Real}=(1, 1)) = TAPASTellurics(templates, Float64.(vel_bounds), Float64.(τ_water_bounds), Float64.(τ_dry_bounds))


function build(tellurics::TAPASTellurics, λ::AbstractVector{<:Real}, params::Parameters, data::DataFrame)
    s = tellurics.templates.water .^ params["τ_water"] .* tellurics.templates.dry .^ params["τ_dry"]
    vel = params["vel_tell"]
    if vel != 0
        λ′ = @. λ * (1 + vel / SPEED_OF_LIGHT_MPS)
        s = interp1d(λ′, s, λ; extrapolate=false)
    end
    return s
end


function read_tapas(filename::String, λ_out::AbstractVector{<:Real}; q::Union{Real, Nothing}=1)
    println("Reading $filename")
    file = jldopen(filename)
    λ_raw = file["wave"]
    good = findall(@. λ_out[1] - 1 <= λ_raw <= λ_out[end] + 1)
    water = interp1d(λ_raw[good], file["water"][good], λ_out, extrapolate=false)
    dry = ones(length(λ_out))
    for key in keys(file)
        if key ∉ ("water", "wave")
            dry .*= interp1d(λ_raw[good], file[key][good], λ_out, extrapolate=false)
        end
    end
    close(file)
    if !isnothing(q)
        water ./= nanquantile(water, q)
        dry ./= nanquantile(dry, q)
    end
    return (;water, dry)
end

function get_initial_params!(params::Parameters, tellurics::TAPASTellurics, data::DataFrame)
    vel0 = get_initial_value(tellurics.vel_bounds)
    τ_water0 = get_initial_value(tellurics.τ_water_bounds)
    τ_dry0 = get_initial_value(tellurics.τ_dry_bounds)
    params["vel_tell"] = (value=vel0, bounds=tellurics.vel_bounds)
    params["τ_water"] = (value=τ_water0, bounds=tellurics.τ_water_bounds)
    params["τ_dry"] = (value=τ_dry0, bounds=tellurics.τ_dry_bounds)
end