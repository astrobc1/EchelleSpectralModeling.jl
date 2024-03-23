export GasCell

struct GasCell
    filename::String
    τ_bounds::Vector{Float64}
end

function GasCell(;filename::String, τ_bounds::Vector{<:Real}=[1, 1])
    return GasCell(filename, Float64.(τ_bounds))
end

function build(::GasCell, templates::Dict{String, <:Any}, params::Parameters, ::DataFrame)
    return templates["gascell"] .^ params["τ_gascell"]
end

function initialize!(gascell::GasCell, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    λ_raw, s = jldopen(gascell.filename) do f
        f["wave"], f["spec"]
    end
    good = findall(templates["λ"][1] - 1 .<= λ_raw .<= templates["λ"][end] + 1)
    s = interp1d(λ_raw[good], s[good], templates["λ"])
    s ./= nanmaximum(s)
    templates["gascell"] = s
    τ0 = gascell.τ_bounds[1] + 0.55 * (gascell.τ_bounds[2] - gascell.τ_bounds[1])
    for i in eachindex(data)
        params[i]["τ_gascell"] = (value=τ0, lower_bound=gascell.τ_bounds[1], upper_bound=gascell.τ_bounds[1])
    end
end