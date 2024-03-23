export PolyContinuum


struct PolyContinuum
    deg::Int
    bounds::Vector{Float64}
end

PolyContinuum(;deg::Int, bounds::Vector{<:Real}) = PolyContinuum(deg, Float64.(bounds))

function build(continuum::PolyContinuum, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    deg = length(templates["cont_knots_λs"]) - 1
    knots_spec = [params.values[params.indices["c$i"]] for i=1:continuum.deg+1]
    pfit = Polynomials.fit(ArnoldiFit, templates["cont_knots_λs"], knots_spec, deg)
    y = pfit.(templates["λ"])
    return y
end


function initialize!(continuum::PolyContinuum, templates::Dict{String, <:Any}, params::Vector{Parameters}, data::Vector{DataFrame})
    templates["cont_knots_λs"] = get_λ_knots(continuum, templates["λ"])
    c0 = (continuum.bounds[1] + continuum.bounds[2]) / 2
    for i in eachindex(data)
        for j=1:continuum.deg+1
            params[i]["c$j"] = (value=c0, lower_bound=continuum.bounds[1], upper_bound=continuum.bounds[2])
        end
    end
    return params
end

function get_λ_knots(continuum::PolyContinuum, λ::Vector{Float64})
    λi, λf = λ[1], λ[end]
    if continuum.deg > 0
        knots_λs = range(λi, stop=λf, length=continuum.deg + 1)
    else
        knots_λs = Float64[(λi + λf) / 2]
    end
    return knots_λs
end