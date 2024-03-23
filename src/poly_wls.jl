export PolyλSolution


struct PolyλSolution
    deg::Int
    bounds::Vector{Float64}
end

PolyλSolution(;deg::Int, bounds::Vector{<:Real}) = PolyλSolution(deg, Float64.(bounds))

function build(λsolution::PolyλSolution, templates::Dict{String, <:Any}, params::Parameters, data::DataFrame)
    deg = length(templates["λsol_knots_xs"]) - 1
    knots_λs = [params["λ$i"] for i=1:λsolution.deg+1]
    pfit = Polynomials.fit(ArnoldiFit, templates["λsol_knots_xs"], knots_λs, deg)
    y = pfit.(1:length(data.spec))
    return y
end

function initialize!(λsolution::PolyλSolution, templates::Dict{String, <:Any},  params::Vector{Parameters}, data::Vector{DataFrame})
    templates["λsol_knots_xs"] = get_pixel_knots(λsolution, data)
    for i in eachindex(data)
        knots_λ0 = data[i].λ[templates["λsol_knots_xs"]]
        for j=1:λsolution.deg+1
            params[i]["λ$j"] = (value=knots_λ0[j], lower_bound=knots_λ0[j] + λsolution.bounds[1], upper_bound=upper_bound=knots_λ0[j] + λsolution.bounds[2])
        end
    end
    return params
end

function get_pixel_knots(λsolution::PolyλSolution, data::Vector{DataFrame})
    xi, xf = get_data_pixel_bounds(data)
    return Int.(round.(collect(range(xi, stop=xf, length=λsolution.deg + 1))))
end