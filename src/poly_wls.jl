export PolyλSolution


struct PolyλSolution
    knots_pix::Vector{Int}
    bounds::NTuple{2, Float64}
end


function PolyλSolution(;pix_range::NTuple{2, Int}, deg::Int, bounds::Tuple{<:Real, <:Real})
    knots_pix = Int.(round.(collect(range(pix_range[1], stop=pix_range[2], length=deg + 1))))
    return PolyλSolution(knots_pix, Float64.(bounds))
end


function build(λsolution::PolyλSolution, params::Parameters, data::DataFrame)
    knots_λs = [params["λ$i"] for i=1:length(λsolution.knots_pix)]
    @assert length(knots_λs) == length(λsolution.knots_pix)
    deg = length(knots_λs) - 1
    pfit = Polynomials.fit(ArnoldiFit, λsolution.knots_pix, knots_λs, deg)
    y = pfit.(1:length(data.spec))
    return y
end


function get_initial_params!(params::Parameters, λsolution::PolyλSolution, data::DataFrame)
    knots_λ0 = data.λ[λsolution.knots_pix]
    for i=1:length(λsolution.knots_pix)
        params["λ$i"] = (value=knots_λ0[i], bounds=knots_λ0[i] .+ λsolution.bounds)
    end
end