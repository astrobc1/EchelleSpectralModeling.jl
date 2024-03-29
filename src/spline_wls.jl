export SplineλSolution


struct SplineλSolution
    knots_pix::Vector{Int}
    bounds::NTuple{2, Float64}
end


function SplineλSolution(;pix_range::NTuple{2, Int}, n_knots::Int, bounds::Tuple{<:Real, <:Real})
    knots_pix = Int.(round.(collect(range(pix_range[1], stop=pix_range[2], length=n_knots))))
    return SplineλSolution(knots_pix, Float64.(bounds))
end


function build(λsolution::SplineλSolution, params::Parameters, data::DataFrame)
    knots_λs = [params["λ$i"] for i=1:length(λsolution.knots_pix)]
    @assert length(knots_λs) == length(λsolution.knots_pix)
    y = DataInterpolations.CubicSpline(knots_λs, λsolution.knots_pix, extrapolate=true)(1:length(data.spec))
    return y
end


function get_initial_params!(params::Parameters, λsolution::SplineλSolution, data::DataFrame)
    knots_λ0 = data.λ[λsolution.knots_pix]
    for i=1:length(λsolution.knots_pix)
        params["λ$i"] = (value=knots_λ0[i], bounds=knots_λ0[i] .+ λsolution.bounds)
    end
end