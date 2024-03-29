export GasCell, read_gascell


struct GasCell
    template::Vector{Float64}
    τ_bounds::NTuple{2, Float64}
end


GasCell(template::Vector{<:Real}; τ_bounds::Tuple{<:Real, <:Real}=(1, 1)) = GasCell(Float64.(template), Float64.(τ_bounds))


function build(gascell::GasCell, λ::Vector{<:Real}, params::Parameters, ::DataFrame)
    return gascell.template .^ params["τ_gascell"]
end


function read_gascell(filename::String, λ_out::Vector{<:Real}; q::Real=1)
    println("Reading $filename")
    template = readdlm(filename, ',', comments=true, comment_char='#')
    gas_λ, gas_spec = template[:, 1], template[:, 2]
    good = findall(λ_out[1] - 1 .<= gas_λ .<= λ_out[end] + 1)
    gas_spec = interp1d(gas_λ[good], gas_spec[good], λ_out)
    if !isnothing(gas_spec)
        gas_spec ./= nanquantile(gas_spec, q)
    end
    return gas_spec
end


function get_initial_params!(params::Parameters, gascell::GasCell, data::DataFrame)
    τ0 = get_initial_value(gascell.τ_bounds)
    params["τ_gascell"] = (value=τ0, bounds=gascell.τ_bounds)
end