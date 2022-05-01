using DataFrames
using CSV

export SpectralModelComponent, load_template, build, get_init_parameters

abstract type SpectralModelComponent end

get_init_parameters(m::SpectralModelComponent, args...) = Parameters()

function load_template(m::SpectralModelComponent, λ_out)
    return load_template(m.input_file, λ_out)
end

function load_template(fname, λ_out)
    df = DataFrame(CSV.File(fname, comment="#", header=false))
    λ, flux = df.Column1, df.Column2
    λi, λf = λ_out[1], λ_out[end]
    good = findall((λ .> λi) .& (λ .< λf))
    flux_out = maths.cspline_interp(λ[good], flux[good], λ_out)
    flux_out ./= maths.weighted_median(flux_out, p=0.999)
    return flux_out
end