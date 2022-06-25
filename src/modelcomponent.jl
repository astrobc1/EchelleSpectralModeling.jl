using DelimitedFiles
using CurveFitParameters

export SpectralModelComponent, load_template, build, get_init_parameters

"""
    SpectralModelComponent
A base type for spectral model components to be used with SpectralForwardModel.
"""
abstract type SpectralModelComponent end

"""
    get_init_parameters(m::SpectralModelComponent, args...)
Primary method to get the initial parameters for this model component. Must be implemented.
"""
get_init_parameters(m::SpectralModelComponent, args...) = Parameters()

"""
    load_template(m::SpectralModelComponent, λ_out)
    load_template(fname::String, λ_out)
Loads the template for this component. By default, the template is assumed to be stored ina  2-column csv file with comments starting with #.
"""
function load_template(m::SpectralModelComponent, λ_out)
    return load_template(m.input_file, λ_out)
end

function load_template(fname::String, λ_out)
    f = readdlm(fname, ',', comments=true)
    λ, flux = f[:, 1], f[:, 2]
    λi, λf = λ_out[1], λ_out[end]
    good = findall((λ .> λi) .& (λ .< λf))
    flux_out = maths.cspline_interp(λ[good], flux[good], λ_out)
    flux_out ./= maths.weighted_median(flux_out, p=0.999)
    return flux_out
end