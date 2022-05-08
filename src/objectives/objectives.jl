module SpectralModelObjectiveFunctions

abstract type SpectralModelObjectiveFunction end

export SpectralModelObjectiveFunction, compute_obj

function compute_obj end

(obj::SpectralModelObjectiveFunction)(pars, data, model) = compute_obj(obj, pars, data, model)

include("rms.jl")
include("chi2.jl")

end