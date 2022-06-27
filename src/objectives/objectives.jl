module SpectralModelObjectiveFunctions

abstract type SpectralModelObjectiveFunction end

export SpectralModelObjectiveFunction, compute_obj

"""
    compute_obj
Computes the objective function. Must be implemented.
"""
function compute_obj end

(obj::SpectralModelObjectiveFunction)(pars, data, model) = compute_obj(obj, pars, data, model)

include("rms.jl")
include("chi2.jl")

end