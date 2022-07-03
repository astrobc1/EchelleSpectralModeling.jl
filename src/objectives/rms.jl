using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using Infiltrator

export RMS

"""
    Container for a RMS objective.
"""
struct RMS <: SpectralModelObjectiveFunction
    flag_n_worst::Int
    remove_edges::Int
end

RMS(;flag_n_worst=10, remove_edges=4) = RMS(flag_n_worst, remove_edges)

"""
    compute_obj(obj::RMS, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
Computes the RMS objective.
"""
function compute_obj(obj::RMS, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
    try
        _, model_flux = build(model, pars, data)
        rms = maths.rmsloss(data.data.flux .- model_flux, data.data.mask, flag_worst=obj.flag_n_worst, remove_edges=obj.remove_edges)
        return rms
    catch
        return 1000
    end
end