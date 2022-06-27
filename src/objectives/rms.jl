using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using Infiltrator

export RMS

"""
Container (empty) for an RMS objective.
"""
struct RMS <: SpectralModelObjectiveFunction
end

"""
    compute_obj(obj::RMS, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
Computes the RMS objective. The worst 2% of pixels and 4 edge pixels on each side are masked.
"""
function compute_obj(obj::RMS, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
    try
        _, model_flux = build(model, pars, data)
        n_good = sum(data.data.mask)
        n_flag = Int(round(0.02 * n_good)) # Flag 2% of pixels
        rms = maths.rmsloss(data.data.flux .- model_flux, data.data.mask, flag_worst=n_flag, remove_edges=4)
        return rms
    catch
        return 100
    end
end