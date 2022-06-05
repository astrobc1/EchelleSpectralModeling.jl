using EchelleBase
using IterativeNelderMead
using EchelleSpectralModeling
using Infiltrator

export RMS

struct RMS <: SpectralModelObjectiveFunction
end

function compute_obj(obj::RMS, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
    try
        _, model_flux = build(model, pars, data)
        n_good = sum(data.data.mask)
        n_flag = Int(round(0.02 * n_good))
        rms = maths.rmsloss(data.data.flux .- model_flux, data.data.mask, flag_worst=n_flag, remove_edges=4)
        return rms
    catch
        return 100
    end
end