using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using Infiltrator

export Chi2

"""
    Container (empty) for a Chi2 objective.
"""
struct Chi2 <: SpectralModelObjectiveFunction
end

"""
    compute_obj(obj::Chi2, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
Computes the Reduced-χ2 objective. The worst 2% of pixels and 4 edge pixels on each side are masked.
"""
function compute_obj(obj::Chi2, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
    try
        _, model_flux = build(model, pars, data)
        n_good = sum(data.data.mask)
        n_flag = Int(round(0.02 * n_good))
        ν = n_good - num_varied(pars)
        redχ² = maths.redχ2loss(data.data.flux .- model_flux, data.data.fluxerr, data.data.mask, flag_worst=n_flag, remove_edges=4, ν=ν)
        return redχ²
    catch
        return 1000
    end
end