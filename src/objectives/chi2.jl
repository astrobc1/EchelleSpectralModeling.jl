using EchelleBase
using CurveFitParameters
using EchelleSpectralModeling
using Infiltrator

export Chi2

"""
    Container for a Chi2 objective.
"""
struct Chi2 <: SpectralModelObjectiveFunction
    flag_n_worst::Int
    remove_edges::Int
end

"""
    Chi2(;flag_n_worst=10, remove_edges=4)
Construct a Chi2 objective. The worst `flag_n_worst` pixels are ignored after having masked `remove_edges` on each side.
"""
Chi2(;flag_n_worst=10, remove_edges=4) = Chi2(flag_n_worst, remove_edges)

"""
    compute_obj(obj::Chi2, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
Computes the Reduced-χ2 objective.
"""
function compute_obj(obj::Chi2, pars::Parameters, data::SpecData1d, model::SpectralForwardModel)
    try
        _, model_flux = build(model, pars, data)
        n_good = sum(data.data.mask)
        ν = n_good - num_varied(pars)
        redχ² = maths.redχ2loss(data.data.flux .- model_flux, data.data.fluxerr, data.data.mask, flag_worst=obj.flag_n_worst, remove_edges=obj.remove_edges, ν=ν)
        return redχ²
    catch
        return 10000
    end
end