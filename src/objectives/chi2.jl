
export RMS

struct RMS <: SpectralModelObjectiveFunction
end

function compute_obj(pars, model::SpectralModel, data::SpecData1d)
    model_flux = build(model, pars)
    data_flux = data.data.flux
    data_fluxerr = data.data.fluxerr
    data_mask = data.data.mask
    return compute_obj(model_flux, data_flux, data_fluxerr, data_mask)
end

function compute_obj(obj::RMS, model_flux, data_flux, data_fluxerr, data_mask)
    n_good = findall(sum(data_mask))
    n_flag = Int(round(0.02 * n_good))
    rms = maths.rmsloss(data.flux, flux_model, weights=data_mask, flag_worst=n_flag, remove_edges=4)
    return rms
end