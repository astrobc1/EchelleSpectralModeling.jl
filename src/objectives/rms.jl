using EchelleBase
using EchelleSpectralModeling

export RMS

struct RMS <: SpectralModelObjectiveFunction
end

function compute_obj(obj::RMS, pars::Parameters, data, model)
    try
        _, model_flux = build(model, pars, data)
        n_good = sum(data.data.mask)
        n_flag = Int(round(0.02 * n_good))
        rms = maths.rmsloss(data.data.flux, model_flux, data.data.mask, flag_worst=n_flag, remove_edges=4)
        return rms
    catch
        return 1E6
    end
end

function compute_obj(obj::RMS, pars::Vector, data, model)
    try
        _, model_flux = build(model, pars, data)
        good = findall(isfinite.(data.data.flux) .&& isfinite.(model_flux) .&& (data.data.mask .== 1))
        rms = sqrt(sum((data.data.flux[good] .- model_flux[good]).^2) / length(good))
        if !isfinite(rms)
            rms = 1.0
        end
        @show rms
        return rms
    catch
        return 1.0
    end
    #n_good = sum(data.data.mask)
    #n_flag = Int(round(0.02 * n_good))
    #rms = NaN
    #try
    #    #rms = maths.rmsloss(data.data.flux, model_flux, data.data.mask, flag_worst=n_flag, remove_edges=4)
    #catch
    #    rms = 1E6
    #end
    #_, model_flux = build(model, pars, data)
    
end