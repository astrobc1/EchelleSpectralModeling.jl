using EchelleBase
using EchelleSpectralModeling

export SpectralRVEnsembleProblem, IterativeSpectralRVEnsembleProblem, get_init_parameters, load_templates!

abstract type SpectralRVEnsembleProblem end

struct IterativeSpectralRVEnsembleProblem{S, M, O} <: SpectralRVEnsembleProblem
    data::Vector{SpecData1d{S}}
    model::M
    obj::O
end

Base.length(ensemble::IterativeSpectralRVEnsembleProblem) = length(ensemble.data)
SpectralData.get_spectrograph(ensemble::IterativeSpectralRVEnsembleProblem) = String(typeof(ensemble).parameters[1])

function IterativeSpectralRVEnsembleProblem(;spectrograph, data_input_path, filelist, model, obj)
    if string(data_input_path[end]) != Base.Filesystem.path_separator
        data_input_path *= Base.Filesystem.path_separator
    end
    data = [SpecData1d(data_input_path * fname, spectrograph, model.sregion) for fname ∈ eachline(data_input_path * filelist)]
    jds = [parse_exposure_start_time(d) for d ∈ data]
    ss = sortperm(jds)
    data .= data[ss]
    return IterativeSpectralRVEnsembleProblem(data, model, obj)
end

function get_init_parameters(ensemble::IterativeSpectralRVEnsembleProblem)
    return [get_init_parameters(ensemble.model, d) for d ∈ ensemble.data]
end

load_templates(ensemble::IterativeSpectralRVEnsembleProblem) = load_templates(ensemble.model, ensemble.data)