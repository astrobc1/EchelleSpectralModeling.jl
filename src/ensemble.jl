using EchelleBase
using EchelleSpectralModeling

export SpectralRVEnsembleProblem, IterativeSpectralRVEnsembleProblem, get_init_parameters

"""
    SpectralRVEnsembleProblem{S}
Abstract type for an ensemble of 1d spectral data from a single spectrograph with the primary goal of generating radial velocities.
"""
abstract type SpectralRVEnsembleProblem{S} end

"""
    Primary container for iteratively generating RVs for a given dataset with a given model and objective function.
# Fields
- `data::Vector{SpecData1d{S}}` The vector of SpecData1d objects.
- `model::AbstractSpectralForwardModel` The spectral forward model to use.
- `obj::SpectralModelObjectiveFunction` The objective function to use.
"""
struct IterativeSpectralRVEnsembleProblem{S, M<:AbstractSpectralForwardModel, O<:SpectralModelObjectiveFunction, T<:TemplateAugmenter} <: SpectralRVEnsembleProblem{S}
    data::Vector{SpecData1d{S}}
    model::M
    obj::O
    augmenter::T
end

"""
    Base.length(ensemble::IterativeSpectralRVEnsembleProblem)
The length of the ensemble is the number of observations.
"""
Base.length(ensemble::IterativeSpectralRVEnsembleProblem) = length(ensemble.data)

"""
    SpectralData.get_spectrograph(ensemble::IterativeSpectralRVEnsembleProblem)
Gets the spectrograph as a string for this ensemble.
"""
SpectralData.get_spectrograph(ensemble::IterativeSpectralRVEnsembleProblem) = String(typeof(ensemble).parameters[1])

"""
    IterativeSpectralRVEnsembleProblem(;spectrograph::String, data_input_path::String, filelist::String, model::AbstractSpectralForwardModel, obj::SpectralModelObjectiveFunction, augmenter::TemplateAugmenter)
    IterativeSpectralRVEnsembleProblem(;spectrograph::String, filenames::AbstractVector{String}, model::AbstractSpectralForwardModel, obj::SpectralModelObjectiveFunction, augmenter::TemplateAugmenter)
Construct an IterativeSpectralRVEnsembleProblem object.
"""
function IterativeSpectralRVEnsembleProblem(spectrograph::String, data_input_path::String, filelist::String, model::AbstractSpectralForwardModel, obj::SpectralModelObjectiveFunction, augmenter::TemplateAugmenter)
    if string(data_input_path[end]) != Base.Filesystem.path_separator
        data_input_path *= Base.Filesystem.path_separator
    end
    data = [SpecData1d(data_input_path * fname, spectrograph, model.sregion) for fname ∈ eachline(data_input_path * filelist) if !startswith(fname, "#")]
    jds = [parse_exposure_start_time(d) for d ∈ data]
    ss = sortperm(jds)
    data .= data[ss]
    return IterativeSpectralRVEnsembleProblem(data, model, obj, augmenter)
end

function IterativeSpectralRVEnsembleProblem(spectrograph::String, filenames::AbstractVector{String}, model::AbstractSpectralForwardModel, obj::SpectralModelObjectiveFunction, augmenter::TemplateAugmenter)
    data = [SpecData1d(fname, spectrograph, model.sregion) for fname ∈ filenames]
    jds = [parse_exposure_start_time(d) for d ∈ data]
    ss = sortperm(jds)
    data .= data[ss]
    return IterativeSpectralRVEnsembleProblem(data, model, obj, augmenter)
end

"""
    get_init_parameters(ensemble::IterativeSpectralRVEnsembleProblem)
Gets the initial parameters for all observations.
"""
function get_init_parameters(ensemble::IterativeSpectralRVEnsembleProblem)

    # Initial params
    p0s = [get_init_parameters(ensemble.model, d) for d ∈ ensemble.data]

    # Check which tellurics we need
    if !isnothing(ensemble.model.tellurics) && ensemble.model.tellurics isa TAPASTellurics
        if !isnothing(ensemble.model.lsf)
            kernel = build(ensemble.model.lsf, p0s[1], ensemble.model.templates)
            use_water = has_water_features(ensemble.model.tellurics, ensemble.model.templates, kernel)
            use_airmass = has_airmass_features(ensemble.model.tellurics, ensemble.model.templates, kernel)
        else
            use_water = has_water_features(ensemble.model.tellurics, ensemble.model.templates)
            use_airmass = has_airmass_features(ensemble.model.tellurics, ensemble.model.templates)
        end
        if ~use_water
            for p0 ∈ p0s
                p0["τ_water"] = Parameter(value=1, lower_bound=1, upper_bound=1)
            end
        end
        if ~use_airmass
            for p0 ∈ p0s
                p0["τ_airmass"] = Parameter(value=1, lower_bound=1, upper_bound=1)
            end
        end
        if ~use_water && ~use_airmass
            for p0 ∈ p0s
                p0["vel_tel"] = Parameter(value=0, lower_bound=0, upper_bound=0)
            end
        end
    end

    # Return
    return p0s
end

"""
    load_templates!(ensemble::IterativeSpectralRVEnsembleProblem)
Loads in the necessary templates for this ensemble.
"""
load_templates!(ensemble::IterativeSpectralRVEnsembleProblem) = load_templates!(ensemble.model, ensemble.data)