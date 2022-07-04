using FileIO
using JLD2
using DelimitedFiles
using NaNStatistics
using Infiltrator
using IterativeNelderMead
using CurveFitParameters
using Distributed

using EchelleBase
using EchelleSpectralModeling

export compute_rvs

const PATHSEP = Base.Filesystem.path_separator

"""
    compute_rvs(ensemble::IterativeSpectralRVEnsembleProblem; output_path, tag, n_iterations::Int, do_ccf::Bool, verbose::Bool=false)
Primary method to iteratively compute the RVs for an IterativeSpectralRVEnsembleProblem object.
"""
function compute_rvs(ensemble::IterativeSpectralRVEnsembleProblem; output_path, tag, n_iterations::Int, do_ccf::Bool, initial_star_from_data::Bool=false, continuum_poly_deg_estimate=nothing, verbose::Bool=true)

    # Start the main clock!
    time_start_main = time()

    # Update the output path
    output_path = output_path * get_spectrograph(ensemble) * "_" * tag * PATHSEP

    # Create output paths
    create_output_dirs(ensemble, output_path)

    # Init the rvs dictionary
    rvs = Dict{String, Any}()
    for d ∈ ensemble.data
        get_barycentric_corrections(d, star_name=ensemble.model.star.star_name)
    end
    rvs["bjds"] = [d.header["bjd"] for d ∈ ensemble.data]
    rvs["bc_vels"] = [d.header["bc_vel"] for d ∈ ensemble.data]
    rvs["rvsfwm"] = fill(NaN, (length(ensemble), n_iterations))

    # CCF
    if do_ccf
        rvs["rvsxc"] = fill(NaN, (length(ensemble), n_iterations))
        rvs["rvsxcerr"] = fill(NaN, (length(ensemble), n_iterations))
        rvs["bis"] = fill(NaN, (length(ensemble), n_iterations))
    end

    # Opt results (vector of vector of named tuples)
    opt_results = Vector{NamedTuple}[]

    # Load templates
    load_templates!(ensemble)

    # Get initial parameters
    p0s = get_init_parameters(ensemble)
    p0scp = deepcopy(p0s)

    # Stellar templates
    stellar_templates = zeros(length(ensemble.model.templates["λ"]), n_iterations + 1)
    stellar_templates[:, 1] .= ensemble.model.templates["λ"]
    stellar_templates[:, 2] .= ensemble.model.templates["star"]

    # If no stellar template provided, get estimate from data
    if isnothing(ensemble.model.star.input_file) && initial_star_from_data
        ensemble.model.templates["star"] = estimate_initial_stellar_template(ensemble.model, ensemble.data, p0s, continuum_poly_deg=continuum_poly_deg_estimate)
    end

    # Loop over iterations
    for iteration=1:n_iterations

        # Timer
        time_iter_start = time()
        println("Starting iteration $iteration [$(label(ensemble.model.sregion))]")

        # Flat stellar template
        if iteration == 1 && isnothing(ensemble.model.star.input_file) && ~initial_star_from_data

            # Fix stellar rv
            for p0 ∈ p0s
                p0["vel_star"] = Parameter(value=0, lower_bound=0, upper_bound=0)
            end
            
            # Fit all observations
            _opt_results = optimize_spectra(ensemble, p0s, iteration, output_path; verbose=verbose)
            push!(opt_results, _opt_results)
            
            # Augment the template
            if iteration < n_iterations
                augment_star!(ensemble.model, ensemble.data, _opt_results, ensemble.augmenter)
            end
        
        else

            # Starting parameters
            if iteration > 1
                p0s = [res.pbest for res ∈ opt_results[end]]
            end

            # Vary stellar rv
            if iteration == 2 && isnothing(ensemble.model.star.input_file) && ~initial_star_from_data
                for i=1:length(p0s)
                    p0s[i]["vel_star"] = p0scp[i]["vel_star"]
                end
            end

            # Run the fit for all spectra and do a cross correlation analysis as well.
            _opt_results = optimize_spectra(ensemble, p0s, iteration, output_path; verbose=verbose)
            push!(opt_results, _opt_results)

            # Cross correlation
            if do_ccf
                p0s = [res.pbest for res ∈ _opt_results]
                ccf_results = cross_correlate_spectra(ensemble, p0s)
                rvs["rvsxc"][:, iteration] .= [res[1] + d.header["bc_vel"] for (d, res) ∈ zip(ensemble.data, ccf_results)]
                rvs["rvsxcerr"][:, iteration] .= [res[2] for res ∈ ccf_results]
            end
        
            # Store and plot rvs
            rvs["rvsfwm"][:, iteration] .= [res.pbest["vel_star"].value + d.header["bc_vel"] for (d, res) ∈ zip(ensemble.data, _opt_results)]
            save_rvs(ensemble, rvs, output_path)
            plot_rvs(ensemble, rvs, iteration, output_path)

            # Print RV Diagnostics
            if length(ensemble) > 1
                rvσ = nanstd(rvs["rvsfwm"][:, iteration])
                println("  Stddev of all fwm RVs = $(round(rvσ, digits=4)) m/s")
            end
        end

        # Save forward model outputs each time
        save_ensemble(ensemble, output_path)
        save_opt_results(ensemble, output_path, opt_results)
        save_stellar_templates(ensemble, output_path, stellar_templates)

        # Augment the stellar template
        if iteration < n_iterations
            augment_star!(ensemble.model, ensemble.data, _opt_results, ensemble.augmenter)
            stellar_templates[:, iteration] .= ensemble.model.templates["star"]
        end

        println("Finished iteration $iteration, [$(label(ensemble.model.sregion))] in $(round((time() - time_iter_start) / 3600, digits=3)) hours")

    end

    # Save forward model outputs
    save_ensemble(ensemble, output_path)
    save_opt_results(ensemble, output_path, opt_results)
    save_rvs(ensemble, rvs, output_path)
    save_stellar_templates(ensemble, output_path, stellar_templates)
    
    # End print
    println("Completed $(label(ensemble.model.sregion)) in $(round((time() - time_start_main) / 3600, digits=3)) hours")

end

##########################
#### OPTIMIZE HELPERS ####
##########################

function optimize_spectra(ensemble::IterativeSpectralRVEnsembleProblem, p0s, iteration::Int, output_path::String; verbose::Bool)

    # Opt results (vector of named tuples)
    if nprocs() > 1
        opt_results = pmap(1:length(ensemble.data)) do i
            optimize_and_plot_spectrum(p0s[i], ensemble.data[i], ensemble.model, ensemble.obj, iteration, output_path)
        end
    else
        opt_results = map(1:length(ensemble.data)) do i
            optimize_and_plot_spectrum(p0s[i], ensemble.data[i], ensemble.model, ensemble.obj, iteration, output_path)
        end
    end

    # Return
    return opt_results
    
end


function optimize_and_plot_spectrum(p0, data, model, obj, iteration, output_path)
    opt_result = optimize_spectrum(p0, data, model, obj, iteration)
    try
        plot_spectrum_fit(data, model, opt_result.pbest, iteration, output_path)
    catch
        nothing
    end
    return opt_result
end


function optimize_spectrum(p0, data, model, obj, iteration; verbose=true)

    # Time the fit
    ti = time()

    # Parameters as vectors
    vecs = to_vecs(p0)

    # Wrapper
    ptest = deepcopy(p0)
    pbest = deepcopy(p0)
    obj_wrapper = (x) -> begin
        set_values!(ptest, x)
        return compute_obj(obj, ptest, data, model)
    end
    opt_result = (;pbest=pbest, fbest=NaN, fcalls=0, simplex=nothing, iteration=0)

    try
        _opt_result = IterativeNelderMead.optimize(obj_wrapper, vecs.values, IterativeNelderMead.IterativeNelderMeadOptimizer(), lower_bounds=vecs.lower_bounds, upper_bounds=vecs.upper_bounds, vary=vecs.vary)
        set_values!(pbest, _opt_result.pbest)
        opt_result = (;pbest=pbest, fbest=_opt_result.fbest, fcalls=_opt_result.fcalls, simplex=_opt_result.simplex, iteration=_opt_result.iteration)
    catch
        nothing
    end

    # Print results
    println("Fit observation $(data), Iteration $iteration, $(label(model.sregion)) in $(round((time() - ti) / 60, digits=3)) min")
    if verbose
        println("Objective = $(round(opt_result.fbest, digits=3))")
        println("Calls: $(opt_result.fcalls)")
        println("Parameters:")
        println("$(opt_result.pbest)")
    end

    # Return
    return opt_result

end

#############
#### CCF ####
#############

function cross_correlate_spectra(ensemble::IterativeSpectralRVEnsembleProblem, p0s; vel_window_coarse=200_000, vel_step_coarse=100, vel_step_fine=10, vel_window_fine=1000)

    # CCF in series or parallel
    if nprocs() > 1
        ccf_results = pmap(1:length(ensemble.data)) do i
            brute_force_ccf(ensemble.model, ensemble.data[i], p0s[i], vel_window_coarse=vel_window_coarse, vel_step_coarse=vel_step_coarse, vel_step_fine=vel_step_fine, vel_window_fine=vel_window_fine)
        end
    else
        ccf_results = map(1:length(ensemble.data)) do i
            brute_force_ccf(ensemble.model, ensemble.data[i], p0s[i], vel_window_coarse=vel_window_coarse, vel_step_coarse=vel_step_coarse, vel_step_fine=vel_step_fine, vel_window_fine=vel_window_fine)
        end
    end

    # Return
    return ccf_results
end

##############
#### SAVE ####
##############

function create_output_dirs(ensemble::IterativeSpectralRVEnsembleProblem, output_path)
    o_folder = label(ensemble.model.sregion) * PATHSEP
    mkpath(output_path)
    mkpath(output_path * o_folder)
    mkpath(output_path * o_folder * "Fits")
    mkpath(output_path * o_folder * "RVs")
    mkpath(output_path * o_folder * "Templates")
end

function save_ensemble(ensemble::IterativeSpectralRVEnsembleProblem, output_path)
    fname = output_path * "$(label(ensemble.model.sregion))" * PATHSEP * "ensemble_$(label(ensemble.model.sregion)).jld"
    @save fname ensemble
end

function save_rvs(ensemble::IterativeSpectralRVEnsembleProblem, rvs, output_path)
    l = label(ensemble.model.sregion)
    fname = output_path * l * PATHSEP * "RVs" * PATHSEP * "rvs_$l.jld"
    @save fname rvs
end

function save_opt_results(ensemble::IterativeSpectralRVEnsembleProblem, output_path, opt_results)
    fname = output_path * label(ensemble.model.sregion) * PATHSEP * "Fits" * PATHSEP * "optimization_results_$(label(ensemble.model.sregion)).jld"
    @save fname opt_results
end

function save_stellar_templates(ensemble::IterativeSpectralRVEnsembleProblem, output_path, stellar_templates)
    fname = output_path * label(ensemble.model.sregion) * PATHSEP * "Templates" * PATHSEP * "stellar_templates_$(label(ensemble.model.sregion)).txt"
    λ = ensemble.model.templates["λ"]
    writedlm(fname, [λ stellar_templates], ',')
end