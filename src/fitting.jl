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
function compute_rvs(ensemble::IterativeSpectralRVEnsembleProblem; output_path, tag, n_iterations::Int, do_ccf::Bool, verbose::Bool=false)

    # Start the main clock!
    time_start_main = time()

    # Update the output path
    output_path = output_path * get_spectrograph(ensemble) * "_" * tag * PATHSEP

    # Create output paths
    create_output_paths(ensemble, output_path)

    # Init the rvs dictionary
    rvs = Dict{String, Any}()
    for d ∈ ensemble.data
        get_barycentric_corrections(d, star_name=ensemble.model.star.star_name)
    end
    rvs["bjds"] = [d.header["bjd"] for d ∈ ensemble.data]
    rvs["bc_vels"] = [d.header["bc_vel"] for d ∈ ensemble.data]
    rvs["rvsfwm"] = fill(NaN, (length(ensemble), n_iterations))

    # Opt results (vector of vector of named tuples)
    opt_results = Vector{NamedTuple}[]

    # Get initial parameters
    p0s = get_init_parameters(ensemble)
    p0scp = deepcopy(p0s)

    # Load templates
    load_templates!(ensemble)

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
                p0["water_depth"] = Parameter(value=1, lower_bound=1, upper_bound=1)
            end
        end
        if ~use_airmass
            for p0 ∈ p0s
                p0["airmass_depth"] = Parameter(value=1, lower_bound=1, upper_bound=1)
            end
        end
        if ~use_water && ~use_airmass
            for p0 ∈ p0s
                p0["vel_tel"] = Parameter(value=0, lower_bound=0, upper_bound=0)
            end
        end
    end

    # Stellar templates
    stellar_templates = zeros(length(ensemble.model.templates["λ"]), n_iterations + 1)
    stellar_templates[:, 1] .= ensemble.model.templates["λ"]
    stellar_templates[:, 2] .= ensemble.model.templates["star"]

    # Iterate over remaining stellar template generations
    for iteration=1:n_iterations

        # Timer
        time_iter_start = time()
        println("Starting iteration $iteration [$(label(ensemble.model.sregion))]")
        
        # Flat stellar template
        if iteration == 1 && isnothing(ensemble.model.star.input_file)

            # Fix stellar rv
            for p0 ∈ p0s
                p0["vel_star"] = Parameter(value=0, lower_bound=0, upper_bound=0)
            end
            
            # Fit all observations
            _opt_results = optimize_all_observations(ensemble, p0s, iteration, output_path; verbose=verbose)
            push!(opt_results, _opt_results)
            
            # Augment the template
            if iteration < n_iterations
                augment_star!(ensemble, _opt_results)
            end
        
        else

            # Starting parameters
            if iteration > 1
                p0s = [res.pbest for res ∈ opt_results[end]]
            end

            # Fix stellar rv
            if iteration == 2 && isnothing(ensemble.model.star.input_file)
                for i=1:length(p0s)
                    p0s[i]["vel_star"] = p0scp[i]["vel_star"]
                end
            end

            # Run the fit for all spectra and do a cross correlation analysis as well.
            _opt_results = optimize_all_observations(ensemble, p0s, iteration, output_path; verbose=verbose)
            push!(opt_results, _opt_results)
        
            # Store and plot rvs
            rvs["rvsfwm"][:, iteration] = [res.pbest["vel_star"].value + d.header["bc_vel"] for (d, res) ∈ zip(ensemble.data, _opt_results)]
            save_rvs(ensemble, rvs, output_path)
            plot_rvs(ensemble, rvs, iteration, output_path)
        
            # Save forward model outputs each time
            save_ensemble(ensemble, output_path)
            save_opt_results(ensemble, output_path, opt_results)
            save_rvs(ensemble, rvs, output_path)
            save_stellar_templates(ensemble, output_path, stellar_templates)

            # Print RV Diagnostics
            if length(ensemble) > 1
                rvσ = nanstd(rvs["rvsfwm"][:, iteration])
                println("  Stddev of all fwm RVs = $(round(rvσ, digits=4)) m/s")
            end

            # Augment the templates
            if iteration < n_iterations
                augment_star!(ensemble, _opt_results)
                stellar_templates[:, iteration] .= ensemble.model.templates["star"]
            end

            println("Finished iteration $iteration, [$(label(ensemble.model.sregion))] in $(round((time() - time_iter_start) / 3600, digits=3)) hours")
        end
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

function optimize_all_observations(ensemble::IterativeSpectralRVEnsembleProblem, p0s, iteration::Int, output_path::String; verbose::Bool)
            
    # Timer
    ti = time()

    # Opt results (vector of named tuples)
    if nprocs() > 1
        opt_results = pmap(1:length(ensemble.data)) do i
            optimize_and_plot_observation(p0s[i], ensemble.data[i], ensemble.model, ensemble.obj, iteration, output_path)
        end
    else
        opt_results = map(1:length(ensemble.data)) do i
            optimize_and_plot_observation(p0s[i], ensemble.data[i], ensemble.model, ensemble.obj, iteration, output_path)
        end
    end


    println("Finished Iteration $(iteration) in $(round((time() - ti) / 60, sigdigits=3)) min")

    # Return
    return opt_results
    
end


function optimize_and_plot_observation(p0, data, model, obj, iteration, output_path)
    opt_result = optimize_observation(p0, data, model, obj, iteration)
    try
        plot_spectrum_fit(data, model, opt_result.pbest, iteration, output_path)
    catch
        nothing
    end
    return opt_result
end


function optimize_observation(p0, data, model, obj, iteration; verbose=true)

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
        println(" Objective = $(round(opt_result.fbest, digits=3))")
        println(" Calls: $(opt_result.fcalls)")
        println(" Parameters:")
        println(" $(opt_result.pbest)")
    end

    # Return
    return opt_result

end

##############
#### SAVE ####
##############

function create_output_paths(ensemble::IterativeSpectralRVEnsembleProblem, output_path)
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