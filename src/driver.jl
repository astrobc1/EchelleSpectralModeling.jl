export drive

function drive(
        data::Vector{DataFrame}, model::SpectralForwardModel;
        n_iterations::Int, output_path::String, fitting_kwargs::NamedTuple=(;),
    )

    # Time
    ti_main = time()

    # Fix output path
    if output_path[end] != PATHSEP
        output_path *= PATHSEP
    end

    # Create dirs for outputs
    create_output_dirs(output_path)

    # Store all rvs in dict
    rvs = make_rvs_dict(data, n_iterations)

    # Initialize model
    params0 = initialize!(model, data)

    # Initialize model with the data
    params_best = deepcopy(params0)

    # Store all opt results in list opt_results[iteration][observation]
    opt_results = Vector{Any}(undef, n_iterations)

    # Store each stellar template
    if !isnothing(model.star)
        stellar_templates = Vector{Any}(undef, n_iterations)
    end

    # Loop over iterations
    for iteration in 1:n_iterations

        # Time
        ti = time()

        # Fit all spectra
        opt_results[iteration] = fit_spectra(data, model, params_best, iteration, output_path; plots=true, fitting_kwargs)

        # Update best fit params
        params_best = [!isnothing(r) ? r.pbest : nothing for r in opt_results[iteration]]

        # Compute, plot, and save RVs
        if !isnothing(model.star.filename) || iteration > 1
            compute_rvs_from_fits(rvs, opt_results[iteration], iteration)
            plot_rvs(model, rvs, iteration, output_path)
            save_rvs(output_path, rvs)
        end

        # Save opt results
        save_opt_results(output_path, opt_results)

        # Save data/model
        save_data_model(output_path, data, model)
        
        # Retrieve this stellar template
        if !isnothing(model.star)
            stellar_templates[iteration] = copy(model.templates["star"])
            save_stellar_templates(output_path, stellar_templates, model)
        end

        # Augment template if we will do another fit
        if iteration < n_iterations
            augment_stellar_template!(model, data, opt_results[iteration])
        end

        # Activate star
        if iteration == 1 && isnothing(model.star.filename)
            activate_star!(params_best)
        end

        println("Finished iteration $iteration of $n_iterations, $(split(output_path, PATHSEP)[end]), in $(round((time() - ti) / 3600, digits=4)) hours")
    end

    println("Finished $(split(output_path, PATHSEP)[end]) in $(round((time() - ti_main) / 3600, digits=4)) hours")

end



function compute_rvs_from_fits(rvs::Dict{String, Any}, opt_results::Vector, iteration::Int)
    rvs["rvsfwm"][:, iteration] = [!isnothing(r) ? r.pbest.values[r.pbest.indices["vel_star"]] : NaN for r in opt_results]
    rvs["rvsfwmerr"][:, iteration] = [!isnothing(r) ? r.pbest.errors[r.pbest.indices["vel_star"]] : NaN for r in opt_results]
    return rvs
end


function save_rvs(output_path::String, rvs::Dict{String, <:Any})

    # Save JLD
    fname = output_path * "RVs" * PATHSEP * "rvs.jld"
    jldsave(fname; rvs)

    # RVs in text file
    fname = output_path * "RVs" * PATHSEP * "rvs.txt"
    writedlm(fname, [rvs["bjds"] rvs["rvsfwm"]], ',')

    # RV errors in text file
    fname = output_path * "RVs" * PATHSEP * "rverrs.txt"
    writedlm(fname, [rvs["bjds"] rvs["rvsfwmerr"]], ',')

    # BJDs and bc vels in text file
    fname = output_path * "RVs" * PATHSEP * "bc_vels.txt"
    writedlm(fname, [rvs["bjds"] rvs["bc_vels"]], ',')

    # Return nothing
    return nothing
end

function save_opt_results(output_path::String, opt_results::Vector)
    fname = output_path * "optimization_results.jld"
    jldsave(fname; opt_results)
end


function create_output_dirs(output_path::String)
    mkpath("$(output_path)Fits")
    mkpath("$(output_path)RVs")
end


function make_rvs_dict(data::Vector{DataFrame}, n_iterations::Int)
    n_spec = length(data)
    rvs = Dict{String, Any}("rvsfwm" => fill(NaN, (n_spec, n_iterations)), "rvsfwmerr" => fill(NaN, (n_spec, n_iterations)))
    rvs["bjds"] = metadata.(data, "bjd")
    rvs["bc_vels"] = metadata.(data, "bc_vel")
    return rvs
end


function save_stellar_templates(output_path::String, stellar_templates::Vector{<:Any}, model::SpectralForwardModel)
    fname = output_path * "stellar_templates.jld"
    jldsave(fname; λ=model.templates["λ"], stellar_templates)
end


function activate_star!(params::Vector{Parameters})
    for i in eachindex(params)
        p = params[i]
        if p.lower_bounds[p.indices["vel_star"]] != p.upper_bounds[p.indices["vel_star"]]
            params[i].vary[params[i].indices["vel_star"]] = true
        end
    end
end

function save_data_model(output_path::String, data::Vector{DataFrame}, model::SpectralForwardModel)
    
    # Save data
    fname = output_path * "data.jld"
    jldsave(fname; data)

    # Save model
    fname = output_path * "model.jld"
    jldsave(fname; model)

end