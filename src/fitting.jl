export fit_spectrum, fit_spectra

function fit_spectrum(data::DataFrame, model::SpectralForwardModel, params::Parameters, iteration::Int; mask_worst::Int=0, mask_edges::Int=10)

    # Param info
    ptest = deepcopy(params)
    pbest = deepcopy(params)
    varied_inds = findall(params.vary)
    n_varied_params = length(varied_inds)
    pnames = collect(keys(params.indices))
    lb = getindex.(params.bounds, 1)
    ub = getindex.(params.bounds, 2)
    lbv = getindex.(params.bounds[varied_inds], 1)
    ubv = getindex.(params.bounds[varied_inds], 2)

    # Good indices
    data_inds_good = findall(@. isfinite(data.spec) && isfinite(data.specerr) && (data.specerr > 0) && (data.spec > 0))

    # Check dof
    ν = length(data_inds_good) - n_varied_params - mask_worst - 2 * mask_edges
    if ν <= 0
        println("Could not fit $(basename(metadata(data, "filename"))), ν=$ν <= 0")
        return nothing
    end

    # Loss func wrapper
    loss_func = (x) -> begin
        ptest.values .= x
        try
            _, y, lsf_kernel = build(model, ptest, data)
            if !check_positive(model.lsf, lsf_kernel)
                return Inf
            end
            residuals = data.spec .- y
            loss = redchi2loss(residuals, data.specerr; mask_worst, mask_edges, n_params=n_varied_params)
            return loss
        catch e
            #Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
            if e isa InterruptException
                error("Hit InterruptException")
            end
            return Inf
        end
    end

    #Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

    # Fit
    nm_result = IterativeNelderMead.optimize(loss_func, params.values;
                        lower_bounds=lb, upper_bounds=ub, vary=params.vary,
                        options=(;ftol_rel=1E-8)
                    )

    # Set new vals
    pbest.values .= nm_result.pbest

    # Best model and residuals
    _, model_best, _ = build(model, pbest, data)
    residuals_best = data.spec .- model_best

    # Get final inds
    residuals_inds_good = findall(isfinite.(residuals_best) .&& isfinite.(data.specerr) .&& (data.specerr .> 0))
    lsq_fitting_inds = residuals_inds_good
    ptest = deepcopy(pbest)
    model_func = (_, p) -> begin
        ptest.values[varied_inds] .= p
        _, y, _ = build(model, ptest, data)
        return y[lsq_fitting_inds]
    end
    wt = 1 ./ data.specerr[lsq_fitting_inds].^2
    lsq_result = LsqFit.curve_fit(model_func, lsq_fitting_inds, data.spec[lsq_fitting_inds], pbest.values[varied_inds], lower=lbv, upper=ubv, maxIter=0)

    # Set errors
    pbest.errors[varied_inds] .= get_stderrors(lsq_result)

    # Current loss
    redχ2 = redchi2loss(residuals_best, data.specerr; mask_worst, mask_edges, n_params=n_varied_params)
    rms = rmsloss(residuals_best; mask_worst, mask_edges)
    
    # Collect results
    opt_result = (;pbest, redχ2, rms)

    # Return
    return opt_result
    
end

function fit_spectrum_wrapper(
        data::DataFrame, model::SpectralForwardModel, params::Parameters, iteration::Int, output_path::String;
        plots::Bool=true, fitting_kwargs::NamedTuple,
    )

    # Try to fit
    r = nothing
    try

        # Time the fit
        ti = time()
        
        # Fit
        r = fit_spectrum(data, model, params, iteration; fitting_kwargs...)
        
        # Print results
        println("Fit observation $(basename(metadata(data, "filename"))), Iteration $iteration, in $(round((time() - ti) / 60, digits=4)) min")
        println("redχ2 = $(round(r.redχ2, digits=4))")
        println("RMS = $(round(100 * r.rms, digits=4))%")
        println("Parameters:")
        println("$(r.pbest)")

    catch e
        if !(e isa InterruptException)
            @warn "Could not fit $(basename(metadata(data, "filename")))" exception=(e, catch_backtrace())
        else
            error("Hit InterruptException")
        end
    end

    # Plot
    if plots
        if !isnothing(r) && isfinite(r.redχ2)
            try
                plot_spectral_fit(data, model, r, iteration, output_path)
            catch e
                if !(e isa InterruptException)
                    @warn "Could not plot fit for $(basename(metadata(data, "filename")))" exception=(e, catch_backtrace())
                else
                    error("Hit InterruptException")
                end
            end
        end
    end

    # Return
    return r

end


function fit_spectra(
        data::Vector{DataFrame}, model::SpectralForwardModel, params0::Vector{Parameters},
        iteration::Int, output_path::String;
        parallel::Bool=true, plots::Bool=true, fitting_kwargs::NamedTuple=(;),
    )

    # Parallel fitting
    if parallel
        opt_results = pmap(zip(data, params0)) do (d, p0)
            fit_spectrum_wrapper(d, model, p0, iteration, output_path; plots, fitting_kwargs)
        end
    else
        opt_results = map(zip(data, params0)) do (d, p0)
            fit_spectrum_wrapper(d, model, p0, iteration, output_path; plots, fitting_kwargs)
        end
    end

    # Return all results
    return opt_results

end



function get_stderrors(fit::LsqFit.LsqFitResult; rtol::Real=NaN, atol::Real=0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    J = fit.jacobian
    good_rows = findall([@views all(isfinite.(J[i, :])) for i=1:size(J, 1)])
    J = J[good_rows, :]
    if isempty(fit.wt)

        loss = LsqFit.mse(fit)

        # compute the covariance matrix from the QR decomposition
        Q, R = qr(J)
        Rinv = inv(R)
        covar = Rinv * Rinv' * loss
    else
        covar = inv(J' * J)
    end
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars) / maximum(vars)
    if !isapprox(
        vratio,
        0.0,
        atol=atol,
        rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol,
    ) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end


function redchi2loss(
        residuals::AbstractArray{<:Real}, errors::AbstractArray{<:Real};
        mask_worst::Int=0, mask_edges::Int=0, n_params::Int=1
    )

    # Same shape
    @assert size(residuals) == size(errors)

    # Compute diffs2
    good = findall(@. isfinite(residuals) && isfinite(errors))
    norm_res2 = @views (residuals[good] ./ errors[good]).^2

    # As a vector
    norm_res2 = vec(norm_res2)

    # Remove edges
    if mask_edges > 0
        norm_res2 = @views norm_res2[mask_edges+1:end-mask_edges]
    end

    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(norm_res2)
        norm_res2 = @views norm_res2[ss[1:end-mask_worst]]
    end

    # Degrees of freedom
    n_good = length(norm_res2)
    ν = n_good - n_params

    # Ensure positive dof
    @assert ν > 0

    # Compute chi2
    redχ² = nansum(norm_res2) / ν

    # Return
    return redχ²

end


function rmsloss(
        residuals::AbstractArray{<:Real}, weights::Union{AbstractArray{<:Real}, Nothing}=nothing;
        mask_worst::Int=0, mask_edges::Int=0, n_params::Int=1
    )

    # Same shape
    if !isnothing(weights)
        @assert size(residuals) == size(errors)
    end

    # Compute diffs2
    if !isnothing(weights)
        good = findall(@. isfinite(residuals) && isfinite(weights) && (weights > 0))
        res2 = @views weights[good] .* residuals[good].^2
    else
        good = findall(isfinite.(residuals))
        res2 = @views residuals[good].^2
    end

    # As a vector
    res2 = vec(res2)

    # Remove edges
    if mask_edges > 0
        res2 = @views res2[mask_edges+1:end-mask_edges]
    end

    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(res2)
        res2 = @views res2[ss[1:end-mask_worst]]
    end

    # N good
    n_good = length(res2)

    # Compute chi2
    rms = sqrt(nansum(res2) / n_good)

    # Return
    return rms
end