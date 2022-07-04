export bin_rvs_single_order, bin_jds, combine_rvs


"""
    combine_rvs(bjds::Vector{Float64}, rvs::Matrix{Float64}, weights::Matrix{Float64}, indices; n_iterations=10, nσ=4)
Wrapper to iteratively combines RVs from different chunks or orders in a relative fashion. After each iterations, outliers are flagged according to their deviation from the corresponding binned value.
"""
function combine_rvs(bjds::Vector{Float64}, rvs::Matrix{Float64}, weights::Matrix{Float64}, indices; n_iterations=10, nσ=4)
    weights = copy(weights)
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)
    norm_residuals = zeros(n_chunks, n_spec)
    rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out = combine_relative_rvs(bjds, rvs, weights, indices)
    for i=1:n_iterations
        println("Iteration $i")
        rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out = combine_relative_rvs(bjds, rvs, weights, indices)
        rvli, wli = align_chunks(rvs, weights)
        for j=1:n_bins
            f, l = indices[j]
            norm_residuals[:, f:l] .= (rvli[:, f:l] .- rvs_binned_out[j]) ./ (1 ./ weights[:, f:l])
        end
        good = findall(isfinite.(norm_residuals))
        rms = sqrt(sum(norm_residuals[good].^2) / length(good))
        bad = findall(abs.(norm_residuals) .> rms * nσ)
        weights[bad] .= 0
        if length(bad) == 0
            break
        end
    end
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
end

"""
    combine_relative_rvs(bjds::Vector{Float64}, rvs::Matrix{Float64}, weights::Matrix{Float64}, indices)
Combines RVs from different chunks or orders in a relative fashion according to Rajpaul et al. 2020 (2020MNRAS.492.3960R).  This also bins RVs according to indices, a vector of vectors, where each sub-vector contains the starting and ending entry of a bin (both inclusive). Such a vector is returned by `bin_jds`.
"""
function combine_relative_rvs(bjds::Vector{Float64}, rvs::Matrix{Float64}, weights::Matrix{Float64}, indices)

    # Numbers
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)

    # Align chunks
    rvli, wli = align_chunks(rvs, weights)
    
    # Output arrays
    rvs_single_out = fill(NaN, n_spec)
    unc_single_out = fill(NaN, n_spec)
    t_binned_out = fill(NaN, n_bins)
    rvs_binned_out = fill(NaN, n_bins)
    unc_binned_out = fill(NaN, n_bins)
    bad = findall(.~isfinite.(wli))
    wli[bad] .= 0
        
    # Per-observation RVs
    for i=1:n_spec
        rvs_single_out[i] = maths.weighted_mean(rvli[:, i], wli[:, i])
        n_good = length(findall(wli[:, i] .> 0))
        if n_good > 0
            unc_single_out[i] = maths.weighted_stddev(rvli[:, i], wli[:, i]) / sqrt(n_good)
        end
    end
        
    # Per-night RVs
    for i=1:n_bins
        f, l = indices[i]
        rr = rvli[:, f:l][:]
        ww = wli[:, f:l][:]
        bad = findall(.~isfinite.(rr))
        ww[bad] .= 0
        good = findall(ww .> 0)
        rvs_binned_out[i] = maths.weighted_mean(rr, ww)
        if length(good) > 0
            unc_binned_out[i] = maths.weighted_stddev(rr, ww) / sqrt(length(good))
        end
        t_binned_out[i] = mean(bjds[f:l])
    end
    
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
end

function bin_rvs_single_order(rvs::Vector{Float64}, weights::Vector{Float64}, indices::Vector{Float64})

    # The number of spectra and nights
    n_spec = length(rvs)
    n_bins = len(indices)
    
    # Initialize the binned rvs and uncertainties
    rvs_binned = fill(n_bins, NaN)
    unc_binned = fill(n_bins, NaN)
    
    # Bin
    for i=1:n_bins
        f, l = indices[i]
        rr = @view rvs[f:l+1]
        ww = @view weights[f:l+1]
        rvs_binned[i], unc_binned[i] = maths.weighted_combine(rr, ww, yerr, err_type="empirical")
    end
            
    return rvs_binned, unc_binned
end

function align_chunks(rvs::Matrix{Float64}, weights::Matrix{Float64})

    n_chunks, n_spec = size(rvs)
    
    # Determine differences and weights tensors
    rvlij = fill(NaN, (n_chunks, n_spec, n_spec))
    wlij = fill(NaN, (n_chunks, n_spec, n_spec))
    wli = fill(NaN, (n_chunks, n_spec))
    for l=1:n_chunks
        for i=1:n_spec
            wli[l, i] = weights[l, i]
            for j=1:n_spec
                rvlij[l, i, j] = rvs[l, i] - rvs[l, j]
                wlij[l, i, j] = sqrt(weights[l, i] * weights[l, j])
            end
        end
    end

    # Average over differences
    rvli = fill(NaN, (n_chunks, n_spec))
    for l=1:n_chunks
        for i=1:n_spec
            rvli[l, i] = maths.weighted_mean(rvlij[l, i, :], wlij[l, i, :])
        end
    end

    return rvli, wli
end

"""
    bin_jds(jds::Vector{Float64}; sep=0.5, utc_offset=-8)
Bins JDs (or BJDs). Returns the binned JDs as well as a vector of vectors, where each sub-vector contains the starting/ending index of the bin (both inclusive) - e.g., `[[1, 5], [6, 8], [9, 17], [18, 21]]`.
"""
function bin_jds(jds::Vector{Float64}; sep=0.5, utc_offset=-8)
    
    # Number of spectra
    n_obs_tot = length(jds)

    # Keep track of previous night's last index
    prev_i = 1

    # Calculate mean JD date and number of observations per night for binned
    # Assume that observations are in separate bins if noon passes or if Δt > sep
    jds_binned = Float64[]
    indices_binned = Vector{Int64}[]
    if n_obs_tot == 1
        push!(jds_binned, jds[1])
        push!(indices_binned, [1, 1])
    else
        for i=1:n_obs_tot-1
            t_noon = ceil(jds[i] + utc_offset / 24) - utc_offset / 24
            if jds[i+1] > t_noon || jds[i+1] - jds[i] > sep
                jd_avg = mean(jds[prev_i:i])
                push!(jds_binned, jd_avg)
                push!(indices_binned, [prev_i, i])
                prev_i = i + 1
            end
        end
        push!(jds_binned, mean(jds[prev_i:end]))
        push!(indices_binned, [prev_i, n_obs_tot - 1])
    end

    return jds_binned, indices_binned
end