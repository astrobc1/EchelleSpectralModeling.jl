export bin_rvs_single_order, bin_jds, combine_rvs, align_chunks

"""
"""
function combine_rvs(
        bjds::Vector{Float64}, rvs::Matrix{Float64},
        weights::Matrix{Float64},
        indices::Vector{<:AbstractVector{Int}},
        rverrs::Union{Nothing, Matrix{Float64}}=nothing;
        max_iterations::Int=10, nσ::Real=5
    )
    rvs = copy(rvs)
    weights = copy(weights)
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)
    norm_residuals = zeros(n_chunks, n_spec)
    bad = findall(.~isfinite.(rvs) .|| .~isfinite.(weights) .|| (weights .== 0))
    rvs[bad] .= NaN
    weights[bad] .= 0
    rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out = combine_relative_rvs(bjds, rvs, weights, indices, rverrs)
    for i=1:max_iterations
        println("Combining RVs, Iteration $i")
        rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out = combine_relative_rvs(bjds, rvs, weights, indices, rverrs)
        rvsa = align_chunks(rvs, weights)
        for j=1:n_bins
            inds = indices[j]
            norm_residuals[:, inds] .= (rvsa[:, inds] .- rvs_binned_out[j]) ./ unc_single_out[inds][:, :]'
        end
        good = findall(isfinite.(norm_residuals))
        rms = sqrt(sum(norm_residuals[good].^2) / length(good))
        bad = findall(isfinite.(norm_residuals) .&& abs.(norm_residuals) .> rms * nσ)
        weights[bad] .= 0
        rvs[bad] .= NaN
        if length(bad) == 0
            break
        end
    end
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
end

function combine_relative_rvs(
        bjds::Vector{Float64}, rvs::Matrix{Float64},
        weights::Matrix{Float64}, indices::Vector{<:AbstractVector{Int}},
        rverrs::Union{Nothing, Matrix{Float64}}=nothing
    )

    rvs = copy(rvs)
    weights = copy(weights)
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)
    bad = findall(.~isfinite.(rvs) .|| .~isfinite.(weights) .|| (weights .== 0))
    rvs[bad] .= NaN
    weights[bad] .= 0
    bjds_matrix = collect(transpose(repeat(bjds, n_spec, n_chunks)))

    # Numbers
    n_chunks, n_spec = size(rvs)
    n_bins = length(indices)

    # Align chunks
    rvsa = align_chunks(rvs, weights)
    bad = findall(.~isfinite.(rvsa) .|| .~isfinite.(weights) .|| (weights .== 0))
    rvsa[bad] .= NaN
    weights[bad] .= 0
    
    # Output arrays
    rvs_single_out = fill(NaN, n_spec)
    unc_single_out = fill(NaN, n_spec)
    t_binned_out = fill(NaN, n_bins)
    rvs_binned_out = fill(NaN, n_bins)
    unc_binned_out = fill(NaN, n_bins)
        
    # Per-observation RVs
    for i=1:n_spec
        rr, ww = rvsa[:, i], weights[:, i]
        good = findall(ww .> 0)
        n_good = length(good)
        if n_good > 0
            rvs_single_out[i] = @views weighted_mean(rr[good], ww[good])
        end
        if n_good == 1 && !isnothing(rverrs)
            unc_single_out[i] = rverrs[good[1]]
        elseif n_good > 1
            @views unc_single_out[i] = weighted_stddev(rr[good], ww[good]) / sqrt(n_good - 1)
        end
    end
        
    # Per-night RVs
    for i=1:n_bins
        inds = indices[i]
        rr = rvsa[:, inds][:]
        ww = weights[:, inds][:]
        good = findall(ww .> 0)
        n_good = length(good)
        if n_good > 0
            rvs_binned_out[i] = @views weighted_mean(rr[good], ww[good])
        end
        if n_good == 1 && !isnothing(rverrs)
            unc_binned_out[i] = rverrs[good[1]]
        elseif n_good > 1
            @views unc_binned_out[i] = weighted_stddev(rr[good], ww[good]) / sqrt(n_good - 1)
        end
        t_binned_out[i] = nanmean(bjds_matrix[:, inds][:][good])
    end
    
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
end


function bin_rvs_single_order(
        bjds::Vector{Float64}, rvs::Vector{Float64},
        weights::Vector{Float64},
        indices::Vector{<:AbstractVector{Int}}
    )

    # The number of spectra and nights
    n_bins = length(indices)
    
    # Initialize the binned rvs and uncertainties
    t_binned = fill(NaN, n_bins)
    rvs_binned = fill(NaN, n_bins)
    unc_binned = fill(NaN, n_bins)
    
    # Bin
    for i=1:n_bins
        inds = indices[i]
        rr = rvs[inds]
        ww = weights[inds]
        good = findall(ww .> 0)
        if length(good) > 0
            rvs_binned[i] = weighted_mean(rr, ww)
        end
        if length(good) > 1
            unc_binned[i] = weighted_stddev(rr, ww) / sqrt(length(good))
        end
        t_binned[i] = nanmean(bjds[inds][good])
    end

    return t_binned, rvs_binned, unc_binned
end


function align_chunks(rvs::Matrix{Float64}, weights::Matrix{Float64})
    n_chunks, n_spec = size(rvs)
    rvsa = fill(NaN, (n_chunks, n_spec))
    for l=1:n_chunks
        rvsa[l, :] .= rvs[l, :] .- weighted_mean(rvs[l, :], weights[l, :])
    end
    return rvsa
end


function bin_jds(jds::Vector{Float64}; sep::Real=0.5, utc_offset::Union{Int, Nothing}=nothing)
    
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
            if !isnothing(utc_offset)
                t_noon = ceil(jds[i] + utc_offset / 24) - utc_offset / 24
                if jds[i+1] > t_noon || jds[i+1] - jds[i] > sep
                    jd_avg = nanmean(jds[prev_i:i])
                    push!(jds_binned, jd_avg)
                    push!(indices_binned, prev_i:i)
                    prev_i = i + 1
                end
            elseif jds[i+1] - jds[i] > sep
                jd_avg = nanmean(jds[prev_i:i])
                push!(jds_binned, jd_avg)
                push!(indices_binned, prev_i:i)
                prev_i = i + 1
            end
        end
        push!(jds_binned, nanmean(jds[prev_i:end]))
        push!(indices_binned, prev_i:(n_obs_tot - 1))
    end

    return jds_binned, indices_binned
end

function weighted_mean(x, w)
    good = findall(isfinite.(x) .&& isfinite.(w) .&& (w .> 0))
    if length(good) == 0
        return NaN
    else
        return @views sum(x[good] .* w[good]) ./ sum(w[good])
    end
end