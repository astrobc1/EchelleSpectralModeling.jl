export read_btsettl

function air2vac(λ)
    ref_index = pyimport("ref_index")
    λ′ = ref_index.air2vac(λ)
    return λ′
end

# Broadening
function doppler_broaden(λ, spec; vsini, ϵ=0.6)
    pya = pyimport("PyAstronomy.pyasl")
    if vsini > 0
        return pya.fastRotBroad(collect(λ) .* 10, spec, ϵ, vsini=vsini)
    else
        return copy(spec)
    end
end

# One way to correct repeated values
function correct_repeated_vals(λ, spec)
    λ_unq = unique(λ)
    n_unq = length(λ_unq)
    spec_unq = fill(NaN, n_unq)
    for i=1:n_unq
        inds = findall(λ .== λ_unq[i])
        spec_unq[i] = nanmean(spec[inds])
    end
    return λ_unq, spec_unq
end


function read_btsettl(filename::String, λ_out::Union{AbstractVector{<:Real}, Nothing}; vsini::Real=0, ϵ::Real=0.6, q::Union{Real, Nothing}=1)

    println("Reading $filename")

    # Read template
    template = readdlm(filename, comments=true, comment_char='#')
    λ, spec = template[:, 1], template[:, 2]

    # Angstroms to nm
    λ ./= 10

    # BT Settl wavelengths are in air
    λ .= air2vac(λ)

    # Interpolate
    if !isnothing(λ_out)
        good = findall(@. λ_out[1] - 1 <= λ <= λ_out[end] + 1)
        λ, spec = λ[good], spec[good]
        λ, spec = correct_repeated_vals(λ, spec)
        spec = DataInterpolations.CubicSpline(spec, λ).(λ_out)
        λ = λ_out
    else
        λ, spec = correct_repeated_vals(λ, spec)
    end

    # Doppler broaden
    if vsini > 0
        spec .= doppler_broaden(λ, spec; vsini, ϵ)
    end

    # Parse template params
    l = read_nth_line(filename, 2)
    teff = parse(Float64, split(split(l, "= ")[2], "K")[1])

    l = read_nth_line(filename, 3)
    logg = parse(Float64, split(split(l, "= ")[2], "log(")[1])

    l = read_nth_line(filename, 4)
    feh = parse(Float64, split(split(l, "= ")[2], "(value for the Metallicity")[1])

    l = read_nth_line(filename, 5)
    α = parse(Float64, split(split(l, "= ")[2], "(Alpha enhancement)")[1])

    # Normalize
    if !isnothing(q)
        spec ./= nanquantile(spec, q)
    end

    return spec, (;teff, feh, logg, vsini, ϵ, α)

end


function read_nth_line(filename::String, n::Int)
    for (i, l) in enumerate(eachline(filename))
        if i == n
            return l
        end
    end
    return nothing
end