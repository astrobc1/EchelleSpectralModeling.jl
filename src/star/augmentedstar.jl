using EchelleBase
using EchelleSpectralModeling

export AugmentedStar

struct AugmentedStar <: SpectralModelComponent
    input_file::Union{String, Nothing}
    vel_bounds::Vector{Float64}
    star_name::String
    absolute_rv_guess::Float64
end

function AugmentedStar(;input_file, vel_bounds=[-5000, 5000], star_name, absolute_rv_guess=0.0)
    return AugmentedStar(input_file, vel_bounds, star_name, absolute_rv_guess)
end

function EchelleSpectralModeling.build(m::AugmentedStar, pars, templates)
    return build(m, templates["λ"], templates["star"], pars["vel_star"].value)
end

function EchelleSpectralModeling.build(m::AugmentedStar, λ, flux, vel)
    return maths.doppler_shift_flux(λ, flux, vel)
end

function EchelleSpectralModeling.load_template(m::AugmentedStar, λ_out)
    if !isnothing(m.input_file)
        return load_template(m.input_file, λ_out)
    else
        return ones(length(λ_out))
    end
end

function EchelleSpectralModeling.get_init_parameters(m::AugmentedStar, data, sregion)
    pars = Parameters()
    if !isnothing(m.input_file)
        v = m.absolute_rv_guess - data.header["bc_vel"]
    else
        v = -1 * data.header["bc_vel"]
    end
    pname = "vel_star"
    pars[pname] = Parameter(value=v, lower_bound=v + m.vel_bounds[1], upper_bound=v + m.vel_bounds[2])
    return pars
end


