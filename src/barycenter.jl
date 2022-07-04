using PyCall
using AstroTime
using EchelleBase

export compute_barycentric_corrections

"""
    compute_barycentric_corrections(data::SpecData, star_name=nothing, obs_name=nothing; store=true)
    compute_barycentric_corrections(jdmid::Float64, obs_name::String, star_name::String)
Compute the barycentric corrections using barycorrpy for the observation `data`. The `star_name` must be a recognized by simbad. If `store=true`, the variable data.header is updated with the BJD and velocity correction (bjd) and (bc_vel).
Returns the BJD and velocity correction as a tuple.
"""
function compute_barycentric_corrections(data::SpecData, star_name=nothing, obs_name=nothing; store=true)
    if isnothing(star_name)
        spec_mod = get_spec_module(data)
        star_name = parse_object(data)
    end
    star_name = replace(star_name, "_" => " ")
    jdmid = get_exposure_midpoint(data)

    if isnothing(obs_name)
        spec_mod = get_spec_module(data)
        obs_name = spec_mod.observatory
    end
    
    # BJD and BC vel
    bjd, bc_vel = compute_barycentric_corrections(jdmid, obs_name, star_name)
    if store
        data.header["bjd"] = bjd
        data.header["bc_vel"] = bc_vel
    end
    return bjd, bc_vel
end

function compute_barycentric_corrections(jdmid::Float64, obs_name::String, star_name::String)
    barycorrpy = pyimport("barycorrpy")
    bjd = barycorrpy.utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=obs_name, leap_update=true)[1][1]
    bc_vel = barycorrpy.get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=obs_name, leap_update=true)[1][1]
    return bjd, bc_vel
end