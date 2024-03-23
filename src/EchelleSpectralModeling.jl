module EchelleSpectralModeling

using LinearAlgebra
using JLD2, DelimitedFiles
using NaNStatistics, DataInterpolations
using DataFrames
using SavitzkyGolay
using LoopVectorization
using LsqFit
using PyPlot
using OrderedCollections
using Distributed
using Polynomials
using StatsBase, Statistics
using Infiltrator

using IterativeNelderMead, Echelle

include("parameters.jl")

include("lsf_utils.jl")
include("continuum_utils.jl")
include("doppler_shift.jl")

include("tapas.jl")
include("star.jl")
include("poly_wls.jl")
include("poly_continuum.jl")
include("static_wls.jl")
include("gascell.jl")
include("gausshermite_lsf.jl")

include("model.jl")

include("fitting.jl")
include("plotting.jl")

include("augment_star.jl")

include("rvcontent.jl")
include("driver.jl")

include("combine_rvs.jl")


end
