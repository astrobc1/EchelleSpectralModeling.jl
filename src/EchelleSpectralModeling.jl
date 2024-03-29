module EchelleSpectralModeling

using LinearAlgebra
using JLD2, DelimitedFiles
using NaNStatistics, DataInterpolations
using DataFrames
using SavitzkyGolay
using LoopVectorization
using LsqFit
using PyCall
using PyPlot
using OrderedCollections
using Distributed
using Polynomials
using StatsBase, Statistics
using Infiltrator

using IterativeNelderMead, Echelle

# Model Parameters
include("parameters.jl")

# Utils
include("data_utils.jl")
include("btsettl_utils.jl")
include("lsf_utils.jl")
include("continuum_utils.jl")

# Model components
include("tapas.jl")
include("star.jl")
include("poly_wls.jl")
include("spline_wls.jl")
include("poly_continuum.jl")
include("spline_continuum.jl")
include("static_wls.jl")
include("gascell.jl")
include("gausshermite_lsf.jl")

# Full Model
include("model.jl")

# Augmenting the star
include("augment_star.jl")

# Fitting API
include("fitting.jl")

# Driver for fitting
include("driver.jl")

# Spectral fit plots and RV plots
include("plotting.jl")

# Compute rv information content
include("rvcontent.jl")

# Combine final rvs
include("combine_rvs.jl")


end
