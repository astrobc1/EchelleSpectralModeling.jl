using EchelleBase
using EchelleSpectralModeling
using Infiltrator
using CurveFitParameters
using Statistics
using NaNStatistics

const SPEED_OF_LIGHT_MPS = 299792458.0

include("rvcontent.jl")
include("coaddrvs.jl")
include("ccf.jl")