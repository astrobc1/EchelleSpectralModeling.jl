module Wavelength

# Polynomial
include("polywave.jl")
include("splinewave.jl")
include("apriori.jl")

# Etalons (LFC and fp etalons)
include("etalons.jl")

# 2d polynomial fitting
include("chebyshevwave.jl")

end