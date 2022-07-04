module EchelleSpectralModeling

using Reexport

# Barycenter corrections
include("barycenter.jl")

# Modeling API
include("spectralmodeling.jl")

# Model components
include("modelcomponent.jl")

# Implementations of components
include("continuum/Continuum.jl")
@reexport using .Continuum
include("gascell/GasCells.jl")
@reexport using .GasCells
include("lsf/LSF.jl")
@reexport using .LSF
include("star/Star.jl")
@reexport using .Star
include("tellurics/Tellurics.jl")
@reexport using .Tellurics
include("wavelength/Wavelength.jl")
@reexport using .Wavelength

# Primary model container
include("spectralmodel.jl")

# RV specific methods
include("rvcalc/rvcalc.jl")

# Objectives for optimizing
include("objectives/Objectives.jl")
@reexport using .SpectralModelObjectiveFunctions

# Template Augmenting
include("augmenting/TemplateAugmenting.jl")
@reexport using .TemplateAugmenting

# Primary container to hold data, model, objective, augmenter
include("ensemble.jl")

# Plotting
include("plotting.jl")

# Fitting api
include("fitting.jl")

end