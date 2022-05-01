module EchelleSpectralModeling

using Reexport

include("barycenter.jl")

include("ensemble.jl")

include("spectralmodel.jl")

include("modelcomponent.jl")

include("continuum/Continuum.jl")
@reexport using .Continuum
include("gascell/gascells.jl")
@reexport using .GasCells
include("lsf/lsf.jl")
@reexport using .LSF
include("star/star.jl")
@reexport using .Star
include("tellurics/tellurics.jl")
@reexport using .Tellurics
include("wavelength/wavelength.jl")
@reexport using .Wavelength

include("fitting.jl")

include("plotting.jl")

include("rvcalc.jl")

include("augmenting/augmenters.jl")
@reexport using .TemplateAugmenters

include("objectives/objectives.jl")
@reexport using .SpectralModelObjectiveFunctions

end