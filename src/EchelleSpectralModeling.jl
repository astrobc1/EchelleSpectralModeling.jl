module EchelleSpectralModeling

using Reexport

include("barycenter.jl")

function build end

include("modelcomponent.jl")

include("continuum/continuum.jl")
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

include("spectralmodel.jl")

include("plotting.jl")

include("rvcalc/rvcalc.jl")

include("objectives/objectives.jl")
@reexport using .SpectralModelObjectiveFunctions

include("ensemble.jl")

include("augmenting/augmenters.jl")
@reexport using .TemplateAugmenters

include("fitting.jl")

end