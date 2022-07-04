module TemplateAugmenting

using EchelleSpectralModeling
using DelimitedFiles

export TemplateAugmenter, augment_star!

"""
    Base type for a template augmenter.
"""
abstract type TemplateAugmenter end

function augment_star! end


include("star_weighted_median.jl")
include("star_lsqspline.jl")
include("augment_utils.jl")

end