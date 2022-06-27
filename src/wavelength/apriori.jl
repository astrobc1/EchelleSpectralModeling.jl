using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

export APrioriλSolution

"""
    APrioriλSolution
An empty type for wavelength solutions known a priori, likely from an LFC, etalon, or ThAr lamp.
"""
struct APrioriλSolution <: SpectralModelComponent
end


function EchelleSpectralModeling.build(m::APrioriλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    return data.data.λ
end