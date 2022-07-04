using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

export APrioriλSolution

"""
    APrioriλSolution
An empty type for wavelength solutions known a priori, likely from an LFC, etalon, or ThAr lamp. By default, this grid is stored in the data object as `data.data.λ`.
"""
struct APrioriλSolution <: SpectralModelComponent
end

"""
    build(m::APrioriλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
Wrapper to return the predetermined wavelength solution, assumed to be stored in `data.data.λ`. 
"""
function EchelleSpectralModeling.build(m::APrioriλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    return data.data.λ
end