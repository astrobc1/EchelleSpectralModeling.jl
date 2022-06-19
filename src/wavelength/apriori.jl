using EchelleBase
using EchelleSpectralModeling
using CurveFitParameters

export APrioriλSolution

struct APrioriλSolution <: SpectralModelComponent
end

function EchelleSpectralModeling.build(m::APrioriλSolution, data::SpecData1d, pars::Parameters, sregion::SpecRegion1d)
    return data.data.λ
end