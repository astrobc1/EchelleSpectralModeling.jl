export StaticλSolution

struct StaticλSolution end

build(::StaticλSolution, ::Dict{String, <:Any}, ::Parameters, data::DataFrame) = data.λ