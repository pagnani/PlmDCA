module PlmDCA
using SharedArrays,Distributed,Printf, LinearAlgebra, Statistics
using NLopt
using GaussDCA
using LoopVectorization

export PlmOut, plmdca, plmdca_asym, plmdca_sym, plmdca_asym

include("types.jl")
include("utils.jl")
include("plmdca_asym.jl")
include("plmdca_sym.jl")
include("decimation_sym.jl")
include("mi.jl")
end
