module PlmDCA
using SharedArrays,Distributed,Printf, LinearAlgebra, Statistics
using NLopt
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization

export PlmOut, plmdca, plmdca_asym, plmdca_sym, plmdca_asym

include("types.jl")
include("utils.jl")
include("plmdca_asym.jl")
include("plmdca_sym.jl")
include("decimation_sym.jl")
include("mi.jl")
end
