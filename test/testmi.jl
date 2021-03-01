module TestMI
using Test, PlmDCA
MI = PlmDCA.mutualinfo("test.fasta",pseudocount=0)
@test MI == zero(MI)
@test true
printstyled("All TestMI passed!\n",color=:light_green,bold=true)
end
