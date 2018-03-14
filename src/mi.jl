function mutualinfo(filename::AbstractString;
                    remove_dups::Bool=true,
                    min_separation::Int=1,
                    max_gap_fraction::Real=0.9,
                    pseudocount::Real=0.2,
                    theta=:auto,
                    output=:matrix # either :matrix, :score (i,j,val) or :apcscore
                    )

    Z = GaussDCA.read_fasta_alignment(filename, max_gap_fraction)
    q = Int(maximum(Z))
    Pi_true, Pij_true, Meff, _ = GaussDCA.compute_new_frequencies(Z, q,  theta)
    N = div(length(Pi_true),q-1)
    if 0.0 < pseudocount <= 1.
        Pi, Pij = GaussDCA.add_pseudocount(Pi_true, Pij_true, pseudocount, N, q)
        mi = mut_inf(Pij,Pi,q)
    elseif pseudocount == 0
        mi = mut_inf(Pij_true,Pi_true,q)
    else
        error("pseudocount = $pseudocount should be in [0,1]")
    end    
    if output == :matrix
        return mi
    elseif output == :score
        Neff = N - min_separation
        score   = Vector{Tuple{Int,Int,Float64}}(Neff*(Neff+1)>>1)

        ctr = 0
        for i=1:N-min_separation, j=i+min_separation:N
            ctr += 1
            score[ctr]=(i,j,mi[i,j])
        end
        sort!(score,by=x->x[3],rev=true)
        return score
    elseif output == :apcscore
        mi=GaussDCA.correct_APC(mi) 
        score = GaussDCA.compute_ranking(mi,min_separation)
        return score
    else
        error("$output: output format not supported. Only :matrix, :score, :apcscore")
    end
end


# This computation is complicated by the fact that in Pij we
# store only probabilities 1,...,q-1 × 1,...,q-1.
# We need therefore to recontruct the frame values (bottom, and right)
# A special case is for the element q,q

function mut_inf(Pij::Matrix{Float64},Pi::Vector{Float64},Q::Int)
    q=Q-1

    N1,N2 = size(Pij)
    N1 == N2 || error("matrix should be square. size(matrix) = $N1 != $N2")
    N3 = length((Pi))
    N3 == N2 || error("vector dim=$N3 size(matrix) = ($N1,$N1)")

    rem(N3,q) == 0 || error("dimension dim=$N3 should be multiple of $q")    
    N = div(N1,q)
    
    scoreM = zeros(Float64,N,N)
    piq = zeros(Float64, N)    # vector of values for Pi[Q]
    pijq = zeros(Float64, N,q) # vector of values for Pij[i,j,Q,a] a in 1,...,q 

    vec_row = zeros(Float64,q)
    vec_col = zeros(Float64,q)


    for i=1:N
        blk = (i-1)*q+1:i*q
        for j in blk
            piq[i] -= Pi[j]
        end
        piq[i] += 1.
        if 0.0 > piq[i] > 1.0
            error("probability is not normalized???")
        end
    end

    ctr = 0
    for i=1:N-1
        row0 = (i-1)*q
        for j=i+1:N
            compute_frame_col!(vec_col, i, j, Pij,Pi,q)
            compute_frame_row!(vec_row, i, j, Pij,Pi,q)
            col0 = (j-1)*q
            mi = 0.0
            for b=1:q
                for a=1:q
                    pij = Pij[row0 + a, col0 + b]
                    pi  = Pi[row0 + a]
                    pj  = Pi[col0 + b] 
                    if pij > 0
                        mi += pij * log(pij / (pi*pj))                     

                    end
                end
            end            
            for a=1:q
                pij = vec_row[a]
                pi  = piq[i]
                pj  = Pi[col0 + a]
                pipj = pi * pj
                if pij > 0 && pipj >0 
                    mi += pij * log(pij / (pi*pj)) # contribution bottom frame
                end
                pij = vec_col[a]
                pi  = Pi[row0 + a]
                pj  = piq[j]
                pipj = pj * pj
                if pij > 0 && pipj > 0  
                    mi += pij * log(pij / (pi*pj)) # contribution rigth frame
                end                
            end                        
            
            _srow = sum(vec_row)  # contribution from Pij[q,q]
            _scol = sum(vec_col)
            pqqrow = piq[i] - _srow
            pqqcol = piq[j] - _scol
            pipj = piq[i]*piq[j]
            if pqqrow > 0 && pipj > 0
                mi += pqqrow * log(pqqrow/pipj)
            end                       
            scoreM[i,j] = mi
            scoreM[j,i] = mi
        end
    end
    return scoreM
end

function compute_frame_row!(vec_row::Vector{Float64},i::Int, j::Int, Pij::Matrix{Float64}, Pi::Vector{Float64}, q::Int)

    blki = (i-1)*q+1:i*q
    blkj = (j-1)*q+1:j*q
    ctr  = 0
    for b in blkj
        s = 0.0
        for a in blki
            s += Pij[a,b]
        end
        ctr += 1
        vec_row[ctr] = Pi[(j-1)*q + ctr] - s
        if 0.0 >= vec_row[ctr] >= 1.0
            println("element $ctr = ", vec_row[ctr])
            error("")
        end
    end
end


function compute_frame_col!(vec_col::Vector{Float64},i::Int, j::Int, Pij::Matrix{Float64}, Pi::Vector{Float64}, q::Int)

    blki = (i-1)*q+1:i*q
    blkj = (j-1)*q+1:j*q
    ctr  = 0
    for a in blki
        s = 0.0
        for b in blkj
            s += Pij[a,b]
        end
        ctr += 1
        vec_col[ctr] = Pi[(i-1)*q + ctr] - s        
        if 0.0 >= vec_col[ctr] >= 1.0
            println("element $ctr = ", vec_col[ctr])
            error("");
        end
    end
end
