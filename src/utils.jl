function optimfunwrapper(x::Vector, g::Vector, site, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site,  var)
end

function optimfunwrapper(x::Vector, g::Vector, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGradSym!(x, g, var)
end

function inflate_matrix(J::Array{Float64,3},N)
    q,q,NN = size(J)

    @assert (N*(N-1))>>1 == NN

    Jt = zeros(q,q,N,N)
    ctr = 0
    for i in 1:N-1
        for j in i+1:N
            ctr += 1
            Jt[:,:,i,j] = J[:,:,ctr]
            Jt[:,:,j,i] = J[:,:,ctr]'
        end
    end
    return Jt
end

function correct_APC(S::Matrix)
    N = size(S, 1)
    Si = sum(S, dims=1)
    Sj = sum(S, dims=2)
    Sa = sum(S) * (1 - 1/N)

    S -= (Sj * Si) / Sa
    return S
end

function compute_APC(J::Array{Float64,4},N,q)
    FN = fill(0.0, N,N)
    for i=1:N-1
        for j=i+1:N
            FN[i,j] = norm(J[1:q-1,1:q-1,i,j],2)
            FN[j,i] =FN[i,j]
        end
    end
    FN=correct_APC(FN)
    return FN
end


function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool)

    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z)
    end

    N, M = size(Z)
    q = round(Int,maximum(Z))

    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta)
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W, Zint,N,M,q
end

function compute_ranking(S::Matrix{Float64}, min_separation::Int = 5)
    N = size(S, 1)
    R = Array{Tuple{Int,Int,Float64}}(undef, div((N-min_separation)*(N-min_separation+1), 2))
    counter = 0
    for i = 1:N-min_separation, j = i+min_separation:N
        counter += 1
        R[counter] = (i, j, S[j,i])
    end

    sort!(R, by=x->x[3], rev=true)
    return R 
end

function sumexp(vec::Array{Float64,1})
    mysum = 0.0
    @inbounds @simd for i=1:length(vec)
        mysum += exp(vec[i])
    end
    return mysum
end
