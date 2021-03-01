function plmdca_sym(Z::Array{T,2},W::Vector{Float64};
                decimation::Bool=false,
                fracmax::Float64 = 0.3,
                fracdec::Float64 = 0.1,
                blockdecimate::Bool=true, # decimate on a per-block base (all J[:,:,i,j] = 0)
                remove_dups::Bool = true,
                min_separation::Int = 1,
                max_gap_fraction::Real = 0.9,
                theta = :auto,
                lambdaJ::Real=0.01,
                lambdaH::Real=0.01,
                epsconv::Real=1.0e-5,
                maxit::Int=1000,
                verbose::Bool=true,
                method::Symbol=:LD_LBFGS) where T<: Integer


    all(x->x>0,W) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(W),1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))

    N,M = size(Z)
    M = length(W)
    q = Int(maximum(Z))

    plmalg = PlmAlg(method,verbose, epsconv ,maxit)
    plmvar = PlmVar(N,M,q,q*q,lambdaJ,lambdaH,Z,W)

    Jmat,pseudolike = if decimation == false
        MinimizePLSym(plmalg,plmvar)
    else
        if blockdecimate
            decvar = DecVar{1}(fracdec, fracmax, blockdecimate, ones(Bool, binomial(N,2)))
        else
            decvar = DecVar{1}(fracdec, fracmax, blockdecimate, ones(Bool, binomial(N,2)*q*q+N*q))
        end
        DecimateSym!(plmvar, plmalg, decvar)
    end
    score, Jtens, htens = ComputeScoreSym(Jmat, plmvar, min_separation)

    return PlmOut(pseudolike, Jtens, htens, score)
end

function plmdca_sym(filename::String;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    W,Z,N,M,q = ReadFasta(filename,max_gap_fraction, theta, remove_dups)
    plmdca_sym(Z,W; kwds...)
end

function MinimizePLSym(alg::PlmAlg, var::PlmVar)

    N  = var.N
    q  = var.q
    q2 = var.q2
    Z = var.Z
    Nc2 = binomial(N,2)
    LL  = Nc2 * q2  + N * q

    x0 = zeros(Float64, LL)
    batchidx = myrange(Z)
    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x,g)->like_grad!(g,x,var,batchidx,alg.verbose))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t exit status = ", minf, elapstime)
    alg.verbose && println(ret)

    return minx, minf
end

function myrange(q::DenseMatrix)
    nchunks = nprocs()
    nchunks > 2 || (return [1:size(q,2),])
    splits = [round(Int, s) for s in range(0, stop=size(q,2), length=nchunks)]
    [splits[idx]+1:splits[idx+1] for idx in 1:nchunks-1]
end

reduce_res(x::Tuple) = x
(reduce_res(x::Vector{T}) where T<:Tuple) = broadcast(+,x...)


function L2norm_sym(vec::AbstractVector, var::PlmVar)

    q = var.q
    N = var.N
    lambdaJ = var.lambdaJ
    lambdaH = var.lambdaH

    LL = length(vec)


    mysum1 = 0.0
    @inbounds @simd for i in 1:(LL-N*q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJ

    mysum2 = 0.0
    @inbounds @simd for i in (LL - N*q + 1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= 2lambdaH
# 	mysum2 *= lambdaH
    return mysum1+mysum2
end

function add_l2grad!(grad::Vector,vecJ::AbstractVector,plmvar::PlmVar)
    N = plmvar.N
    LL = length(grad)
    q = plmvar.q
    lambdaJ = plmvar.lambdaJ
    lambdaH = plmvar.lambdaH
    @inbounds begin
        @simd for i in 1:LL-N*q
            grad[i] += 1.0 * vecJ[i] * lambdaJ
        end
        @simd for i in (LL-N*q + 1):LL
            grad[i] += 2.0 * vecJ[i] * lambdaH
#			grad[i] += 2.0 * vecJ[i] * lambdaH
        end
    end
    nothing
end
function like_grad!(g::Vector,vecJ::AbstractVector,plmvar::PlmVar,vec_chunk,verbose)

    g === nothing && (g = zero(vecJ))
    sJ = SharedArray(vecJ)
    res = @distributed vcat for chunk in vec_chunk
        pl_chunk, gr_chunk = plm_site_grad(sJ, plmvar, chunk)
    end
    pl,gr = reduce_res(res)
    pl += L2norm_sym(vecJ, plmvar)
    verbose && println("pl = $pl")
    add_l2grad!(gr,vecJ, plmvar)
    g .= gr
    return pl
end

function plm_site_grad(vecJ::AbstractVector, plmvar::PlmVar,chunk)

    LL = length(vecJ)
    q2 = plmvar.q2
    q = plmvar.q
    N = plmvar.N
    M = plmvar.M
    Z = plmvar.Z
    W = plmvar.W
    grad = zero(vecJ)
    pseudolike = 0.0
    for a in chunk
        pseudolike += ComputePatternPLSym!(grad, vecJ, Z[:,a],W[a], N, q, q2)
    end

    return pseudolike,grad
end


function ComputePatternPLSym!(grad::Array{Float64,1}, vecJ::AbstractVector, Z::AbstractArray{Int,1}, Wa::Float64, N::Int, q::Int, q2::Int)
    vecene = zeros(Float64,q)
    expvecenesunorm = zeros(Float64,q)
    pseudolike = 0.0
    offset = mygetindex(N-1, N, q, q, N, q, q2)

    @inbounds for site=1:N    # site < i
        fillvecenesym!(vecene, vecJ, Z, site, q,N)

  	    norm = sumexp(vecene)
        expvecenesunorm .= exp.(vecene .- log(norm))
        pseudolike -= Wa * ( vecene[Z[site]] - log(norm) )
		@simd for i = 1:(site-1)
             for s = 1:q
                grad[ mygetindex(i, site, Z[i], s, N, q, q2) ] += 0.5 * Wa * expvecenesunorm[s]
            end
            grad[ mygetindex(i, site , Z[i], Z[site],  N,q,q2)] -= 0.5 * Wa
        end
		@simd for i = (site+1):N
             for s = 1:q
                grad[ mygetindex(site, i , s,  Z[i], N,q,q2) ] += 0.5 * Wa * expvecenesunorm[s]
            end
            grad[ mygetindex(site, i , Z[site], Z[i], N,q,q2)] -= 0.5* Wa
        end
        @simd for s = 1:q
            grad[ offset + s ] += Wa *  expvecenesunorm[s]
        end
		grad[ offset + Z[site] ] -= Wa
 		offset += q
    end
    return pseudolike
end


function fillvecenesym!(vecene::AbstractArray, vecJ::AbstractVector, Z::AbstractArray{Int64,1}, site::Int, q::Int ,N::Int)
    
    q2 = q*q
    LL=length(vecJ)

    @inbounds begin
        @simd for l = 1:q
            offset::Int = 0
            scra::Float64 = 0.0

            for i=1:site-1
                scra += vecJ[ mygetindex(i, site, Z[i], l,  N, q, q2)]
            end
    	    for i = site+1:N
                scra += vecJ[ mygetindex(site, i, l, Z[i], N, q, q2)]
            end # End sum_i \neq site J
           	#scra *= 0.5
            #offset = mygetindex(N-1, N, q, q, N, q, q2)  + ( site - 1) * q  # last J element + (site-1)*q

            scra += vecJ[LL + ( site - 1 - N ) * q + l] # sum H
            vecene[l] = scra
        end
    end
end 



@inline function mygetindex( i::Int, j::Int, coli::Int, colj::Int, N::Int, q::Int, q2::Int)
    offset_i = ( (i-1) * N  - ( (i * ( i -1 ) ) >> 1 ) )  # (i-1) N q2 + i (i-1) q2 / 2
    offset_j = (j - i - 1 ) 
    return ( offset_i + offset_j ) * q2 + coli + q * (colj - 1)
end



function ComputeScoreSym(Jvec::Array{Float64,1}, var::PlmVar, min_separation::Int)

    LL = length(Jvec)
    N=var.N
    q=var.q
    Nc2 = binomial(N,2)

    Jtens=reshape(Jvec[1:LL-N*q],q,q,Nc2)
    htens=fill(0.0,q,N)
    J = zeros(q,q,Nc2)

    htens=reshape(Jvec[LL-N*q + 1:end],q,N)

    for l=1:Nc2
        J[:,:,l] = Jtens[:,:,l] -
					repeat(mean(Jtens[:,:,l],dims=1),q,1) -
					repeat(mean(Jtens[:,:,l],dims=2),1,q) .+
					mean(Jtens[:,:,l])
    end


    FN = zeros(Float64, N,N)
    l = 1
    for i=1:N-1
        for j=i+1:N
            FN[i,j] = norm(J[:,:,l],2)
            FN[j,i] = FN[i,j]
            l+=1
        end
    end
    FN = correct_APC(FN)
    score = compute_ranking(FN,min_separation)
	GC.gc() # something wrong with SharedArray
    return score, inflate_matrix(Jtens,N),htens
end
