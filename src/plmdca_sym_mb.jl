function plmdca_sym_mb(Z::Array{T,2},W::Vector{Float64};
                decimation::Bool=false,
                fracmax::Float64 = 0.3,
                fracdec::Float64 = 0.1,
                blockdecimate::Bool=true, # decimate on a per-block base (all J[:,:,i,j] = 0)
                remove_dups::Bool = true,
                min_separation::Int = 1,
                lambdaJ::Real=0.01,
                lambdaH::Real=0.01,
                epsconv::Real=1.0e-5,
                maxit::Int=20, #n_epoch
                n_minibatch::Int=10,
                verbose::Bool=true,
                method::Symbol=:RMSProp) where T<: Integer


    all(x->x>0,W) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(W),1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))

    N,M = size(Z)
    M = length(W)
    q = Int(maximum(Z))

    plmalg = PlmAlg(method,verbose, epsconv ,maxit)
    plmvar = PlmVar(N,M,q,q*q,lambdaJ,lambdaH,Z,W)

    Jmat,pseudolike = if decimation == false
        MinimizePLSym_mb(plmalg,plmvar,n_minibatch = n_minibatch)
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

function plmdca_sym_mb(filename::String;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    W,Z,N,M,q = ReadFasta(filename,max_gap_fraction, theta, remove_dups)
    plmdca_sym_mb(Z,W; kwds...)
end

# SUGGESTED OPTIMIZER:
# RMSProp()
# AMSGrad()
function MinimizePLSym_mb(alg::PlmAlg, var::PlmVar; n_minibatch::Int=10, method=RMSProp(), full_opt::Bool=true,cb::Int=10)

    N  = var.N
    q  = var.q
    q2 = var.q2
    Z = var.Z
    W = var.W

    fatol=alg.epsconv
    frtol=alg.epsconv
    xatol=alg.epsconv
    xrtol=alg.epsconv

    n_epoch=alg.maxit
    verbose=alg.verbose

    Nc2 = binomial(N,2)
    LL  = Nc2 * q2  + N * q

    x = zeros(Float64, LL)
    x_t = zeros(Float64, LL)
    grad=zero(x)

    pl=0.0; pl_t=0.0;

    opt = RMSProp();
    for e in 1:n_epoch

        ridx=randperm(size(Z,2))
        verbose && println("epoch: ",e," ")

        for mb in get_minibatches(ridx,n_minibatch)

            pls=pl_grad!(grad, x, view(Z,:,mb), view(W,mb), var)
            Flux.Optimise.update!(opt, x, grad)

        end

        if rem(e,cb) == 0 

            pl = pl_grad!(zeros(size(x)), x, Z, W, var)  #full pseudolikelihood

            verbose &&  println("pl = ",pl)
            #check convergence
            fa=abs(pl-pl_t);fr=abs(pl-pl_t)/pl; xa=sum(abs2.(x.-x_t));xr=sum(abs2.(x.-x_t))/sum(abs2.(x));
            if fa < fatol || fr < frtol || xa < xatol || xr < xrtol  
                verbose &&  println("TOL reached")
                return x, pl
            end
            
            x_t.=x;
            pl_t=pl
        end
    end

    # full optimization
    if full_opt
        verbose && println("Full optimize: ")
        for op in 1:n_epoch
           
            print(op," ")

            pl=pl_grad!(grad, x, Z, W, var)
            Flux.Optimise.update!(opt, x, grad)
            println("pl = ",pl)

            #check convergence
            fa=abs(pl-pl_t);fr=abs(pl-pl_t)/pl; xa=sum(abs2.(x.-x_t));xr=sum(abs2.(x.-x_t))/sum(abs2.(x));
            println("fa = ",fa," fr = ",fr," xa = ",xa," xr = ",xr)
            if fa < fatol || fr < frtol || xa < xatol || xr < xrtol  
                verbose &&  println("TOL reached")
                return x, pl
            end

            x_t.=x
            pl_t=pl
        end
    end


    verbose && println("MAXEVAL reached")

    return x, pl
end

function get_minibatches(idx,n_mb)

    splits = [round(Int, s) for s in range(0, stop=length(idx), length=n_mb+1)]
    [idx[splits[m]+1:splits[m+1]] for m in 1:n_mb]
end

function pl_grad!(grad::Array{Float64,1}, vecJ::AbstractVector, Zmb::AbstractArray{Int,2}, W::AbstractArray{Float64,1}, plmvar::PlmVar)
    
    
    LL = length(vecJ)
    q2 = plmvar.q2
    q = plmvar.q
    N = plmvar.N
    M = size(Zmb,2)
    
    vecene = zeros(Float64,q)
    expvecenesunorm = zeros(Float64,q)
    pseudolike =  L2norm_sym(vecJ, plmvar)

    offset = mygetindex(N-1, N, q, q, N, q, q2)

    @inbounds for m=1:M   # site < i

        Z=Zmb[:,m]
        Wa=W[m]

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
    end

    add_l2grad!(grad,vecJ, plmvar)
    return pseudolike
end
