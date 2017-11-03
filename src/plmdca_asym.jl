function plmdca(filename::String;
                        decimation::Bool=false,
                        boolmask::Union{Array{Bool,2},Void}=nothing,
                        fracmax::Float64 = 0.3,
                        fracdec::Float64 = 0.1,
                        remove_dups::Bool = true,
                        min_separation::Int = 1,
                        max_gap_fraction::Real = 0.9,
                        theta = :auto,
                        lambdaJ::Real=0.005,
                        lambdaH::Real=0.01,
                        gaugecol::Int=-1,
                        epsconv::Real=1.0e-5,
                        maxit::Int=1000,
                        verbose::Bool=true,
                        stability::Float64=0.0,
                        method::Symbol=:LD_LBFGS)

    W,Z,N,M,q = ReadFasta(filename,max_gap_fraction, theta, remove_dups)

    boolmask != nothing && size(boolmask) != (N,N) && error("size boolmask different from ( $N, $N )")

    plmalg = PlmAlg(method,verbose, epsconv ,maxit, boolmask)
    plmvar = PlmVar(N,M,q,q*q,gaugecol,lambdaJ,lambdaH,stability,Z,W)


    if decimation  == false
        Jmat, pslike = MinimizePLAsym(plmalg,plmvar)                
    else
        plmalg.boolmask != nothing && println("Warning: decimation with boolmask not implemented. Continuing ...")
        decvar = DecVar{2}(fracdec, fracmax, ones(Bool, (N-1)*q*q, N)) 
        Jmat, pslike = DecimateAsym!(plmvar, plmalg, decvar)        
    end


    score, FNAPC, Jtensor = ComputeScore(Jmat, plmvar, min_separation)
    return output = PlmOut{4}(sdata(pslike), Jtensor, score)
end

function optimfunwrapper(x::Vector, g::Vector, site, var)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site,  var)            
end



function MinimizePLAsym(alg::PlmAlg, var::PlmVar)

    LL = (var.N - 1) * var.q2 + var.q
    x0 = zeros(Float64, LL)
    vecps = SharedArray{Float64}(var.N)
    Jmat = @parallel hcat for site=1:var.N #1:12
        
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        if alg.boolmask != nothing # constrain to zero boolmask variables
            lb,ub=ComputeUL(alg,var,site,LL)
            lower_bounds!(opt, lb)
            upper_bounds!(opt, ub)
        end
        min_objective!(opt, (x,g)->optimfunwrapper(x,g,site,var))
        elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        vecps[site] = minf
        minx
    end 
    return Jmat, vecps
end

function ComputeUL(alg::PlmAlg, var::PlmVar, site::Int, LL::Int)

    boolmask = alg.boolmask
    N  = var.N
    q2 = var.q2
    lb = -Inf * ones(Float64, LL)
    ub =  Inf * ones(Float64, LL)
    tiny::Float64 = 1.0e-6
    offset::Int = 0

    for i=1:site-1
        if boolmask[i,site] == false
            for s = 1:q2            
                lb[offset + s] = -tiny
                ub[offset + s] =  tiny
            end
        end
        offset += q2 
    end
    for i=site+1:N
        if boolmask[i, site] == false
            for s = 1:q2            
                lb[offset + s] = -tiny
                ub[offset + s] =  tiny
            end
        end
        offset += q2 
    end
    return lb,ub
end

function PLsiteAndGrad!(vecJ::Array{Float64,1},  grad::Array{Float64,1}, site::Int, plmvar::PlmVar)

    LL = length(vecJ)
    q2 = plmvar.q2
    q = plmvar.q
    gaugecol = plmvar.gaugecol
    N = plmvar.N
    M = plmvar.M
    stability = plmvar.stability
    Z = sdata(plmvar.Z)
    W = sdata(plmvar.W)


    for i=1:LL-q
        grad[i] = 2.0 * plmvar.lambdaJ * vecJ[i]
    end
    for i=(LL-q+1):LL
       grad[i] = 2.0 * plmvar.lambdaH * vecJ[i]
    end 

    vecene = zeros(Float64,q)
    expvecenesunorm = zeros(Float64,q)
    pseudolike = 0.0
 
    @inbounds begin 
        for a = 1:M       
            fillvecene!(vecene, vecJ, site, a, q, Z, N, stability)        
            lnorm = log(sumexp(vecene))
            expvecenesunorm .= exp.(vecene .- lnorm)
            pseudolike -= W[a] * (vecene[Z[site,a]] - lnorm)
            offset = 0         
            for i = 1:site-1 
                @simd for s = 1:q
                    grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] *  expvecenesunorm[s]
                end
                grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] 
                offset += q2 
            end
	    for i = site+1:N 
                @simd for s = 1:q
                    grad[ offset + s + q * ( Z[i,a] - 1 ) ] += W[a] *  expvecenesunorm[s]
                end
                grad[ offset + Z[site,a] + q * ( Z[i,a] - 1 ) ] -= W[a] 
                offset += q2 
            end

            @simd for s = 1:q 
                grad[ offset + s ] += W[a] *  expvecenesunorm[s] 
            end
	    grad[ offset + Z[site,a] ] -= W[a] 	
        end
    end

   

    if 1 <= gaugecol <= q         
        offset = 0;
        @inbounds begin 
            for i=1:N-1
                for s=1:q
                    grad[offset + gaugecol + q * (s - 1)  ] = 0.0; # Gauge!!! set gradJ[a,q] = 0
                    grad[offset + s + q * (gaugecol - 1) ] = 0.0; # Gauge!!! set gradJ[q,a] = 0
                end
                offset += q2
            end
            grad[offset + gaugecol] = 0.0 # Gauge!!! set gradH[q] = 0
        end
    end

    pseudolike += L2norm_asym(vecJ, plmvar)

    return pseudolike 
end

function fillvecene!(vecene::Array{Float64,1}, vecJ::Array{Float64,1},site::Int, a::Int, q::Int, sZ::DenseArray{Int,2},N::Int,stability::Float64)

    q2 = q*q   
    Z = sdata(sZ)
    colorsite = Z[site, a]
    
    @inbounds for l = 1:q
        offset::Int = 0
        scra::Float64 = 0.0
        for i = 1:site-1 # Begin sum_i \neq site J
            scra += vecJ[offset + l + q * (Z[i,a]-1)] 
            offset += q2 
        end
        # skipping sum over residue site
    	for i = site+1:N
            scra += vecJ[offset + l + q * (Z[i,a]-1)] 
            offset += q2
        end # End sum_i \neq site J
        scra += vecJ[offset + l] # sum H 
        vecene[l] = scra
    end
    vecene[colorsite]-= stability
    nothing
end


function L2norm_asym(vec::Array{Float64,1}, plmvar::PlmVar)
    q = plmvar.q    
    N = plmvar.N
    lambdaJ = plmvar.lambdaJ
    lambdaH = plmvar.lambdaH

    LL = length(vec)

    mysum1 = 0.0
    @inbounds @simd for i=1:(LL-q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJ

    mysum2 = 0.0
    @inbounds @simd for i=(LL-q+1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= lambdaH

    return mysum1+mysum2
end


