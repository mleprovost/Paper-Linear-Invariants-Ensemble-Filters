export LocLiksEnKF

"""
$(TYPEDEF)

A structure for the localized likelihood-based stochastic ensemble Kalman filter (LiksEnKF)

References:

$(TYPEDFIELDS)
"""

struct LocLiksEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Localization structure"
    Loc::Localization

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LocLiksEnKF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Loc::Localization; isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocLiksEnKF(G, ϵy, Δtdyn, Δtobs, Loc, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LocLiksEnKF(ϵy::InflationType, Δtdyn, Δtobs, Loc::Localization)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocLiksEnKF(x -> x, ϵy, Δtdyn, Δtobs, Loc, false)
end


function Base.show(io::IO, enkf::LocLiksEnKF)
	println(io,"Localized Likelihood-based sEnKF with filtered = $(enkf.isfiltered)")
end


# Version with localization 

function (enkf::LocLiksEnKF)(X, ystar::Array{Float64,1}, t::Float64)

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enkf.ϵy) <: AdditiveInflation
        E .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
    elseif typeof(enkf.ϵy) <: TDistAdditiveInflation
        E .= rand(enkf.ϵy.dist, Ne)
    else
        print("Inflation type not defined")
    end


    locXY = Locgaspari((Nx, Ny), enkf.Loc.L, enkf.Loc.Gxy)
    
    # Add observational noise samples as we can only sample from the likelihood model
    X[1:Ny,:] .+= E

    μX = mean(X[Ny+1:Ny+Nx,:], dims = 2)[:,1]
    μY = mean(X[1:Ny,:], dims = 2)[:,1]

    AX = 1/sqrt(Ne-1)*(X[Ny+1:Ny+Nx,:] .- μX)
    AY = 1/sqrt(Ne-1)*(X[1:Ny,:] .- μY)

    ΣY = AY*AY'

    AXY_loc = locXY .* (AX*(AY)')

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]

        bi = ΣY \ (yi - ystar)
       
        X[Ny+1:Ny+Nx,i] = xi - AXY_loc*bi
    end

	return X
end