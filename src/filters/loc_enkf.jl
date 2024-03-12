export LocEnKF

"""
$(TYPEDEF)

A structure for the localized semi-empirical ensemble Kalman filter (LocEnKF)

References:

$(TYPEDFIELDS)
"""

struct LocEnKF<:SeqFilter
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

    "Observation matrix"
    H::AbstractMatrix

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LocEnKF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Loc::Localization, H::AbstractMatrix; isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF(G, ϵy, Δtdyn, Δtobs, Loc, H, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LocEnKF(ϵy::InflationType, Δtdyn, Δtobs, Loc::Localization, H::AbstractMatrix)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocEnKF(x -> x, ϵy, Δtdyn, Δtobs, Loc, H, false)
end


function Base.show(io::IO, enkf::LocEnKF)
	println(io,"Localized EnKF with filtered = $(enkf.isfiltered)")
end


# Version with localization 

function (enkf::LocEnKF)(X, ystar::Array{Float64,1}, t::Float64)

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    H = enkf.H

    # Generate observational noise samples
    E = enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m

    # Compute covariance of observational noise samples
    Σϵ = cov(E')

    # Build localization for the state variables
    locX = Locgaspari((Nx, Nx), enkf.Loc.L, enkf.Loc.Gxx)
    
    # Apply element-wise localization of the state covariance
    ΣX_loc = locX .* (cov(X[Ny+1:Ny+Nx,:]'))

    # Construct regularized observation covariance
    ΣY_loc = Symmetric(H*ΣX_loc*H' + Σϵ )


    # Apply Kalman's update using representers. 
    bi = zeros(Ny)
    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        hi = X[1:Ny,i]
        ϵi = E[:,i]

        bi .=  ΣY_loc \ (hi  + ϵi - ystar)
       
        X[Ny+1:Ny+Nx,i] = xi - ΣX_loc*(H'*bi)
    end

	return X
end