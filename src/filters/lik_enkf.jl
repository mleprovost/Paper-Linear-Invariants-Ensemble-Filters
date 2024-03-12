import TransportBasedInference: SeqFilter

export LikEnKF

"""
$(TYPEDEF)

A structure for the likelihood-based ensemble Kalman filter (EnKF)

References:

$(TYPEDFIELDS)
"""

struct LikEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LikEnKF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs; isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"
    return LikEnKF(G, ϵy, Δtdyn, Δtobs, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LikEnKF(ϵy::InflationType, Δtdyn, Δtobs)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"
    return LikEnKF(x -> x, ϵy, Δtdyn, Δtobs, false)
end



function Base.show(io::IO, enkf::LikEnKF)
	println(io,"Likelihood-based EnKF  with filtered = $(enkf.isfiltered)")
end


function (enkf::LikEnKF)(X, ystar::Array{Float64,1}, t::Float64)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enkf.ϵy) <: AdditiveInflation
        E .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
    end

    # Add observational noise samples to non-perturbed observations
    X[1:Ny,:] .+= E

    # Compute joint covariance matrix
    ΣYX = cov(X')
                                                         
    # Factorization of ΣY
    ΣY = factorize(Symmetric(ΣYX[1:Ny, 1:Ny]))

    # Cross covariance matrix between X and Y
    ΣXcrossY = ΣYX[Ny+1:Ny+Nx,1:Ny]

    # Apply Kalman's update using representers. 
    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]
        
        bi = ΣY \ (yi - ystar)
        X[Ny+1:Ny+Nx,i] =  xi - ΣXcrossY*bi
    end

	return X
end 