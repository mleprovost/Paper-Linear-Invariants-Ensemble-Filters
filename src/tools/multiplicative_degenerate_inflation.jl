import TransportBasedInference: InflationType
import Statistics: mean, cov

export MultiDegenAddInflation

"""
MultiDegenAddInflation <: InflationType

A type to store a linear invariants-preserving additive and multiplicative inflation:
x <- β(x - μX) + μX + PUperp(ϵ) with ϵ is drawn from a Gaussian distribution
and PUperp is an orthogonal projector on the complement of Uperp.

## Fields:
$(TYPEDFIELDS)

## Constructors
- `MultiDegenAddInflation(Nx::Int64, α::ContinuousMultivariateDistribution)`
- `MultiDegenAddInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}})`
- `MultiDegenAddInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `MultiDegenAddInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)`
- `MultiDegenAddInflation(m::Array{Float64,1}, σ::Float64)`
"""

struct MultiDegenAddInflation <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Multiplicative inflation factor β"
    β::Float64

    "Mean of the additive inflation"
    m::Array{Float64,1}

    "Covariance of the additive inflation"
    Σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Square-root of the covariance matrix"
    σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Orthonormal projector"
    PUperp::OrthoProjector
end

# Some convenient constructors 
# By default, the distribution of the additive inflation is a multivariate
# normal distribution with zero mean and identity as the covariance matrix

function MultiDegenAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}}, Uperp::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(Σ,1)==size(Σ,2) "Error dimension of the covariance matrix"
# @assert isapprox(Uperp'*Uperp, I, atol = 100*eps()) "Columns of Uperp are not orthonormal"
return MultiDegenAddInflation(Nx, β, m, Σ, sqrt(Σ), OrthoProjector(Uperp))

end

function MultiDegenAddInflationn(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Array{Float64,1}, Uperp::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(σ,1) "Error dimension of the std vector"
# @assert isapprox(Uperp'*Uperp, I, atol = 100*eps()) "Columns of Uperp are not orthonormal"

return MultiDegenAddInflation(Nx, β::Float64, m, Diagonal(σ .^2), Diagonal(σ), OrthoProjector(Uperp))

end

function MultiDegenAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Float64, 
         Uperp::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
# @assert isapprox(Uperp'*Uperp, I, atol = 100*eps()) "Columns of Uperp are not orthonormal"

return MultiDegenAddInflation(Nx, β, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)), OrthoProjector(Uperp))

end

function MultiDegenAddInflation(β::Float64, m::Array{Float64,1}, σ::Float64,
            Uperp::Union{Array{Float64, 1}, Array{Float64, 2}})
    Nx = size(m,1)
    # @assert isapprox(Uperp'*Uperp, I, atol = 100*eps()) "Columns of Uperp are not orthonormal"

    return  MultiDegenAddInflation(Nx, β, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)), OrthoProjector(Uperp))
end

Base.size(A::MultiDegenAddInflation)  = A.Nx
mean(A::MultiDegenAddInflation) = A.PUperp(A.m)
cov(A::MultiDegenAddInflation) = (I - A.PUperp.Uperp*A.PUperp.Uperp')*A.Σ*(I - A.PUperp.Uperp*A.PUperp.Uperp')

"""
(A::MultiDegenAddInflation)(X, start::Int64, final::Int64)

Apply the degenerate additive and multiplicative inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> β(xⁱ - μX) + μX + PUperp(ϵⁱ) with ϵⁱ is drawn from a Gaussian distribution.
"""
function (A::MultiDegenAddInflation)(X, start::Int64, final::Int64)
    Ne = size(X,2)
    @assert A.Nx == final - start + 1 "final-start + 1 doesn't match the length of the additive noise"
    # @show X[start:final, 1]

    μX = mean(X[start:final,:], dims= 2)[:,1]
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .= A.β*(col - μX) +  μX + A.PUperp(A.m +  A.σ*randn(A.Nx))
    end
end

"""
(A::MultiDegenAddInflation)(X)

Apply the degenerate additive and multiplicative `A` to an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ is drawn from a Gaussian distribution.
"""
(A::MultiDegenAddInflation)(X) = A(X, 1, size(X,1))
