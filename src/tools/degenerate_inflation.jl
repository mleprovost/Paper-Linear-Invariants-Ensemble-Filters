
import TransportBasedInference: InflationType
import Statistics: mean, cov

export DegenerateAdditiveInflation

"""
DegenerateAdditiveInflation <: InflationType

A type to store a linear invariants-preserving additive inflation:
x <- x + Pϕ(ϵ) with ϵ is drawn from a Gaussian distribution
and Pϕ is an orthogonal projector on the complement of ϕ.


## Fields:
$(TYPEDFIELDS)

## Constructors
- `DegenerateAdditiveInflation(Nx::Int64, α::ContinuousMultivariateDistribution, β)`
- `DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}})`
- `DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)`
- `DegenerateAdditiveInflation(m::Array{Float64,1}, σ::Float64)`
"""

struct DegenerateAdditiveInflation <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Mean of the additive inflation"
    m::Array{Float64,1}

    "Covariance of the additive inflation"
    Σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Square-root of the covariance matrix"
    σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Orthonormal projector"
    Pϕ::OrthoProjector
end

# Some convenient constructors for multivariate Gaussian distributions
# By default, the distribution of the additive inflation α is a multivariate
 # normal distribution with zero mean and identity as the covariance matrix

function DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}}, ϕ::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(Σ,1)==size(Σ,2) "Error dimension of the covariance matrix"
# @assert isapprox(ϕ'*ϕ, I, atol = 100*eps()) "Columns of ϕ are not orthonormal"
return DegenerateAdditiveInflation(Nx, m, Σ, sqrt(Σ), OrthoProjector(ϕ))

end

function DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1}, ϕ::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(σ,1) "Error dimension of the std vector"
# @assert isapprox(ϕ'*ϕ, I, atol = 100*eps()) "Columns of ϕ are not orthonormal"

return DegenerateAdditiveInflation(Nx, m, Diagonal(σ .^2), Diagonal(σ), OrthoProjector(ϕ))

end

function DegenerateAdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64, 
         ϕ::Union{Array{Float64, 1}, Array{Float64, 2}})
@assert Nx==size(m,1) "Error dimension of the mean"
# @assert isapprox(ϕ'*ϕ, I, atol = 100*eps()) "Columns of ϕ are not orthonormal"

return DegenerateAdditiveInflation(Nx, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)), OrthoProjector(ϕ))

end

function DegenerateAdditiveInflation(m::Array{Float64,1}, σ::Float64,
            ϕ::Union{Array{Float64, 1}, Array{Float64, 2}})
    Nx = size(m,1)
    # @assert isapprox(ϕ'*ϕ, I, atol = 100*eps()) "Columns of ϕ are not orthonormal"

    return  DegenerateAdditiveInflation(Nx, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)), OrthoProjector(ϕ))
end

Base.size(A::DegenerateAdditiveInflation)  = A.Nx
mean(A::DegenerateAdditiveInflation) = A.Pϕ(A.m)
cov(A::DegenerateAdditiveInflation) = (I - A.Pϕ.ϕ*A.Pϕ.ϕ')*A.Σ*(I - A.Pϕ.ϕ*A.Pϕ.ϕ')

"""
(A::DegenerateAdditiveInflation)(X, start::Int64, final::Int64)

Apply the degenerate additive inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + Pϕ(ϵⁱ) with ϵⁱ drawn from a Gaussian distribution.
"""
function (A::DegenerateAdditiveInflation)(X, start::Int64, final::Int64)
    Ne = size(X,2)
    @assert A.Nx == final - start + 1 "final-start + 1 doesn't match the length of the additive noise"
    # @show X[start:final, 1]
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .+= A.Pϕ(A.m +  A.σ*randn(A.Nx))
    end
end

"""
(A::DegenerateAdditiveInflation)(X)

Apply the degenerate additive inflation `A` to an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ ∼ `A.α`.
"""
(A::DegenerateAdditiveInflation)(X) = A(X, 1, size(X,1))

"""
(A::DegenerateAdditiveInflation)(x::Array{Float64,1})

Apply the degenerate additive inflation `A` to the vector `x`,
i.e. x -> x + ϵ with ϵ ∼ drawn from a Gaussian distribution.
"""
function (A::DegenerateAdditiveInflation)(x::Array{Float64,1})
    x .+=  A.Pϕ(A.m + A.σ*randn(A.Nx))
    return x
end

