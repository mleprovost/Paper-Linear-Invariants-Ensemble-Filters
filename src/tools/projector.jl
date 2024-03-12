export Projector, OrthoProjector

# An abtract type for Projector
abstract type Projector end

# Create a Orthonormal Projector type as a subtype of Projector
struct OrthoProjector <: Projector

    "Basis for linear subspace"
    Uperp::Union{Vector{Float64}, Matrix{Float64}}

    function OrthoProjector(Uperp)
        if typeof(Uperp) <: AbstractVector
            @assert norm(Uperp'*Uperp - 1.0) < 100*eps() "Column Uperp is not orthonormal"
        elseif typeof(Uperp) <: AbstractMatrix
            @assert norm(Uperp'*Uperp - 1.0*I) < 100*eps() "Columns of Uperp are not orthonormal"
        end
        return new(Uperp)
    end
end

# Compute the action of an orthogonal projector on a vector x, i.e.,
# P(x) = (Id - Uperp * Uperp')*x = x - Uperp * Uperp' * x
(P::OrthoProjector)(x::AbstractVector) = x - P.Uperp*(P.Uperp'*x)

