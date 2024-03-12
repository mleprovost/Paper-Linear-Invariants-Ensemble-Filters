

@testset "Test orthonormal projector" begin
    # Column vector
    Nx = 5
    ϕ = ones(Nx)
    ϕ = ϕ/norm(ϕ)
    P = OrthoProjector(ϕ)
    x = randn(Nx)
 
    Ptrue_x = x - P.ϕ*P.ϕ'*x
    @test isapprox(P(x), Ptrue_x, atol = 100*eps())
        
    # Multi-columns
    
    Nx = 5
    ϕmulti = Matrix(qr(randn(Nx, 3)).Q)
    Pmulti = OrthoProjector(ϕ)
    x = randn(Nx)

    Pmultitrue_x = x - Pmulti.ϕ*Pmulti.ϕ'*x
    @test isapprox(Pmulti(x), Pmultitrue_x, atol = 100*eps())
end