@testset "Test degenerate inflation for single column" begin
    # Column vector
    Nx = 5
    σx = 1.0

    ϕ = ones(Nx)
    ϕ = ϕ/norm(ϕ)
    
    x = randn(Nx)
    mx = copy(ϕ'*x)
    
    ϵx = DegenerateAdditiveInflation(zeros(Nx), σx, ϕ)
    
    #Apply inflation
    ϵx(x)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*x, mx, atol = 100*eps())
    

    # Check on an ensemble matrix
    Ne = 20
    X = randn(Nx, Ne)
    Mx = copy(ϕ'*X)
    
    #Apply inflation
    ϵx(X)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*X, Mx, atol = 100*eps())
        
    # Check if only applied to some rows of an ensemble matrix
    Ne = 20
    X = randn(10 + Nx, Ne)
    Xinfl = deepcopy(X)
    Mx = copy(ϕ'*X[10+1 : 10+Nx, :])
    
    
    #Apply inflation
    ϵx(Xinfl, 10 + 1, 10 + Nx)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(Xinfl[1 : 10, :], X[1 : 10, :], atol = 100*eps())
    # Check that the first 10 components haven't been modified
    @test isapprox(ϕ'*Xinfl[10+1 : 10+Nx, :], Mx, atol = 100*eps())
end

@testset "Test degenerate inflation for multi columns" begin
    # Column vector
    Nx = 5
    σx = 1.0

    ϕ = Matrix(qr(randn(Nx, 3)).Q)
    
    x = randn(Nx)
    mx = copy(ϕ'*x)
    
    ϵx = DegenerateAdditiveInflation(zeros(Nx), σx, ϕ)
    
    #Apply inflation
    ϵx(x)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*x, mx, atol = 100*eps())
    

    # Check on an ensemble matrix
    Ne = 20
    X = randn(Nx, Ne)
    Mx = copy(ϕ'*X)
    
    #Apply inflation
    ϵx(X)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*X, Mx, atol = 100*eps())
        
    # Check if only applied to some rows of an ensemble matrix
    Ne = 20
    X = randn(10 + Nx, Ne)
    Xinfl = deepcopy(X)
    Mx = copy(ϕ'*X[10+1 : 10+Nx, :])
    
    
    #Apply inflation
    ϵx(Xinfl, 10 + 1, 10 + Nx)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(Xinfl[1 : 10, :], X[1 : 10, :], atol = 100*eps())
    # Check that the first 10 components haven't been modified
    @test isapprox(ϕ'*Xinfl[10+1 : 10+Nx, :], Mx, atol = 100*eps())
end



@testset "Test degenerate inflation for multi columns" begin
    # Column vector
    Nx = 5
    σx = 1.0

    ϕ = ones(Nx)
    ϕ = ϕ/norm(ϕ)
    
    x = randn(Nx)
    mx = copy(ϕ'*x)
    
    ϵx = DegenerateAdditiveInflation(zeros(Nx), σx, ϕ)
    
    #Apply inflation
    ϵx(x)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*x, mx, atol = 100*eps())


    # Check on an ensemble matrix
    Ne = 20
    X = randn(Nx, Ne)
    Mx = copy(ϕ'*X)
    
    #Apply inflation
    ϵx(X)
    
    # Check that the projection along ϕ hasn't changed
    @test isapprox(ϕ'*X, X, atol = 100*eps())
    
end