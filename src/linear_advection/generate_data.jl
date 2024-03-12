export generate_data_rfft

"""
generate_data_rfft(model::Model, x0, J::Int64)

A routine to generate ground truth state and observation samples over `J` times steps with initial condition `x0`
This version is designed for dynamical models formulated in the spectral domain
and taking as input the real FFT (rfft) of the state components. 
"""
function generate_data_rfft(model::Model, x0, J::Int64)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)

    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = ODEProblem(model.F.f,deepcopy(rfft(x)),tspan)

    # Run dynamics and save results
    for i=1:J

        # Advection is made in the Fourier domain
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, u0 = deepcopy(rfft(x)), tspan = tspan)
    	sol = solve(prob, SSPRK43(), adaptive = true,
		            dense = false, save_everystep = false)
    	x .= deepcopy(irfft(sol.u[end], model.Nx))
    	model.ϵx(x)

    	# Collect observations
    	tt[i] = deepcopy(i*model.Δtobs)
    	xt[:,i] = deepcopy(x)
		yt[:,i] = deepcopy(model.F.h(x, tt[i]))

		# Perturb the observations
		if typeof(model.ϵy) <: AdditiveInflation
			yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
		end
    end
	
    return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end