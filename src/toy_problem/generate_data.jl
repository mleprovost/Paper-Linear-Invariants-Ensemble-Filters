export generate_data_toy_problem

"""
generate_data_toy_problem(model::Model, x0, J::Int64, Aop::MatrixOperator)

A routine to generate ground truth state and observation samples over `J` times steps with initial condition `x0` 
for the linear dynamical system dx/dt = Aop x.
This version doesn't use the dynamical model typically passed in `model.F` 
"""
function generate_data_toy_problem(model::Model, x0::Array{Float64,1}, J::Int64, Aop::MatrixOperator)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)
    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = ODEProblem(Aop,x,tspan)

    for i=1:J
    	# Run dynamics and save results
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, tspan = tspan)

    	sol = solve(prob, LinearExponential(),
		            dense = false, save_everystep = false)
    	x .= deepcopy(sol.u[end])

		# Apply process noise
    	model.ϵx(x)

    	# Collect observations
    	tt[i] = deepcopy(i*model.Δtobs)
    	xt[:,i] = deepcopy(x)

		# Generate non-perturbed observations
		yt[:,i] = deepcopy(model.F.h(x, tt[i]))

		# Apply observation noise
		yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end