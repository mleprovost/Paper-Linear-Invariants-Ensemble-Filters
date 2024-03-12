export seqassim_toy_problem

"""
seqassim_toy_problem(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64, Aop::MatrixOperator)

Generic API for sequential data assimilation for any sequential filter of parent type `SeqFilter` 
for the linear dynamical system dx/dt = Aop x.
This version doesn't use the dynamical model typically passed in the `StateSpace` object `F` 
"""
function seqassim_toy_problem(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, 
                       X, Ny, Nx, t0::Float64, Aop::MatrixOperator)

Ne = size(X, 2)

step = ceil(Int, algo.Δtobs/algo.Δtdyn)

statehist = Array{Float64,2}[]
push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))

n0 = ceil(Int64, t0/algo.Δtobs) + 1
Acycle = n0:n0+J-1
tspan = (t0, t0 + algo.Δtobs)

prob = ODEProblem(Aop, zeros(Nx), tspan)

# Run filtering algorithm
for i=1:length(Acycle)
    # Forecast
	tspan = (t0+(i-1)*algo.Δtobs, t0+i*algo.Δtobs)

    function  prob_func(prob,i,repeat)
        remake(prob, u0 = X[Ny+1:Ny+Nx,i], tspan = tspan)
    end

    ensemble_prob = EnsembleProblem(prob,output_func = (sol,i) -> (sol[end], false),
                                    prob_func=prob_func)
	sim = solve(ensemble_prob, LinearExponential(), EnsembleThreads(),trajectories = Ne,
					dense = false, save_everystep=false);

	@inbounds for i=1:Ne
	    X[Ny+1:Ny+Nx, i] .= deepcopy(sim[i])
	end

    # Obtain ground truth observation to assimilate
    ystar = data.yt[:,Acycle[i]]

	# Perform state inflation
	ϵx(X, Ny+1, Ny+Nx)

	# Compute non-perturbed observations
	observe(F.h, X, t0+i*algo.Δtobs, Ny, Nx)

    # Generate filtering samples.
	# Note that the perturbation of the observations is performed within the sequential filter.

    X = algo(X, ystar, t0+i*algo.Δtobs-t0)

	# (Optional) post-processing of the filtering samples
	if algo.isfiltered == true
		for i=1:Ne
			statei = view(X, Ny+1:Ny+Nx, i)
			statei .= algo.G(statei)
		end
	end

    push!(statehist, copy(X[Ny+1:Ny+Nx,:]))
	end

    return statehist
end
