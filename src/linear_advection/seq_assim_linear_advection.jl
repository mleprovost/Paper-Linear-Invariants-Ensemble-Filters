export seqassim_rfft

"""
seqassim_rfft(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64)

Generic API for sequential data assimilation for any sequential filter of parent type `SeqFilter`. 
This version is designed for dynamical models formulated in the spectral domain
and taking as input the real FFT (rfft) of the state samples. 
"""
function seqassim_rfft(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, 
                       X, Ny, Nx, t0::Float64)

Ne = size(X, 2)

step = ceil(Int, algo.Δtobs/algo.Δtdyn)

statehist = Array{Float64,2}[]
push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))

n0 = ceil(Int64, t0/algo.Δtobs) + 1
Acycle = n0:n0+J-1
tspan = (t0, t0 + algo.Δtobs)

prob = ODEProblem(F.f, rfft(zeros(Nx)), tspan)

# Run filtering algorithm
@showprogress for i=1:length(Acycle)

    # Forecast step
	tspan = (t0+(i-1)*algo.Δtobs, t0+i*algo.Δtobs)

    function  prob_func(prob,i,repeat)
        remake(prob, u0 = rfft(X[Ny+1:Ny+Nx,i]), tspan = tspan)
    end

    ensemble_prob = EnsembleProblem(prob,output_func = (sol,i) -> (sol[end], false),
                                    prob_func=prob_func)

	sim = solve(ensemble_prob, SSPRK43(), adaptive = true, EnsembleThreads(),trajectories = Ne,
					dense = false, save_everystep=false);

	@inbounds for i=1:Ne
	    X[Ny+1:Ny+Nx, i] .= deepcopy(irfft(sim[i], Nx))
	end


   	# Get ground truth observation ystar
    ystar = data.yt[:,Acycle[i]]

	# Perform inflation for each ensemble member
	ϵx(X, Ny+1, Ny+Nx)

	# Compute non-perturbed observations
	observe(F.h, X, t0+i*algo.Δtobs, Ny, Nx)

    # Generate filtering samples.
	# Note that the perturbation of the observations is performed within the sequential filter.

    X = algo(X, ystar, t0+i*algo.Δtobs-t0)

	# (Optional) post-processing of filtering samplesß
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