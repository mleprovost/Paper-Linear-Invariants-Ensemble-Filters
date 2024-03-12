module InvariantDA

using LinearAlgebra
using DocStringExtensions
using FFTW
using Statistics
using Distributions
using TransportBasedInference
using ProgressMeter
using OrdinaryDiffEq
using SparseArrays
using BandedMatrices

# Ensemble Kalman filters
include("filters/lik_enkf.jl")
include("filters/loc_lik_senkf.jl")
include("filters/loc_enkf.jl")

# Linear advection problem
include("linear_advection/linear_advection.jl")
include("linear_advection/generate_data.jl")
include("linear_advection/seq_assim_linear_advection.jl")
include("linear_advection/seq_assim_project_mass_linear_advection.jl")

# Toy problem
include("toy_problem/toy_problem.jl")
include("toy_problem/generate_data.jl")
include("toy_problem/seq_assim_toy_problem.jl")
include("toy_problem/seq_assim_project_mass_toy_problem.jl")


# Tools for the projection
include("tools/real_module.jl")
include("tools/projector.jl")
include("tools/degenerate_inflation.jl")
include("tools/multiplicative_degenerate_inflation.jl")


end # module InvariantDA
