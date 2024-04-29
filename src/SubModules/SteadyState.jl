# __precompile__(false)
# Code runs on Julia 1.9.3
# ------------------------------------------------------------------------------
## Package Calls
# ------------------------------------------------------------------------------

module SteadyState
# Sibling modules
using ..Tools
using ..Parsing
using ..IncomesETC

# 3rd Party modules
using LinearAlgebra,
    SparseArrays, Distributions, Roots, ForwardDiff, Flatten, PCHIPInterpolation

using Parameters: @unpack
using KrylovKit: eigsolve
using FFTW: dct, ifft

export EGM_policyupdate!,
    EGM_policyupdate, DirectTransition, DirectTransition_Splines, Ksupply, find_steadystate


# ------------------------------------------------------------------------------
## Define Functions
# ------------------------------------------------------------------------------
include("../Model/input_aggregate_names.jl") # this dependency is not ideal
include("SteadyState/EGM/EGM_policyupdate.jl")
include("SteadyState/find_steadystate.jl")
include("SteadyState/IM_fcns/fcn_directtransition.jl")
include("SteadyState/IM_fcns/fcn_kdiff.jl")
include("SteadyState/IM_fcns/fcn_ksupply.jl")
include("SteadyState/IM_fcns/fcn_makeweights.jl")
include("SteadyState/IM_fcns/fcn_maketransition.jl")
end # module BASEforHANK.SteadyState
