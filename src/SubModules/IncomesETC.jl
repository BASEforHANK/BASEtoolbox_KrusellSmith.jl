# ------------------------------------------------------------------------------
## Package Calls
# ------------------------------------------------------------------------------

module IncomesETC
# 3rd Party modules
using LinearAlgebra, Roots, ForwardDiff, PCHIPInterpolation

export incomes!,
    incomes,
    util,
    mutil!,
    mutil,
    invmutil!,
    invmutil,
    employment,
    interest,
    wage,
    output,
    integrate_capital

# ------------------------------------------------------------------------------
## Define Functions
# ------------------------------------------------------------------------------
include("../Model/IncomesETC/fcn_incomes.jl")
include("../Model/IncomesETC/fcn_util_etc.jl")
end # module BASEforHANK.IncomesETC
