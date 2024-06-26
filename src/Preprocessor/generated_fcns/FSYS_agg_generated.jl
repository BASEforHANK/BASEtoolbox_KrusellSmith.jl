# This file has been automatically generated by PreprocessInputs.jl. Any user inputs might be overwritten!




@doc raw"""
    Fsys_agg(X, XPrime, XSS, CDFSS, m_par, n_par, indexes)

Return deviations from aggregate equilibrium conditions.

`indexes` can be both `IndexStruct` or `IndexStructAggr`; in the latter case
(which is how function is called by [`LinearSolution_estim()`](@ref)), variable-vectors
`X`,`XPrime`, and `XSS` only contain the aggregate variables of the model.
"""
function Fsys_agg(
    X::AbstractArray,
    XPrime::AbstractArray,
    XSS::Array{Float64,1},
    CDFSS::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    indexes::Union{IndexStructAggr,IndexStruct},
)
    # The function call with Duals takes
    # Reserve space for error terms
    F = zeros(eltype(X), size(X))
    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(XSS[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(XSS[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # @generate_equations(aggr_names)
    @generate_equations()

    # Take aggregate model from model file


#------------------------------------------------------------------------------
# THIS FILE CONTAINS THE "AGGREGATE" MODEL EQUATIONS, I.E. EVERYTHING  BUT THE 
# HOUSEHOLD PLANNING PROBLEM. THE lATTER IS DESCRIBED BY ONE EGM BACKWARD STEP AND 
# ONE FORWARD ITERATION OF THE DISTRIBUTION.
#
# AGGREGATE EQUATIONS TAKE THE FORM 
# F[EQUATION NUMBER] = lhs - rhs
#
# EQUATION NUMBERS ARE GENEREATED AUTOMATICALLY AND STORED IN THE INDEX STRUCT
# FOR THIS THE "CORRESPONDING" VARIABLE NEEDS TO BE IN THE LIST OF STATES 
# OR CONTROLS.
#------------------------------------------------------------------------------



############################################################################
#           Error term calculations (i.e. model starts here)          #
############################################################################

# constant N
N = m_par.N

# Shocks
F[indexes.Z] = log.(ZPrime) - m_par.ρ_Z * log.(Z)
F[indexes.delta] =
    log.(deltaPrime) - m_par.ρ_delta * log.(delta) -
    (1 - m_par.ρ_delta) * XSS[indexes.deltaSS]

# Prices
F[indexes.r] = log.(r) - log.(interest(K, Z, N, m_par; delta = delta - 1.0) + 1.0)       # rate of return on capital
F[indexes.w] = log.(w) - log.(wage(K, Z, N, m_par))     # wages that firms pay

# Aggregate Quantities
F[indexes.I] = KPrime .- K .* (1.0 .- (delta .- 1.0)) .- I          # Capital accumulation equation
F[indexes.Y] = log.(Y) - log.(Z .* N .^ (1.0 .- m_par.α) .* K .^ m_par.α)                                          # production function
F[indexes.C] = log.(Y .- I) .- log(C)                            # Resource constraint

# Capital market clearing
F[indexes.K] = log.(K) - (XSS[indexes.KSS])

# Lags
F[indexes.Ylag] = log(YlagPrime) - log(Y)

# Growth rates
F[indexes.Ygrowth] = log(Ygrowth) - log(Y / Ylag)


    # @include "../3_Model/input_aggregate_model.jl"

    return F
end
