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
