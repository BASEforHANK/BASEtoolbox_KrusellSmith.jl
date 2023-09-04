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

# Shocks
F[indexes.Z] = log.(ZPrime) - m_par.ρ_Z * log.(Z)               # TFP

# Prices
F[indexes.r] = log.(r) - log.(interest(K, Z, N, m_par) + 1.0)       # rate of return on capital
F[indexes.w] = log.(w) - log.(wage(K, Z, N, m_par))     # wages that firms pay

# Aggregate Quantities
F[indexes.I] = KPrime .- K .* (1.0 .- m_par.δ_0) .- I          # Capital accumulation equation
F[indexes.N] = log.(N) - log.(w .^ (1.0 / m_par.γ))   # labor supply
F[indexes.Y] = log.(Y) - log.(Z .* N .^ (1.0 .- m_par.α) .* K .^ m_par.α)                                          # production function
F[indexes.C] = log.(Y .- I) .- log(C)                            # Resource constraint

# Capital market clearing
F[indexes.K] = log.(K) - (XSS[indexes.KSS])


# Lags
F[indexes.Ilag] = log(IlagPrime) - log(I)
F[indexes.wlag] = log(wlagPrime) - log(w)
F[indexes.Nlag] = log(NlagPrime) - log(N)
F[indexes.Clag] = log(ClagPrime) - log(C)

# Growth rates
F[indexes.Ngrowth] = log(Ngrowth) - log(N / Nlag)
F[indexes.Igrowth] = log(Igrowth) - log(I / Ilag)
F[indexes.wgrowth] = log(wgrowth) - log(w / wlag)
F[indexes.Cgrowth] = log(Cgrowth) - log(C / Clag)
