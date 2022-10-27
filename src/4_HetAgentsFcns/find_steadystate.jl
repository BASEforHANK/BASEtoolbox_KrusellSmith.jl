@doc raw"""
    find_steadystate(m_par)

Find the stationary equilibrium capital stock.

# Returns
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_steadystate(m_par)

# -------------------------------------------------------------------------------
## STEP 1: Find the stationary equilibrium for coarse grid
# -------------------------------------------------------------------------------
#-------------------------------------------------------
# Income Process and Income Grids
#-------------------------------------------------------
# Read out numerical parameters for starting guess solution with reduced income grid.

# Numerical parameters 
n_par                   = NumericalParameters(m_par = m_par, naggrstates = length(state_names), naggrcontrols = length(control_names),
                                              aggr_names  = aggr_names, distr_names = distr_names)
if n_par.verbose
    println("Finding equilibrium capital stock for coarse income grid")
end
# Capital stock guesses
Kmin=10.0
Kmax=75.0
println("Kmin: ", Kmin)
println("Kmax: ", Kmax)

# a.) Define excess demand function
d(  K, 
    initial::Bool=true, 
    Vm_guess = zeros(1,1),  
    distr_guess = n_par.dist_guess) = Kdiff(K, n_par, m_par, initial, Vm_guess, distr_guess)

# Find stationary equilibrium for refined economy
BrentOut                = CustomBrent(d, Kmin, Kmax; tol = n_par.ϵ)
KSS                     = BrentOut[1]
VmSS                    = BrentOut[3][2]
distrSS                 = BrentOut[3][3]
if n_par.verbose
    println("Capital stock is")
    println(KSS)
end
return KSS, VmSS, distrSS, n_par, m_par

end

