@doc raw"""
    Ksupply(RB_guess,R_guess,w_guess,profit_guess,n_par,m_par)

Calculate the aggregate savings when households face idiosyncratic income risk.

Idiosyncratic state is tuple ``(m,k,y)``, where
``m``: liquid assets, ``k``: illiquid assets, ``y``: labor income

# Arguments
- `R_guess`: real interest rate illiquid assets
- `RB_guess`: nominal rate on liquid assets
- `w_guess`: wages
- `profit_guess`: profits
- `n_par::NumericalParameters`
- `m_par::ModelParameters`

# Returns
- `K`,`B`: aggregate saving in illiquid (`K`) and liquid (`B`) assets
-  `TransitionMat`,`TransitionMat_a`,`TransitionMat_n`: `sparse` transition matrices
    (average, with [`a`] or without [`n`] adjustment of illiquid asset)
- `distr`: ergodic steady state of `TransitionMat`
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
- `V_m`,`V_k`: marginal value functions
"""
function Ksupply(r_guess::Float64, n_par::NumericalParameters,
    m_par::ModelParameters, Vm::AbstractArray, inc::AbstractArray)

    #   initialize distance variables
    dist                = 9999.0

    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------
    count               = 0
    n                   = size(Vm)
    # containers for policies
    m_n_star            = zeros(n)
    c_n_star            = zeros(n)

    while dist > n_par.ϵ && count < 10000 # Iterate consumption policies until converegence
        count           = count + 1
        # Take expectations for labor income change
        EVm             = r_guess .* Vm * n_par.Π'

        # Policy update step
        c_n_star, m_n_star =
            EGM_policyupdate(EVm, r_guess, inc, n_par, m_par, false)

        # marginal value update step
        
        Vm_new   = mutil(c_n_star)  # Expected marginal utility at consumption policy (w &w/o adjustment)

        # Calculate distance in updates
        dist     = maximum(abs, invmutil(Vm_new) .- invmutil(Vm))


        # update policy guess/marginal values of liquid/illiquid assets
        Vm       = Vm_new
    end
    # println("EGM Iterations: ", count)    
    #------------------------------------------------------
    # Find stationary distribution (Is direct transition better for large model?)
    #------------------------------------------------------

    # Define transition matrix
    S_n, T_n, W_n    = MakeTransition(m_n_star, n_par.Π, n_par)
    TransitionMat    = sparse(S_n, T_n, W_n, n_par.nm * n_par.ny, n_par.nm *  n_par.ny)

    # Calculate left-hand unit eigenvector
    aux = real.(eigsolve(TransitionMat', 1)[2][1])
    distr = reshape((aux[:]) ./ sum((aux[:])),  (n_par.nm, n_par.ny))

    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    
    K = sum(distr[:] .* n_par.mesh_m[:])
    return K, TransitionMat, c_n_star, m_n_star, Vm, distr
end
