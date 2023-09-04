@doc raw"""
    Ksupply(r_guess::Float64,n_par::NumericalParameters,m_par::ModelParameters,Vm::AbstractArray,inc::AbstractArray)

Calculate the aggregate savings when households face idiosyncratic income risk.

Idiosyncratic state is tuple ``(m,y)``, where
``m``: liquid assets, ``y``: labor income

# Arguments
- `r_guess`: real interest rate illiquid assets
- `n_par::NumericalParameters`
- `m_par::ModelParameters`
- `Vm::AbstractArray`: marginal value of liquid assets
- `distr_guess::AbstractArray`: initial guess for stationary distribution `distr_guess[m,y]
- `inc::AbstractArray`: income array

# Returns
- `K`: aggregate savings (`K`)
-  `TransitionMat`: transition matrices
- `distr`: ergodic steady state of `TransitionMat`
- `c_star`,`m_star`: optimal policies for
    consumption [`c`], assets [`m`]
- `V_m`: marginal value function
"""
function Ksupply(
    r_guess::Float64,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    Vm::AbstractArray,
    distr_guess::AbstractArray,
    inc::AbstractArray,
)

    #   initialize distance variables
    dist = 9999.0
    dist1 = dist
    dist2 = dist

    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------
    count = 0
    # containers for policies, marginal value functions etc.
    m_n_star = similar(Vm)
    c_n_star = similar(Vm)
    EVm = similar(Vm)
    Vm_new = similar(Vm)
    iVm = invmutil(Vm, m_par)
    iVm_new = similar(iVm)
    EMU = similar(EVm)
    c_star_n = similar(EVm)
    m_star_n = similar(EVm)
    D = similar(EVm)

    while dist > n_par.ϵ && count < 10000 # Iterate consumption policies until converegence
        count = count + 1
        # Take expectations for labor income change
        EVm = r_guess .* Vm * n_par.Π'

        # Policy update step
        EGM_policyupdate!(
            c_n_star,
            m_n_star,
            EMU,
            c_star_n,
            m_star_n,
            EVm,
            r_guess,
            inc,
            n_par,
            m_par,
            false,
        )

        # marginal value update step
        Vm_new = mutil(c_n_star, m_par)
        invmutil!(iVm_new, Vm_new, m_par)
        D .= iVm_new .- iVm
        dist = maximum(abs, D)

        # update policy guess/marginal values of liquid/illiquid assets
        Vm .= Vm_new
        iVm .= iVm_new
    end
    println("EGM Iterations: ", count)
    println("EGM Dist: ", dist)
    #------------------------------------------------------
    # Find stationary distribution (Is direct transition better for large model?)
    #------------------------------------------------------

    # Define transition matrix
    S_n, T_n, W_n = MakeTransition(m_n_star, n_par.Π, n_par)
    Γ = sparse(S_n, T_n, W_n, n_par.nm * n_par.ny, n_par.nm * n_par.ny)

    # Calculate left-hand unit eigenvector
    aux = real.(eigsolve(Γ', distr_guess[:], 1)[2][1])

    ## Exploit that the Eigenvector of eigenvalue 1 is the nullspace of TransitionMat' -I
    #     Q_T = LinearMap((dmu, mu) -> dist_change!(dmu, mu, Γ), n_par.nm * n_par.nk * n_par.ny, ismutating = true)
    #     aux = fill(1/(n_par.nm * n_par.nk * n_par.ny), n_par.nm * n_par.nk * n_par.ny)#distr_guess[:] # can't use 0 as initial guess
    #     gmres!(aux, Q_T, zeros(n_par.nm * n_par.nk * n_par.ny))  # i.e., solve x'(Γ-I) = 0 iteratively
    ##qr algorithm for nullspace finding
    #     aux2 = qr(Γ - I)
    #     aux = Array{Float64}(undef, n_par.nm * n_par.nk * n_par.ny)
    #     aux[aux2.prow] = aux2.Q[:,end]
    #
    distr = (reshape((aux[:]) ./ sum((aux[:])), (n_par.nm, n_par.ny)))

    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    K = sum(distr[:] .* n_par.mesh_m[:])
    return K, Γ, c_n_star, m_n_star, Vm, distr
end
