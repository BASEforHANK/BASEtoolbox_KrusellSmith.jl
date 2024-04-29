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
- `CDF_guess::AbstractArray`: initial guess for stationary cumulative joint distribution (in capital) `CDF_guess[m,y]
- `inc::AbstractArray`: income array

# Returns
- `K`: aggregate savings (`K`)
-  `TransitionMat`: transition matrices
- `CDF_joint`: ergodic steady state for CDF over income states
- `c_star`,`m_star`: optimal policies for
    consumption [`c`], assets [`m`]
- `V_m`: marginal value function
"""
function Ksupply(
    r_guess::Float64,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    Vm::AbstractArray,
    CDF_guess::AbstractArray,
    inc::AbstractArray,
)

    # 1. Find steady state policies given the interest rate guess.
    c_n_star, m_n_star, Vm, m_star_n = find_ss_policies(r_guess, n_par, m_par, Vm, inc)

    # 2. Find stationary distribution given optimal policy.
    CDF_joint = find_ss_distribution_splines(m_n_star, CDF_guess, n_par)

    # 3. Calculate capital stock
    CDF_m = sum(CDF_joint, dims = 2)[:]
    K = integrate_capital(CDF_m, n_par)

    return K, c_n_star, m_n_star, Vm, CDF_joint
end

@doc raw"""
    find_ss_policies(
        r_guess::Float64,
        n_par::NumericalParameters,
        m_par::ModelParameters,
        Vm::AbstractArray,
        inc::AbstractArray,
    )

    Find steady state policies given guess for r.

    # Arguments
    - `r_guess`: real interest rate illiquid assets
    - `n_par::NumericalParameters`
    - `m_par::ModelParameters`
    - `Vm::AbstractArray`: marginal value of liquid assets
    - `inc::AbstractArray`: income array

    # Returns
    - `c_star`,`m_star`: optimal policies for
        consumption [`c`], assets [`m`]
    - `V_m`: marginal value function
"""
function find_ss_policies(
    r_guess::Float64,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    Vm_initial::AbstractArray,
    inc::AbstractArray,
)
    Vm = copy(Vm_initial)
    #   initialize distance variables
    dist = 9999.0
    # Iterate over consumption policies
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
        Vm_new .= mutil(c_n_star, m_par)
        invmutil!(iVm_new, Vm_new, m_par)
        D .= iVm_new .- iVm
        dist = maximum(abs, D)

        # update policy guess/marginal values of liquid/illiquid assets
        Vm .= Vm_new
        iVm .= iVm_new
    end
    if n_par.verbose
        println("EGM Iterations: ", count)
        println("EGM Dist: ", dist)
    end

    return c_n_star, m_n_star, Vm, m_star_n
end


@doc raw"""
    find_ss_distribution_splines(
        m_n_star::AbstractArray,
        cdf_guess_intial::AbstractArray,
        n_par::NumericalParameters,
        )

    Find steady state disribution given steady state policies and guess for the interest 
        rate through iteration. Iteration of distribution is done using monotonic spline
        interpolation to bring the next periods cdf's back to the reference grid.

    # Arguments
    - `m_n_star::AbstractArray`: optimal savings choice given fixed grid of assets and income shocks
    - `cdf_guess_intial::AbstractArray`: initial guess for stationary cumulative joint distribution (in m) `cdf_guess[m,y]
    - `n_par::NumericalParameters`


    # Returns
    - `pdf_ss`: stationary distribution over m and y

"""
function find_ss_distribution_splines(
    m_n_star::AbstractArray,
    cdf_guess_intial::AbstractArray,
    n_par::NumericalParameters,
)

    # Tolerance for change in cdf from period to period
    tol = n_par.ϵ
    # Maximum iterations to find steady state distribution
    max_iter = 100000

    # # Calculate cdf over individual income states
    # cdf_guess = cumsum(distr_guess, dims=1)
    cdf_guess = copy(cdf_guess_intial)

    # Iterate on distribution until convergence
    distance = 9999.0
    iter = 0
    while distance > tol && iter < max_iter
        iter = iter + 1
        cdf_guess_old = copy(cdf_guess)
        DirectTransition_Splines!(cdf_guess, m_n_star, cdf_guess_old, n_par.Π, n_par)
        difference = cdf_guess_old .- cdf_guess
        distance = maximum(abs, difference)
    end

    cdf_ss = cdf_guess

    if n_par.verbose
        println("Distribution Iterations: ", iter)
        println("Distribution Dist: ", distance)
    end

    return cdf_ss
end


@doc raw"""
    find_ss_distribution_young(
        m_n_star::AbstractArray,
        distr_guess::AbstractArray,
        n_par::NumericalParameters,
        )

    Find steady state disribution given steady state policies and guess for the interest 
        rate through iteration. Iteration of distribution is done using monotonic spline
         interpolation to bring the next periods cdf's back to the reference grid.

    # Arguments
    - `m_n_star::AbstractArray`: optimal policies for assets
    - `distr_guess::AbstractArray`: initial guess for stationary distribution `distr_guess[m,y]
    - `n_par::NumericalParameters`

    # Returns
    - `distr`: stationary distribution over m and y
    - `Γ`: transition matrix

"""
function find_ss_distribution_young(
    m_n_star::AbstractArray,
    n_par::NumericalParameters,
    distr_guess::AbstractArray,
)
    # Define transition matrix
    S_n, T_n, W_n = MakeTransition(m_n_star, n_par.Π, n_par)
    Γ = sparse(S_n, T_n, W_n, n_par.nm * n_par.ny, n_par.nm * n_par.ny)

    # Calculate left-hand unit eigenvector
    aux = real.(eigsolve(Γ', distr_guess[:], 1)[2][1])
    # get stationary distribution
    distr = (reshape((aux[:]) ./ sum((aux[:])), (n_par.nm, n_par.ny)))

    return distr, Γ
end
