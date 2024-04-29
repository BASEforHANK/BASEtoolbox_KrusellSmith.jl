@doc raw"""
    Fsys(X, XPrime, XSS, m_par, n_par, indexes, Γ, compressionIndexes, DC, IDC, DCD, IDCD)

Equilibrium error function: returns deviations from equilibrium around steady state.

Split computation into *Aggregate Part*, handled by [`Fsys_agg()`](@ref),
and *Heterogeneous Agent Part*.

# Arguments
- `X`,`XPrime`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `XSS`: states and controls in steady state
- `Γ`, `DC`, `IDC`, `DCD`,`IDCD`: transformation matrices to retrieve marginal distributions [`Γ`],
    marginal value functions [`DC`,`IDC`], and the (linear) interpolant of the copula [`DCD`,`IDCD`] from deviations
- `indexes`,`compressionIndexes`: access `XSS` by variable names
    (DCT coefficients of compressed ``V_m`` and ``V_k`` in case of `compressionIndexes`)

# Example
```jldoctest
julia> # Solve for steady state, construct Γ,DC,IDC as in LinearSolution()
julia> Fsys(zeros(ntotal),zeros(ntotal),XSS,m_par,n_par,indexes,Γ,compressionIndexes,DC,IDC)
*ntotal*-element Array{Float64,1}:
 0.0
 0.0
 ...
 0.0
```
"""
function Fsys(
    X::AbstractArray,
    XPrime::AbstractArray,
    XSS::Array{Float64,1},
    m_par::ModelParameters,
    n_par::NumericalParameters,
    indexes::IndexStruct,
    Γ::Array{Float64,2},
    compressionIndexes::Array{Array{Int,1},1},
    DC::Array{Array{Float64,2},1},
    IDC::Array{Adjoint{Float64,Array{Float64,2}},1},
    DCD::Array{Array{Float64,2},1},
    IDCD::Array{Adjoint{Float64,Array{Float64,2}},1};
    only_F = true,
)
    # The function call with Duals takes
    # Reserve space for error terms
    F = zeros(eltype(X), size(X))

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################
    # rougly 10% of computing time, more if uncompress is actually called

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(XSS[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(XSS[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # @generate_equations(aggr_names)
    @generate_equations()


    ############################################################################
    # I.2. Read out  perturbed distributions
    ############################################################################

    # Copula parameters (deviations from steads state)
    θD = uncompress(compressionIndexes[2], X[indexes.COP], DCD, IDCD)
    COP_Dev = reshape(copy(θD[:]), (n_par.nm_copula, n_par.ny_copula))
    # COP_Dev = pdf_to_cdf(COP_Dev)

    θDPrime = uncompress(compressionIndexes[2], XPrime[indexes.COP], DCD, IDCD)
    COP_DevPrime = reshape(copy(θDPrime), (n_par.nm_copula, n_par.ny_copula))

    # marginal distributions (pdfs, including steady state)

    CDF_m = XSS[indexes.CDF_mSS] .+ [X[indexes.CDF_m]; 0] # Last value of CDF is always 1

    PDF_y = XSS[indexes.PDF_ySS] .+ Γ * X[indexes.PDF_y]
    CDF_y = cumsum(PDF_y)

    CDF_m_Prime = XSS[indexes.CDF_mSS] .+ [XPrime[indexes.CDF_m]; 0] # Last value of CDF is always 1

    PDF_y_Prime = XSS[indexes.PDF_ySS] .+ Γ * XPrime[indexes.PDF_y]

    ############################################################################
    # I.3. Read out steady state distributions
    ############################################################################

    # steads state cdfs (on value grid)
    CDF_mSS = XSS[indexes.CDF_mSS] .+ zeros(eltype(θD), n_par.nm)
    PDF_ySS = XSS[indexes.PDF_ySS] .+ zeros(eltype(θD), n_par.ny)
    CDF_ySS = cumsum(PDF_ySS)

    # steady state copula (on copula grid)
    COPSS = reshape(XSS[indexes.COPSS] .+ zeros(eltype(θD), 1), (n_par.nm, n_par.ny))

    # steady state copula marginals (cdfs) 
    s_m_m = n_par.copula_marginal_m .+ zeros(eltype(θD), 1)
    s_m_y = n_par.copula_marginal_y .+ zeros(eltype(θD), 1)

    ############################################################################
    # I.4. Produce perturbed joint distribution using the copula
    ############################################################################

    Copula(x::Vector, y::Vector) =
        mylinearinterpolate2(CDF_mSS, CDF_ySS, COPSS, x, y) .+
        mylinearinterpolate2(s_m_m, s_m_y, COP_Dev, x, y)

    CDF_joint = Copula(CDF_m[:], CDF_y[:])

    ############################################################################
    # I.5 uncompressing policies/value functions
    ###########################################################################
    VmSS = XSS[indexes.VmSS]
    VmPrime = mutil(
        exp.(VmSS .+ uncompress(compressionIndexes[1], XPrime[indexes.Vm], DC, IDC)),
        m_par,
    )

    ############################################################################
    #           III. Error term calculations (i.e. model starts here)          #
    ############################################################################

    ############################################################################
    #           III. 1. Aggregate Part #
    ############################################################################
    F = Fsys_agg(X, XPrime, XSS, CDF_joint, m_par, n_par, indexes)


    ###########################################################################
    # Replace distributional objects used in Fsys_agg
    ###########################################################################
    TotAsset = integrate_capital(CDF_m, n_par)

    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    N = m_par.N # constant N
    inc = incomes(n_par, m_par, r, w, N)

    # Calculate optimal policies
    # expected margginal values
    EVmPrime = (reshape(VmPrime, (n_par.nm, n_par.ny)) * n_par.Π')

    c_n_star, m_n_star = EGM_policyupdate(EVmPrime, r, inc, n_par, m_par, false) # policy iteration (assume HH hold deposits)

    # Update marginal values
    Vm_err = r .* mutil(c_n_star, m_par)  # update expected marginal values time t

    # Update cumulative joint distribution (ina) 
    CDF_jointPrime = DirectTransition_Splines(m_n_star, CDF_joint, n_par.Π, n_par)

    #----------------------------------------------------------------------------------------
    # Calculate Error Terms
    #----------------------------------------------------------------------------------------
    # Error terms on marginal values (controls)
    invmutil!(Vm_err, Vm_err, m_par)
    Vm_err .= log.(Vm_err) .- reshape(VmSS, (n_par.nm, n_par.ny))
    Vm_thet = compress(compressionIndexes[1], Vm_err, DC, IDC)
    F[indexes.Vm] = X[indexes.Vm] .- Vm_thet


    # Error Terms on marginal distribution (in levels, states)
    CDF_m_PrimeUp = sum(CDF_jointPrime, dims = 2)[:]
    PDF_y_PrimeUp = (PDF_ySS'*n_par.Π)[:]
    CDF_y_PrimeUp = cumsum(PDF_y_PrimeUp)


    F[indexes.CDF_m] = (CDF_m_PrimeUp.-CDF_m_Prime)[1:end-1]
    F[indexes.PDF_y] = (PDF_y_PrimeUp.-PDF_y_Prime)[1:end-1]

    CopulaDevPrime(x::Vector, y::Vector) =
        mylinearinterpolate2(CDF_m_PrimeUp, CDF_y_PrimeUp, CDF_jointPrime, x, y) .-
        mylinearinterpolate2(CDF_mSS, CDF_ySS, COPSS, x, y)

    CDF_Dev = CopulaDevPrime(s_m_m, s_m_y) # interpolate deviations on copula grid
    COP_thet = compress(compressionIndexes[2], CDF_Dev - COP_DevPrime, DCD, IDCD) # calculate DCT of deviations

    F[indexes.COP] = COP_thet

    # Calculate distribution statistics (generalized moments)
    _, _, TOP10WshareT, TOP10IshareT, GiniWT, GiniCT, sdlogyT =
        distrSummaries(CDF_joint, c_n_star, n_par, inc, m_par)

    # Error Term on prices/aggregate summary vars (logarithmic, controls)
    F[indexes.K] = log.(K) - log.(TotAsset)

    # Error Terms on  distribution summaries
    F[indexes.GiniW] = log.(GiniW) - log.(GiniWT)
    F[indexes.TOP10Ishare] = log.(TOP10Ishare) - log.(TOP10IshareT)
    F[indexes.TOP10Wshare] = log.(TOP10Wshare) - log.(TOP10WshareT)
    F[indexes.GiniC] = log.(GiniC) - log.(GiniCT)
    F[indexes.sdlogy] = log.(sdlogy) - log.(sdlogyT)


    if only_F
        return F
    else
        Vm_star = r .* mutil(c_n_star, m_par)
        return F, c_n_star, m_n_star, Vm_star
    end
end
