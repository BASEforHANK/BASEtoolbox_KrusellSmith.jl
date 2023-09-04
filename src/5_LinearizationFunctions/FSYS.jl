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
    Γ::Array{Array{Float64,2},1},
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
    COP_Dev = pdf_to_cdf(COP_Dev)

    θDPrime = uncompress(compressionIndexes[2], XPrime[indexes.COP], DCD, IDCD)
    COP_DevPrime = reshape(copy(θDPrime), (n_par.nm_copula, n_par.ny_copula))
    COP_DevPrime = pdf_to_cdf(COP_DevPrime)

    # marginal distributions (pdfs, including steady state)
    distr_m = XSS[indexes.distr_mSS] .+ Γ[1] * X[indexes.distr_m]
    distr_y = XSS[indexes.distr_ySS] .+ Γ[2] * X[indexes.distr_y]

    distr_m_Prime = XSS[indexes.distr_mSS] .+ Γ[1] * XPrime[indexes.distr_m]
    distr_y_Prime = XSS[indexes.distr_ySS] .+ Γ[2] * XPrime[indexes.distr_y]

    # marginal distributions (cdfs) 
    CDF_m = cumsum(distr_m[:])
    CDF_y = cumsum(distr_y[:])

    ############################################################################
    # I.3. Read out steady state distributions
    ############################################################################

    # steads state cdfs (on value grid)
    CDF_mSS = cumsum(XSS[indexes.distr_mSS]) .+ zeros(eltype(θD), n_par.nm)
    CDF_ySS = cumsum(XSS[indexes.distr_ySS]) .+ zeros(eltype(θD), n_par.ny)

    # steady state copula (on copula grid)
    COPSS = reshape(XSS[indexes.COPSS] .+ zeros(eltype(θD), 1), (n_par.nm, n_par.ny))
    COPSS = pdf_to_cdf(COPSS)

    # steady state copula marginals (cdfs) 
    s_m_m = n_par.copula_marginal_m .+ zeros(eltype(θD), 1)
    s_m_y = n_par.copula_marginal_y .+ zeros(eltype(θD), 1)

    ############################################################################
    # I.4. Produce perturbed joint distribution using the copula
    ############################################################################
    # Copula(x::AbstractVector,y::AbstractVector,z::AbstractVector) = 
    # myAkimaInterp3(CDF_mSS, CDF_kSS, CDF_ySS, COPSS, x, y, z) .+
    # myAkimaInterp3(s_m_m, s_m_k, s_m_y, COP_Dev, x, y, z)

    Copula(x::Vector, y::Vector) =
        mylinearinterpolate2(CDF_mSS, CDF_ySS, COPSS, x, y) .+
        mylinearinterpolate2(s_m_m, s_m_y, COP_Dev, x, y)

    CDF_joint = Copula(CDF_m[:], CDF_y[:]) # roughly 5% of time
    PDF_joint = cdf_to_pdf(CDF_joint)

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
    F = Fsys_agg(X, XPrime, XSS, PDF_joint, m_par, n_par, indexes)


    ###########################################################################
    # Replace distributional objects used in Fsys_agg
    ###########################################################################
    # Average Human Capital =
    # average productivity (at the productivit grid, used to normalize to 0)
    H = dot(distr_y[1:end], n_par.grid_y[1:end])
    TotAsset = dot(n_par.grid_m, distr_m[:])

    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    inc = incomes(n_par, m_par, r, w, N)

    # Calculate optimal policies
    # expected margginal values
    EVmPrime = (reshape(VmPrime, (n_par.nm, n_par.ny)) * n_par.Π')

    c_n_star, m_n_star = EGM_policyupdate(EVmPrime, r, inc, n_par, m_par, false) # policy iteration (assume HH hold deposits)

    # Update marginal values
    Vm_err = rPrime .* mutil(c_n_star, m_par)  # update expected marginal values time t

    # Update distribution
    dist_aux = DirectTransition(m_n_star, PDF_joint, n_par.Π, n_par)
    PDF_jointPrime = reshape(dist_aux, n_par.nm, n_par.ny)

    #----------------------------------------------------------------------------------------
    # Calculate Error Terms
    #----------------------------------------------------------------------------------------
    # Error terms on marginal values (controls)
    invmutil!(Vm_err, Vm_err, m_par)
    Vm_err .= log.(Vm_err) .- reshape(VmSS, (n_par.nm, n_par.ny))
    Vm_thet = compress(compressionIndexes[1], Vm_err, DC, IDC)
    F[indexes.Vm] = X[indexes.Vm] .- Vm_thet

    # Error Terms on marginal distribution (in levels, states)
    distr_mPrimeUpdate = dropdims(sum(PDF_jointPrime, dims = (2)), dims = (2))
    distr_yPrimeUpdate = (distr_y'*n_par.Π)[:]
    F[indexes.distr_m] = (distr_mPrimeUpdate.-distr_m_Prime)[1:end-1]
    F[indexes.distr_y] = (distr_yPrimeUpdate.-distr_y_Prime[:])[1:end-1]

    # Error Terms on Copula (states)
    # Deviation of iterated copula from fixed copula
    # CopulaDevPrime(x::AbstractVector,y::AbstractVector) = 
    # myAkimaInterp2(CDF_m_PrimeUp, CDF_y_PrimeUp, pdf_to_cdf(PDF_jointPrime), x, y) .-
    # myAkimaInterp2(CDF_mSS, CDF_ySS, COPSS, x, y)
    CDF_m_PrimeUp = cumsum(distr_mPrimeUpdate)
    CDF_y_PrimeUp = cumsum(distr_yPrimeUpdate)
    CopulaDevPrime(x::Vector, y::Vector) =
        mylinearinterpolate2(
            CDF_m_PrimeUp,
            CDF_y_PrimeUp,
            pdf_to_cdf(PDF_jointPrime),
            x,
            y,
        ) .- mylinearinterpolate2(CDF_mSS, CDF_ySS, COPSS, x, y)

    CDF_Dev = CopulaDevPrime(s_m_m, s_m_y) # interpolate deviations on copula grid
    COP_thet =
        compress(compressionIndexes[2], cdf_to_pdf(CDF_Dev - COP_DevPrime), DCD, IDCD) # calculate DCT of deviations

    F[indexes.COP] = COP_thet

    # Calculate distribution statistics (generalized moments)
    _, _, TOP10WshareT, TOP10IshareT, GiniWT, GiniCT, sdlogyT =
        distrSummaries(PDF_joint, c_n_star, n_par, inc, m_par)

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
        return F, c_n_star, m_n_star, Vm_new
    end
end
