@doc raw"""
    LinearSolution(sr, m_par, A, B; estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys()`](@ref), using Schmitt-Grohé & Uribe (JEDC 2004) style linearization
(apply the implicit function theorem to obtain linear observation and
state transition equations).

The Jacobian is calculated using the package `ForwardDiff`

# Arguments
- `sr`: steady-state structure (variable values, indexes, numerical parameters, ...)
- `A`,`B`: derivative of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
- `m_par`: model parameters

# Returns
- `gx`,`hx`: observation equations [`gx`] and state transition equations [`hx`]
- `alarm_LinearSolution`,`nk`: `alarm_LinearSolution=true` when solving algorithm fails, `nk` number of
    predetermined variables
- `A`,`B`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
"""
function LinearSolution(
    sr::SteadyResults,
    m_par::ModelParameters,
    A::Array,
    B::Array;
    estim = false,
)
    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrix to take care of reduced degree of freedom in marginal distribution
    Γ = shuffleMatrix_1dim(sr.PDF_y, sr.n_par.ny)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 2)
    DC[1] = mydctmx(sr.n_par.nm)
    DC[2] = mydctmx(sr.n_par.ny)
    IDC = [DC[1]', DC[2]']

    DCD = Array{Array{Float64,2},1}(undef, 2)
    DCD[1] = mydctmx(sr.n_par.nm_copula)
    DCD[2] = mydctmx(sr.n_par.ny_copula)
    IDCD = [DCD[1]', DCD[2]']

    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################
    length_X0 = sr.n_par.ntotal
    X0 = zeros(length_X0) .+ ForwardDiff.Dual(0.0, 0.0)
    F = Fsys(
        X0,
        X0,
        sr.XSS,
        m_par,
        sr.n_par,
        sr.indexes,
        Γ,
        sr.compressionIndexes,
        DC,
        IDC,
        DCD,
        IDCD,
    )

    FR = realpart.(F)
    println(findall(abs.(FR) .> 0.001))
    println("Number of States and Controls")
    println(length(F))
    println("Max error on Fsys:")
    println(maximum(abs.(FR[:])))
    println("Max error of COP in Fsys:")
    println(maximum(abs.(FR[sr.indexes.COP])))
    println("Max error of Vm in Fsys:")
    println(maximum(abs.(FR[sr.indexes.Vm])))

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################
    BA = ForwardDiff.jacobian(
        x -> Fsys(
            x[1:length_X0],
            x[length_X0+1:end],
            sr.XSS,
            m_par,
            sr.n_par,
            sr.indexes,
            Γ,
            sr.compressionIndexes,
            DC,
            IDC,
            DCD,
            IDCD,
            only_F = true,
        ),
        zeros(2 * length_X0),
    )

    B = BA[:, 1:length_X0]
    A = BA[:, length_X0+1:end]

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    gx, hx, alarm_LinearSolution, nk = SolveDiffEq(A, B, sr.n_par, estim)

    println("State Space Solution Done")
    println(" ")

    return gx, hx, alarm_LinearSolution, nk, A, B
end
