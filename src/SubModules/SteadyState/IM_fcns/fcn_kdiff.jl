@doc raw"""
    Kdiff(K_guess,n_par,m_par)

Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

Requires global functions `employment(K,A,m_par)`, `interest(K,A,N,m_par)`,
`wage(K,A,N,m_par)`, `output(K,TFP,N,m_par)`, and [`Ksupply()`](@ref).

# Arguments
- `K_guess::Float64`: capital stock guess
- `n_par::NumericalParameters`, `m_par::ModelParameters`
"""
function Kdiff(
    K_guess::Float64,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    initial::Bool = true,
    Vm_guess::AbstractArray = zeros(1, 1),
    CDF_guess::AbstractArray = zeros(1, 1),
)
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
    # #----------------------------------------------------------------------------
    # # Array (inc) to store incomes
    # # inc[1] = labor income , inc[2] = rental income,
    # # inc[3]= liquid assets income, inc[4] = capital liquidation income
    # #----------------------------------------------------------------------------
    # N = employment(K_guess, 1.0, m_par)
    # constant N
    N = m_par.N
    r = interest(K_guess, 1.0, N, m_par) + 1.0
    w = wage(K_guess, 1.0, N, m_par)
    Y = output(K_guess, 1.0, N, m_par)

    inc = incomes(n_par, m_par, r, w, N)

    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

    # initial guess consumption and marginal values (if not set)
    if initial
        c_guess = inc[1] .+ inc[2]
        if any(any(c_guess .< 0.0))
            @warn "negative consumption guess"
        end
        Vm = r .* mutil(c_guess, m_par)
        cdf = n_par.CDF_guess
    else
        Vm = Vm_guess
        cdf = CDF_guess
    end
    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    KS = Ksupply(r, n_par, m_par, Vm, cdf, inc)
    K = KS[1] # capital
    println("K = $K_guess, KS = $K")
    Vm = KS[end-1] # marginal value of liquid assets
    cdf = KS[end] # stationary distribution  
    diff = K - K_guess # excess supply of funds
    return diff, Vm, cdf
end
