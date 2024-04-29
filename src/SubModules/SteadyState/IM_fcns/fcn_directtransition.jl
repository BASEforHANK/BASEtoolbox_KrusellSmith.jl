@doc raw"""
    DirectTransition_Splines(dPrime::Array, m_star::Array, cdf_today::Array, Π::Array, n_par::NumericalParameters)

Iterates the CDF one period forward.

# Arguments
- `m_star::Array`: optimal savings function
- `Π::Array`: transition matrix
- `cdf_today::Array`: # CDF over individual income states 
- `n_par::NumericalParameters`

# Returns
- `dPrime::Array`: distribution in period t+1
"""
function DirectTransition_Splines(
    m_star::Array,
    cdf_today::Array,
    Π::Array,
    n_par::NumericalParameters,
)

    cdf_prime = zeros(eltype(cdf_today), size(cdf_today))
    DirectTransition_Splines!(cdf_prime, m_star, cdf_today, Π, n_par)

    return cdf_prime
end


function DirectTransition_Cond_Splines!(
    cdf_prime_given_y::AbstractArray,
    m_prime_given_y::AbstractArray,
    n_par::NumericalParameters,
)
    # 2a. Specify mapping from assets to cdf with monotonic PCIHP interpolation

    # Find values where the constraint binds:
    # We can only interpolate using the last value at the constraint, because 
    # m_prime is not unique otherwise.
    idx_last_at_constraint = findlast(m_prime_given_y .== n_par.grid_m[1])

    # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
    m_at_max_cdf = m_prime_given_y[end]
    idx_last_increasing_cdf = findlast(diff(cdf_prime_given_y) .> eps())
    if idx_last_increasing_cdf !== nothing
        m_at_max_cdf = m_prime_given_y[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
    end

    # Start interpolation from last unique value (= last value at the constraint)
    if isnothing(idx_last_at_constraint)
        m_to_cdf_spline = Interpolator(m_prime_given_y[1:end], cdf_prime_given_y[1:end])
    else
        m_to_cdf_spline = Interpolator(
            m_prime_given_y[idx_last_at_constraint:end],
            cdf_prime_given_y[idx_last_at_constraint:end],
        )
    end

    # Extrapolation for values below and above observed m_primes and interpolation as defined above otherwise
    function m_to_cdf_spline_extr!(cdf_values::AbstractVector, m::Vector{Float64})
        idx1 = findlast(m .< m_prime_given_y[1])
        if idx1 != nothing
            cdf_values[1:idx1] .= 0.0
        else
            idx1 = 0
        end
        idx2 = findfirst(m .> min(m_at_max_cdf, m_prime_given_y[end]))
        if idx2 != nothing
            cdf_values[idx2:end] .= 1.0 * cdf_prime_given_y[end]
        else
            idx2 = length(m) + 1
        end
        cdf_values[idx1+1:idx2-1] .= m_to_cdf_spline.(m[idx1+1:idx2-1])
    end


    # 2b. Evaluate cdf at fixed grid
    cdfend = copy(cdf_prime_given_y[end])
    m_to_cdf_spline_extr!(cdf_prime_given_y, n_par.grid_m)
    cdf_prime_given_y .= min.(cdf_prime_given_y, cdfend)
    cdf_prime_given_y[end] = cdfend
end

@doc raw"""
    DirectTransition_Splines!(
        m_prime_grid::AbstractArray,
        cdf_initial::AbstractArray,
        n_par::NumericalParameters,

    )

    Direct transition of the savings cdf from one period to the next. 
        Transition is done using monotonic spline interpolation to bring the next periods cdf's 
        back to the reference grid.
        Logic: Given assets in t (on-grid) and an the income shock realization, the decision
        of next periods assets is deterministic and thus the probability mass move from the 
        on grid to the off-grid values. Using the spline interpolation the cdf is evaluated at
        the fix asset grid.

    # Arguments
    - `cdf_prime_on_grid::AbstractArray`: Next periods CDF on fixed asset grid.
    - `m_prime_grid::AbstractArray`: Savings function defined on the fixed asset and income grid.
    - `cdf_initial::AbstractArray`: CDF over fixed assets grid for each income realization.
    - `Π::Array`: Stochastic transition matrix.
    - `n_par::NumericalParameters`
    - `m_par::ModelParameters`
"""
function DirectTransition_Splines!(
    cdf_prime_on_grid::AbstractArray,
    m_prime_grid::AbstractArray,
    cdf_initial::AbstractArray,
    Π::Array,
    n_par::NumericalParameters,
)
    # 1. Map cdf back to fixed asset grid.
    @inbounds for i_y = 1:n_par.ny
        m_prime_given_y = view(m_prime_grid, :, i_y) # Cap at maximum gridpoint
        cdf_prime_given_y = cdf_initial[:, i_y]
        DirectTransition_Cond_Splines!(cdf_prime_given_y, m_prime_given_y, n_par)
        cdf_prime_on_grid[:, i_y] = cdf_prime_given_y
    end

    # 2. Build expectation of cdf over income states
    cdf_prime_on_grid .= cdf_prime_on_grid * Π
end
