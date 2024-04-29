#----------------------------------------------------------------------------
# Basic Functions: Utility, marginal utility and its inverse, 
#                  Return on capital, Wages, Employment, Output
#---------------------------------------------------------------------------

function util(c::AbstractArray, m_par)
    if m_par.ξ == 1.0
        util = log.(c)
    elseif m_par.ξ == 2.0
        util = 1.0 - 1.0 ./ c
    elseif m_par.ξ == 4.0
        util = (1.0 - 1.0 ./ (c .* c .* c)) ./ 3.0
    else
        util = (c .^ (1.0 .- m_par.ξ) .- 1.0) ./ (1.0 .- m_par.ξ)
    end
    return util
end

function mutil(c::AbstractArray, m_par)
    if m_par.ξ == 1.0
        mutil = 1.0 ./ c
    elseif m_par.ξ == 2.0
        mutil = 1.0 ./ (c .^ 2)
    elseif m_par.ξ == 4.0
        mutil = 1.0 ./ ((c .^ 2) .^ 2)
    else
        mutil = c .^ (-m_par.ξ)
    end
    return mutil
end

function mutil!(mu::AbstractArray, c::AbstractArray, m_par)
    if m_par.ξ == 1.0
        mu .= 1.0 ./ c
    elseif m_par.ξ == 2.0
        mu .= 1.0 ./ (c .^ 2)
    elseif m_par.ξ == 4.0
        mu .= 1.0 ./ ((c .^ 2) .^ 2)
    else
        mu .= c .^ (-m_par.ξ)
    end
    return mu
end

function invmutil(mu, m_par)
    if m_par.ξ == 1.0
        c = 1.0 ./ mu
    elseif m_par.ξ == 2.0
        c = 1.0 ./ (sqrt.(mu))
    elseif m_par.ξ == 4.0
        c = 1.0 ./ (sqrt.(sqrt.(mu)))
    else
        c = 1.0 ./ mu .^ (1.0 ./ m_par.ξ)
    end
    return c
end

function invmutil!(c, mu, m_par)
    if m_par.ξ == 1.0
        c .= 1.0 ./ mu
    elseif m_par.ξ == 2.0
        c .= 1.0 ./ (sqrt.(mu))
    elseif m_par.ξ == 4.0
        c .= 1.0 ./ (sqrt.(sqrt.(mu)))
    else
        c .= 1.0 ./ mu .^ (1.0 ./ m_par.ξ)
    end
    return c
end

# Incomes (K:capital, Z: TFP): Interest rate = MPK.-δ, Wage = MPL, profits = Y-wL-(r+\delta)*K
interest(K::Number, Z::Number, N::Number, m_par; delta = m_par.δ_0) =
    Z .* m_par.α .* (K ./ N) .^ (m_par.α - 1.0) .- delta
wage(K::Number, Z::Number, N::Number, m_par) = Z .* (1 - m_par.α) .* (K ./ N) .^ m_par.α

employment(K::Number, Z::Number, m_par) =
    (Z .* (1.0 - m_par.α) .* K .^ (m_par.α)) .^ (1.0 ./ (m_par.γ + m_par.α))

output(K::Number, Z::Number, N::Number, m_par) = Z .* K .^ (m_par.α) .* N .^ (1 - m_par.α)



""" integrate_capital(cdf_m::AbstractVector, n_par::NumericalParameters)

Get aggregate capital through integration over the savings distribution. The savings CDF
is interpolated by splines. The integration utilizes integration by parts to use the 
smoother cdf and get K = int(k*f(k)) = k*F(k) - int(F(k)).

Arguments:
- `cdf_m::AbstractVector`: CDF over savings
- `n_par::NumericalParameters`

Returns:
- `E_K`: Aggregate Capital

"""
function integrate_capital(cdf_m::AbstractVector, n_par)

    m_to_cdf_splines = Interpolator(n_par.grid_m, cdf_m)

    right_part = integrate(m_to_cdf_splines, n_par.grid_m[1], n_par.grid_m[end])
    left_part =
        n_par.grid_m[end] * m_to_cdf_splines(n_par.grid_m[end]) -
        n_par.grid_m[1] * m_to_cdf_splines(n_par.grid_m[1])
    E_K = left_part - right_part
    return E_K
end
