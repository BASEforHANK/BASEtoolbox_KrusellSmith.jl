#----------------------------------------------------------------------------
# Basic Functions: Utility, marginal utility and its inverse, 
#                  Return on capital, Wages, Employment, Output
#---------------------------------------------------------------------------

function util(c::AbstractArray, m_par::ModelParameters)
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

function mutil(c::AbstractArray, m_par::ModelParameters)
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

function mutil!(mu::AbstractArray, c::AbstractArray, m_par::ModelParameters)
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

function invmutil(mu, m_par::ModelParameters)
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

function invmutil!(c, mu, m_par::ModelParameters)
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
interest(K::Number, Z::Number, N::Number, m_par::ModelParameters) =
    Z .* m_par.α .* (K ./ N) .^ (m_par.α - 1.0) .- m_par.δ_0
wage(K::Number, Z::Number, N::Number, m_par::ModelParameters) =
    Z .* (1 - m_par.α) .* (K ./ N) .^ m_par.α

employment(K::Number, Z::Number, m_par::ModelParameters) =
    (Z .* (1.0 - m_par.α) .* K .^ (m_par.α)) .^ (1.0 ./ (m_par.γ + m_par.α))

output(K::Number, Z::Number, N::Number, m_par::ModelParameters) =
    Z .* K .^ (m_par.α) .* N .^ (1 - m_par.α)
