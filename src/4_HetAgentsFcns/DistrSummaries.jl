
@doc raw"""
    distrSummaries(distr,c_a_star,c_n_star,n_par,inc,incgross,m_par)

Compute distributional summary statistics, e.g. Gini indexes, top-10%
income and wealth shares, and 10%, 50%, and 90%-consumption quantiles.

# Arguments
- `distr`: joint distribution over bonds, capital and income ``(m \times k \times y)``
- `c_a_star`,`c_n_star`: optimal consumption policies with [`a`] or without [`n`]
    capital adjustment
- `n_par::NumericalParameters`, `m_par::ModelParameters`
- `inc`: vector of (on grid-)incomes, consisting of labor income (scaled by ``\frac{\gamma-\tau^P}{1+\gamma}``, plus labor union-profits),
    rental income, liquid asset income, capital liquidation income,
    labor income (scaled by ``\frac{1-\tau^P}{1+\gamma}``, without labor union-profits),
    and labor income (without scaling or labor union-profits)
- `incgross`: vector of (on grid-) *pre-tax* incomes, consisting of
    labor income (without scaling, plus labor union-profits), rental income,
    liquid asset income, capital liquidation income,
    labor income (without scaling or labor union-profits)
"""
function distrSummaries(
    distr::AbstractArray,
    c_n_star::AbstractArray,
    n_par::NumericalParameters,
    inc::AbstractArray,
    m_par::ModelParameters,
)
    ## Distributional summaries
    distr_m = sum(distr, dims = 2)[:]
    distr_y = sum(distr, dims = 1)[:]

    wealth = n_par.grid_m
    wealth_pdf = copy(distr_m)
    wealth_cdf = cumsum(wealth_pdf)
    wealth_w = wealth_pdf .* wealth
    wealthshares = cumsum(wealth_w) ./ sum(wealth_w)

    TOP10Wshare = 1.0 - mylinearinterpolate(wealth_cdf, wealthshares, [0.9])[1]
    giniwealth = gini(wealth, wealth_pdf)

    # Consumption distribution
    x = c_n_star
    aux_x = inc[3]
    c = x + aux_x
    distr_c = copy(distr)

    # Gini of goods consumption
    IX = sortperm(c[:])
    c[:] .= c[IX]
    distr_c[:] .= distr_c[IX]
    giniconsumption = gini(c, distr_c)

    # Top 10 gross income share
    Yidio = inc[4] + inc[2] - n_par.mesh_m
    IX = sortperm(Yidio[:])
    Yidio = Yidio[IX]
    Y_pdf = distr[IX]
    Y_cdf = cumsum(Y_pdf)
    Y_w = Y_pdf .* Yidio
    incomeshares = cumsum(Y_w) ./ sum(Y_w)
    TOP10Ishare = 1.0 .- mylinearinterpolate(Y_cdf, incomeshares, [0.9])[1]

    # Standard deviation of log labor earnings
    Yidio = inc[4]
    IX = sortperm(Yidio[:])
    Yidio = Yidio[IX]
    Y_pdf = distr[IX]

    sdlogy = sqrt(dot(Y_pdf, Yidio .^ 2) .- dot(Y_pdf, Yidio) .^ 2)


    return distr_m, distr_y, TOP10Wshare, TOP10Ishare, giniwealth, giniconsumption, sdlogy
end

function gini(x, pdf)
    s = 0.0
    gini = 0.0
    for i in eachindex(x)
        gini -= pdf[i] * s
        s += x[i] * pdf[i]
        gini -= pdf[i] * s
    end
    gini /= s
    gini += 1.0
    return gini
end
