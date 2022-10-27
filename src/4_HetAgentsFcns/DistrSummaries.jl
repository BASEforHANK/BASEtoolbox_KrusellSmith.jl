
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
function distrSummaries(distr::AbstractArray,
                        c_n_star::AbstractArray, n_par::NumericalParameters,
                        inc::AbstractArray, m_par::ModelParameters)
    ## Distributional summaries
    distr_m = sum(distr,dims=2)[:]
    distr_y = sum(distr,dims=1)[:]

    money_pdf= distr_m
    money_cdf= cumsum(money_pdf);
    S               = [0; cumsum(money_pdf.*n_par.grid_m)]
    giniwealth      = 1-(sum(money_pdf.*(S[1:end-1]+S[2:end]))/S[end]);

    FN_wealthshares = cumsum(n_par.grid_m.*money_pdf)./sum(n_par.grid_m.*money_pdf);

    TOP10Wshare        = 1.0 - mylinearinterpolate(money_cdf,FN_wealthshares,[0.9])[1]


    distr_x         = zeros(eltype(c_n_star),(n_par.nm, n_par.ny))
    x               = c_n_star;
    aux_x           = inc[3]
    aux_x[:,end]    = zeros(n_par.nm);
    c               = x +aux_x;
    distr_x         =  distr

    IX              = sortperm(x[:]);
    x               = x[IX];
    x_pdf           = distr_x[IX];
    S               = cumsum(x_pdf.*x);
    S               = [0 S'];
    ginicompconsumption = 1-(sum(x_pdf.*(S[1:end-1]+S[2:end]))/S[end]);

    IX              = sortperm(c[:]);
    c               = c[IX];
    c_pdf           = distr_x[IX];
    S               = cumsum(c_pdf.*c);
    c_cdf           = cumsum(c_pdf);

    S               = [0 S'];
    giniconsumption = 1-(sum(c_pdf.*(S[1:end-1]+S[2:end]))/S[end]);


    Yidio           = inc[4]+inc[2] - n_par.mesh_m
    IX              = sortperm(Yidio[:])
    Yidio           = Yidio[IX]
    Y_pdf           = distr[IX]
    Y_cdf           = cumsum(Y_pdf)
    FN_incomeshares = cumsum(Yidio.*Y_pdf)./sum(Yidio.*Y_pdf);
    TOP10Ishare        = 1.0 .- mylinearinterpolate(Y_cdf,FN_incomeshares,[0.9])[1]

    S               = cumsum(Y_pdf.*Yidio)
    S               = [0 S']
    giniincome      = 1-(sum(Y_pdf.*(S[1:end-1]+S[2:end]))/S[end])

    Yidio           = inc[4]
    Yidio           = Yidio[:,1:end-1]
    IX              = sortperm(Yidio[:])
    Yidio           = Yidio[IX]
    distr_aux       = distr[:,1:end-1]
    distr_aux       = distr_aux./sum(distr_aux[:])
    Y_pdf           = distr_aux[IX]
    Y_cdf           = cumsum(Y_pdf)
   
    sdlogy          = sqrt(Y_pdf[:]'*log.(Yidio[:]).^2-(Y_pdf[:]'*log.(Yidio[:]))^2);



    return     distr_m, distr_y,  TOP10Wshare, TOP10Ishare, giniwealth, ginicompconsumption,#=
            =# giniconsumption, giniincome, sdlogy
end
