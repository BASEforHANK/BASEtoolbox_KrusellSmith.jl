function incomes(n_par, m_par, r ,w ,N)

    
    labor_income = ((n_par.mesh_y/n_par.H).*w.*N)
    GHHFA = ((m_par.γ)/(m_par.γ+1)) # transformation (scaling) for composite good

    inc   = [
                GHHFA.*labor_income, # incomes of workers adjusted for disutility of labor
                r .* n_par.mesh_m, # liquid asset Income
                (1.0/(m_par.γ+1)).*labor_income,
                labor_income#
    ]


return inc
end