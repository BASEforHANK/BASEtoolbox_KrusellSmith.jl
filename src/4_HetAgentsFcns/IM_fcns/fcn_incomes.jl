@doc raw"""
    incomes!(inc::Array, n_par::NumericalParameters, m_par::ModelParameters, r::Float64, w::Float64, N::Float64)

Calculates the incomes households obtain in the current period.

# Arguments
- `inc::Array`: array to store the incomes
- `n_par::NumericalParameters`: numerical parameters
- `m_par::ModelParameters`: model parameters
- `r::Float64`: interest rate
- `w::Float64`: wage
- `N::Float64`: aggregate labor supply

# Output
- `inc::Array`: array with the incomes

"""
function incomes(n_par, m_par, r, w, N)

    # consider ! version that does not allocate inc new
    inc = fill(Array{typeof(r)}(undef, size(n_par.mesh_m)), 4)
    incomes!(inc, n_par, m_par, r, w, N)

    return inc
end

@doc raw"""
    incomes!(inc::Array, n_par::NumericalParameters, m_par::ModelParameters, r::Float64, w::Float64, N::Float64)

Calculates the incomes households obtain in the current period.

# Arguments
- `inc::Array`: array to store the incomes
- `n_par::NumericalParameters`: numerical parameters
- `m_par::ModelParameters`: model parameters
- `r::Float64`: interest rate
- `w::Float64`: wage
- `N::Float64`: aggregate labor supply

"""
function incomes!(inc, n_par, m_par, r, w, N)

    labor_income = ((n_par.mesh_y / n_par.H) .* w .* N)
    GHHFA = ((m_par.γ) / (m_par.γ + 1)) # transformation (scaling) for composite good

    inc .= [
        GHHFA .* labor_income, # incomes of workers adjusted for disutility of labor
        r .* n_par.mesh_m, # liquid asset Income
        (1.0 / (m_par.γ + 1)) .* labor_income, # Adjustment factor for composite
        labor_income,
    ]

end
