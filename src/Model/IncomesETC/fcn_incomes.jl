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
function incomes(n_par, m_par, r, w, N; D = 1.0)

    # consider ! version that does not allocate inc new
    inc = fill(Array{typeof(r)}(undef, size(n_par.mesh_m)), 4)
    incomes!(inc, n_par, m_par, r, w, N; D = D)

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
function incomes!(inc, n_par, m_par, r, w, N; D = 1.0)

    labor_income = n_par.mesh_y .* w .* N
    GHHFA = 1.0

    inc .= [
        GHHFA .* labor_income, # incomes of workers adjusted for disutility of labor
        D .* r .* n_par.mesh_m, # liquid asset Income
        labor_income,
        labor_income,
    ]

end
