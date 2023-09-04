@doc raw"""
    DirectTransition(m_star::Array, distr::Array, Π::Array, n_par::NumericalParameters)

Iterates the distribution one period forward.

# Arguments
- `m_star::Array`: optimal savings function
- `distr::Array`: distribution in period t
- `Π::Array`: transition matrix

# Output
- `dPrime::Array`: distribution in period t+1

"""
function DirectTransition(m_star::Array, distr::Array, Π::Array, n_par::NumericalParameters)

    dPrime = zeros(eltype(distr), size(distr))
    DirectTransition!(dPrime, m_star, distr, Π, n_par)
    return dPrime
end

@doc raw"""
    DirectTransition!(dPrime::Array, m_star::Array, distr::Array, Π::Array, n_par::NumericalParameters)

Iterates the distribution one period forward.

# Arguments
- `dPrime::Array`: distribution in period t+1
- `m_star::Array`: optimal savings function
- `distr::Array`: distribution in period t
- `Π::Array`: transition matrix

"""
function DirectTransition!(
    dPrime::Array,
    m_star::Array,
    distr::Array,
    Π::Array,
    n_par::NumericalParameters,
)

    idm_n, wR_m_n = MakeWeightsLight(m_star, n_par.grid_m)
    blockindex = (0:n_par.ny-1) * n_par.nm
    @views @inbounds begin
        for zz = 1:n_par.ny # all current income states
            for mm = 1:n_par.nm
                dd = distr[mm, zz]
                IDD_n = idm_n[mm, zz]
                DL_n = (dd .* (1.0 .- wR_m_n[mm, zz]))
                DR_n = (dd .* wR_m_n[mm, zz])
                pp = (Π[zz, :])
                for yy = 1:n_par.ny
                    id_n = IDD_n .+ blockindex[yy]
                    dPrime[id_n] += pp[yy] .* DL_n
                    dPrime[id_n+1] += pp[yy] .* DR_n
                end
            end
        end
    end
end
