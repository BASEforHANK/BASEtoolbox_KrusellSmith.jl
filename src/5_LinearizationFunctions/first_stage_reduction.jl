
@doc raw"""
first_stage_reduction(VkSS, VmSS, TransitionMat_aSS, TransitionMat_nSS, NSS, 
    m_a_starSS, k_a_starSS, m_n_starSS, price, n_par, m_par)

Selects the DCT coefficients to be perturbed (first stage reduction, see Appendix C).

The selection is based both on the shape of the steady-state marginal values 
(in terms of log-inverse marginal utilities), and of the approximate factor representation
based on equation (C.8).

`price' is the vector of prices that appear in the household problem. 
Prices with known identical effects to other prices are left out.
`VkSS` and `VmSS` are the steady state marginal value functions, 
`TransitionMat_aSS` and `TransitionMat_nSS` are the transition matrix induced by 
the steady state policies conditional on adjustment (_a) and non-adjustment (_n).
`NSS` is the level of employment in steady state.
`m_a_starSS', `k_a_starSS', and `m_n_starSS` are the liquid and illiquid savings 
policies in steady state conditional on adjustment (_a) or non-adjustment (-n).
`n_par' and `m_par' the numerical and model parameters. 
WARNING: the order of prices is hard-coded in [`VFI()`](@ref)) 
"""
function first_stage_reduction(
    VmSS::Array,
    TransitionMat_SS::SparseMatrixCSC,
    NSS::Float64,
    m_n_starSS::Array,
    price::Vector,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    #-------------------------------------------------------------------------------------
    # Calculate the Jacobian of the marginal value functions w.r.t. contemporaneous prices
    #-------------------------------------------------------------------------------------
    phi = 0.999 # approximate autocorrelation of prices

    J = ForwardDiff.jacobian(p -> VFI(p, VmSS, NSS, n_par, m_par), log.(price))
    Numel = n_par.nm * n_par.ny # Total grid size

    # Derivatives of policy functions as central finite differences
    Dm_nm = spdiagm(
        m_par.β .*
        centralderiv(reshape(m_n_starSS, (n_par.nm, n_par.ny)), n_par.mesh_m, 1)[:],
    )

    # Joint transition matrix taking policy function marginals into account
    GammaTilde = Matrix(Dm_nm * TransitionMat_SS)

    #-----------------------------------------------------------------------------------
    # Cummulate sum of derivatives
    #-----------------------------------------------------------------------------------
    W = (LinearAlgebra.I - phi * GammaTilde) \ hcat(J[1], J[2])
    Wm = W[1:Numel, :]         # Jacobian of the marginal value of liquid assets

    # Take into account that perturbed value is written in log-inverse mutils
    TransformV(V) = log.(invmutil(V, m_par))
    Outerderivative(x) = ForwardDiff.derivative(V -> TransformV(V), x)

    CBarHat_m = realpart.(Outerderivative.(VmSS[:]) .* Wm)  # Chain rule

    #---------------------------------------------------------------------
    # Find the indexes that allow the DCTs to fit CBarHat, Vm, and Vk well 
    #---------------------------------------------------------------------
    indm = Array{Array{Int}}(undef, 2)

    # Calculate average absolute derivative (in DCT terms)
    Theta_m = similar(CBarHat_m)
    for j in eachindex(price)
        Theta_m[:, j] = dct(reshape(CBarHat_m[:, j], (n_par.nm, n_par.ny)))[:]
    end
    theta_m = (sum(abs.(Theta_m); dims = 2))

    # Find those DCT indexes that explain the average derivative well
    indm[1] = select_ind(reshape(theta_m, (n_par.nm, n_par.ny)), n_par.reduc_marginal_value)

    # Add the indexes that fit the shape of the marginal value functions themselves well
    indm[end] = select_ind(
        dct(reshape(log.(invmutil(VmSS, m_par)), (n_par.nm, n_par.ny))),
        n_par.reduc_value,
    )

    return indm, J
end

function VFI(price::Vector, VmSS::Array, NSS::Float64, n_par, m_par)
    # Update contemporaneous value functions for given contemporaneous prices
    # and steady-state continuation marginal values for illiquid and liquid assets

    # Prices in logs (elasticities)
    price = exp.(price)

    # Read out individual "prices" (WARNING: hard coded order)
    r = price[1]
    w = price[2]

    # Fixed inputs / prices that have the same impact on decisions/Value 
    # Functions as some of the above
    N = NSS     # shows up only multiplicatively with w

    # Human capital transition
    Π = n_par.Π .+ zeros(eltype(price), 1)[1]
    #PP = ExTransition(m_par.ρ_h, n_par.bounds_y, sqrt(σ))
    #Π[1:(end-1), 1:(end-1)] = PP .* (1.0 - m_par.ζ)

    # Incomes given prices
    inc = incomes(n_par, m_par, r, w, N)

    # Expected marginal values
    EVmPrime = reshape(VmSS, (n_par.nm, n_par.ny)) .+ zeros(eltype(price), 1)[1]
    EVmPrime .= EVmPrime * Π'


    # Calculate optimal policies
    c_n_star, m_n_star = EGM_policyupdate(EVmPrime, r, inc, n_par, m_par, false) # policy iteration

    # Update marginal values
    Vm_up = mutil(c_n_star, m_par)

    Vm_up .*= r

    return [Vm_up[:]]
end
