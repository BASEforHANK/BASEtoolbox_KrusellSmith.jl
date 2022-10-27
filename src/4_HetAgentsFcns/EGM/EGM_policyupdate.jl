@doc raw"""
    EGM_policyupdate(EVm,EVk,Qminus,πminus,RBminus,Tshock,inc,n_par,m_par,warnme)

Find optimal policies, given marginal continuation values `EVm`, `EVk`, today's
prices [`Qminus`, `πminus`,`RBminus`], and income [`inc`], using the
Endogenous Grid Method.

Optimal policies are defined on the fixed grid, but optimal asset choices (`m` and `k`)
are off-grid values.

# Returns
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal (on-grid) policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
"""
function EGM_policyupdate(EVm::Array,
                          r_minus::Real,
                          inc::Array,
                          n_par::NumericalParameters,
                          m_par::ModelParameters, 
                          warnme::Bool)

################### Copy/read-out stuff#####################################
β::Float64  = m_par.β

# inc[1] = labor income , 
# inc[2]= liquid assets income
inc_lab     = inc[1]
inc_LA      = inc[2]
n           = size(EVm)
mmax        = n_par.grid_m[end]

############################################################################
## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
############################################################################
EMU         = EVm .* β
c_star_n    = invmutil(EMU) # 6% of time with rolled out power function

# Calculate assets consistent with choices being [m']
# Calculate initial money position from the budget constraint
# that leads to the optimal consumption choice
m_star_n    = (c_star_n .+ n_par.mesh_m .- inc_lab)
# Apply correct interest rate
m_star_n   .= m_star_n ./ (r_minus)  # apply borrowing rate

# Next step: Interpolate w_guess and c_guess from new k-grids
# using c[s,h,m"], m(s,h,m")
# Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

# Check monotonicity of m_star_n
if warnme
    m_star_aux    = reshape(m_star_n,(n[1], n[2]))
    if any(any(diff(m_star_aux, dims=1).<0))
        @warn "non monotone future liquid asset choice encountered"
    end
end

# Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
# fixed grid.
c_n_star    = Array{eltype(c_star_n),2}(undef,(n[1],n[2]))#zeros(eltype(c_star_n), size(c_star_n)) # Initialize c_n-container
m_n_star    = Array{eltype(c_star_n),2}(undef,(n[1],n[2]))#zeros(eltype(c_star_n), size(c_star_n)) # Initialize m_n-container

@inbounds @views  begin
    for jj=1:n[2] # Loop over income states
        cc, mn = mylinearinterpolate_mult2(m_star_n[:,jj], c_star_n[:,jj],n_par.grid_m, n_par.grid_m)
        c_n_star[:,jj]=cc
        m_n_star[:,jj]=mn
        # Check for binding borrowing constraints, no extrapolation from grid
        bcpol = m_star_n[1,jj]
        for mm= 1:n[1]
            if n_par.mesh_m[mm,jj] .<bcpol
                c_n_star[mm,jj] = inc_lab[mm,jj] .+ inc_LA[mm,jj] .- n_par.grid_m[1]
                m_n_star[mm,jj] = n_par.grid_m[1]
            end
            if mmax  .< m_n_star[mm,jj]
                m_n_star[mm,jj] = mmax
            end
        end
    end
end



return c_n_star, m_n_star
end

