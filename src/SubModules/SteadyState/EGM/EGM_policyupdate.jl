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
function EGM_policyupdate(
    EVm::Array,
    r_minus::Real,
    inc::Array,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    warnme::Bool,
)
    # Pre-Allocate returns
    c_n_star = similar(EVm) # Initialize c_n-container
    m_n_star = similar(EVm) # Initialize m_n-container
    # containers for auxiliary variables
    EMU = similar(EVm)
    c_star_n = similar(EVm)
    m_star_n = similar(EVm)
    EGM_policyupdate!(
        c_n_star,
        m_n_star,
        EMU,
        c_star_n,
        m_star_n,
        EVm,
        r_minus,
        inc,
        n_par,
        m_par,
        warnme,
    )

    return c_n_star, m_n_star
end

function EGM_policyupdate!(
    c_n_star::Array,
    m_n_star::Array,
    EMU::Array,
    c_star_n::Array,
    m_star_n::Array,
    EVm::Array,
    r_minus::Real,
    inc::Array,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    warnme::Bool,
)

    ################### Copy/read-out stuff#####################################
    β::Float64 = m_par.β

    # inc[1] = labor income , 
    # inc[2]= liquid assets income
    inc_lab = inc[1]
    inc_LA = inc[2]
    n = size(EVm)

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU .= EVm .* β
    invmutil!(c_star_n, EMU, m_par) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [m']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    m_star_n .= (c_star_n .+ n_par.mesh_m .- inc_lab)
    # Apply correct interest rate
    m_star_n .= m_star_n ./ (r_minus)  # apply borrowing rate

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

    # Check monotonicity of m_star_n
    if warnme
        m_star_aux = reshape(m_star_n, (n[1], n[2]))
        if any(any(diff(m_star_aux, dims = 1) .< 0))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid.
    interpolate_policies_to_fixed_grid_splines!(
        c_n_star,
        m_n_star,
        c_star_n,
        m_star_n,
        inc_lab,
        inc_LA,
        n_par,
        n;
        warnme = true,
    )


end


"""
    interpolate_policies_to_fixed_grid!(c_n_star, m_n_star, c_star_n, m_star_n, n_par, n)

Interpolates the policies to a fixed grid using linear interpolation.

# Arguments
- `c_n_star`: The consumption policy function on the endogenous grid.
- `m_n_star`: The cash-on-hand policy function on the endogenous grid.
- `c_star_n`: The consumption policy function on the exogenous grid.
- `inc_lab`: Labor income.
- `inc_LA`: Liquid asset income.	
- `m_star_n`: The cash-on-hand policy function on the exogenous grid.
- `n_par`: Numerical arameters.
- `n`: The number of grid points in both dimension.

# Returns
Updates the policies in-place to be on the fixed grid.
"""
function interpolate_policies_to_fixed_grid!(
    c_n_star,
    m_n_star,
    c_star_n,
    m_star_n,
    inc_lab,
    inc_LA,
    n_par,
    n;
    extrapolate = false,
)

    @inbounds @views begin
        for jj = 1:n[2] # Loop over income states
            mylinearinterpolate_mult2!(
                c_n_star[:, jj],
                m_n_star[:, jj],
                m_star_n[:, jj],
                c_star_n[:, jj],
                n_par.grid_m,
                n_par.grid_m,
            )
            # Check for binding borrowing constraints, no extrapolation from grid
            bcpol = m_star_n[1, jj]
            for mm = 1:n[1]
                if n_par.grid_m[mm] .< bcpol
                    c_n_star[mm, jj] = inc_lab[mm, jj] .+ inc_LA[mm, jj] .- n_par.grid_m[1]
                    m_n_star[mm, jj] = n_par.grid_m[1]
                end
                if !extrapolate
                    if n_par.grid_m[end] .< m_n_star[mm, jj]
                        m_n_star[mm, jj] = n_par.grid_m[end]
                    end
                end
            end
        end
    end
end

"""
interpolate_policies_to_fixed_grid_splines!(c_n_star, m_n_star, c_star_n, m_star_n, n_par, n, warnme)

Interpolates the policies to a fixed grid using monotonic PCHIP interpolation.

# Arguments
- `c_n_star`: The consumption policy function on the endogenous grid.
- `m_n_star`: The cash-on-hand policy function on the endogenous grid.
- `c_star_n`: The consumption policy function on the exogenous grid.
- `m_star_n`: The cash-on-hand policy function on the exogenous grid.
- `inc_lab`: Labor income.
- `inc_LA`: Liquid asset income.	
- `n_par`: Numerical arameters.
- `n`: The number of grid points in both dimension.
- `warnme`: Whether to warn if the policies are not monotone.	

# Returns
Updates the policies in-place to be on the fixed grid.
"""
function interpolate_policies_to_fixed_grid_splines!(
    c_n_star,
    m_n_star,
    c_star_n,
    m_star_n,
    inc_lab,
    inc_LA,
    n_par,
    n;
    warnme = true,
    extrapolate = false,
)

    @inbounds @views begin
        for jj = 1:n[2] # Loop over income states

            bcpol = m_star_n[1, jj]
            max_b = m_star_n[end, jj]

            # a. Specify mapping from m* to fixed grid with monotonic PCIHP interpolation
            if extrapolate && max_b < n_par.grid_m[end]
                # extrapolate by extending the grid and the policy function based on a linear slope
                m_dist_extra = 10.0
                m_grid_wide = [n_par.grid_m; n_par.grid_m[end] .+ m_dist_extra]
                extrapol_slope =
                    (n_par.grid_m[end] - n_par.grid_m[end-1]) /
                    (max_b - m_star_n[end-1, jj])
                extrapol_distance = m_grid_wide[end] - max_b
                m_star_n_wide =
                    [m_star_n[:, jj]; max_b + extrapol_slope * extrapol_distance]
            else
                # no extrapolation
                m_dist_extra = 0.0
                m_grid_wide = n_par.grid_m
                m_star_n_wide = m_star_n[:, jj]
            end

            max_b += m_dist_extra
            m_to_mprime_spline = Interpolator(m_star_n_wide, m_grid_wide)

            # Cut off at borrowing constraint and maximum gridpoint
            function m_to_mprime_spline_extr!(mprime::AbstractArray, m::Vector{Float64})

                idx_below_bc = findlast(m .< bcpol)
                if idx_below_bc != nothing
                    mprime[1:idx_below_bc] .= n_par.grid_m[1]
                else
                    idx_below_bc = 0
                end

                idx_after_max = findfirst(m .> max_b)
                if idx_after_max != nothing
                    if extrapolate
                        @warn "evaluation beyond specified extrapolation range"
                    end
                    mprime[idx_after_max:end] .= m_grid_wide[end]
                else
                    idx_after_max = length(m) + 1
                end

                mprime[idx_below_bc+1:idx_after_max-1] .=
                    m_to_mprime_spline.(m[idx_below_bc+1:idx_after_max-1])
            end

            # b. Evaluate cdf at fixed grid
            m_to_mprime_spline_extr!(m_n_star[:, jj], n_par.grid_m)
            c_n_star[:, jj] = inc_lab[:, jj] .+ inc_LA[:, jj] .- m_n_star[:, jj]

        end

    end

    # Check monotonicity of m_n_star and c_n_star
    if warnme
        m_monoton = any(any(diff(m_n_star, dims = 1) .< 0))
        c_monoton = any(any(diff(c_n_star, dims = 1) .< 0))
        if m_monoton
            @warn "non monotone future liquid asset choice encountered"
        end
        if c_monoton
            @warn "non monotone consumption choice encountered"
        end
    end

end
