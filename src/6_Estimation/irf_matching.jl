function irfmatch(par, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)

    return irfmatch_backend(par, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)
end

function irfmatch_backend(par, IRFtargets, weights, SHOCKs, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)

    # check priors, abort if they are violated
    prior_like::eltype(par), alarm_prior::Bool = prioreval(Tuple(par), Tuple(priors))
    alarm = false
    if alarm_prior
        IRFdist = 9.e15
        alarm = true
        State2Control = zeros(sr.n_par.ncontrols_r, sr.n_par.nstates_r)
        if e_set.debug_print
            println("Parameter try violates constraint")
        end
    else

        # replace estimated values in m_par by last candidate
        m_par = Flatten.reconstruct(m_par, par)

        # solve model using candidate parameters
        # BLAS.set_num_threads(1)
        State2Control::Array{eltype(par),2}, LOMstate::Array{eltype(par),2}, alarm_sgu::Bool = SGU_estim(sr, m_par, lr.A, lr.B; estim = true)

        # BLAS.set_num_threads(Threads.nthreads())
        if alarm_sgu # abort if model doesn't solve
            IRFdist = 9.e15
            alarm = true
            if e_set.debug_print
                println("Parameter try leads to inexistent or unstable equilibrium")
            end
        else
                        
            IRFs = compute_irfs(sr, State2Control, LOMstate, m_par, SHOCKs, indexes_sel_vars, isstate, e_set.irf_horizon)
            IRFdist = (sum((IRFs[:] .- IRFtargets[:]).^2 .* weights[:]) )./2
            # println(sum((IRFs[:] .- IRFtargets[:]).^2))
        end
    end

    return -IRFdist, prior_like, -IRFdist  .+ prior_like * e_set.prior_scale, alarm

end

function compute_irfs(sr, State2Control, LOMstate, m_par, SHOCKs, indexes_sel_vars, isstate, irf_horizon)

    n_vars = length(indexes_sel_vars)
    n_shocks = length(SHOCKs)                
    IRFs = Array{Float64}(undef, irf_horizon+1, n_vars, n_shocks)
    IRFsout = Array{Float64}(undef, irf_horizon, n_vars, n_shocks)

    shock_number = 0
    for i in SHOCKs
        x = zeros(size(LOMstate, 1))
        shock_number += 1
        x[getfield(sr.indexes_r, i)] = getfield(m_par, Symbol("σ_", i))

        MX = [I; State2Control]
        for t = 1:irf_horizon+1
            IRFs[t, :, shock_number] = (MX[indexes_sel_vars, :] * x)'
            x[:] = LOMstate * x
        end
    end
    IRFsout[:,isstate, :] .= IRFs[2:end, isstate, :] # IRFs for state variables represent end-of-period values
    IRFsout[:,.~isstate, :] .= IRFs[1:end-1, .~isstate, :] # IRFs for state variables represent end-of-period values

    return IRFsout

end