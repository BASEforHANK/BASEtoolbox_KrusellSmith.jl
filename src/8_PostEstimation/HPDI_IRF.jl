###############################################################################################
# Compute IRFs and variance decomposition for set of models and variables passed to function
###############################################################################################
function HPDI_IRF(shock_names,select_variables,max_horizon,sr,lr, m_par, draws; n_replic = 10, percentile_bounds = (0.05, 0.95))
    
    n_shocks   = length(shock_names)
    n_vars     = length(select_variables)
    selector_r = []

    isstate = zeros(Bool, n_vars)
    iter = 1
    for i in select_variables
        if i in Symbol.(sr.state_names)
            isstate[iter] = true
        end
        iter += 1
        append!(selector_r, getfield(sr.indexes_r, i))
    end

    draw_ind = rand(1:size(draws, 1), n_replic)
    IRFMAT = hcat([zeros(n_replic) for j = 1:n_vars, i = 1:max_horizon, k = 1:n_shocks])
    for s = 1:length(draw_ind)
        A=lr.A; B=lr.B; State2Control=lr.State2Control; LOMstate=lr.LOMstate; SolutionError=lr.SolutionError; nk=lr.nk
        lr_local = LinearResults(copy(State2Control), copy(LOMstate), copy(A), copy(B), copy(SolutionError), copy(nk))
        # Allocate OBSMATs
        OBSMAT_reduc  = zeros(n_vars, max_horizon, n_shocks) 
        
        # write parameter draw from chain to m_par       
        dd = draws[draw_ind[s], :]
        if e_set.me_treatment != :fixed
            m_par_local = Flatten.reconstruct(m_par, dd[1:size(draws, 2)-length(e_set.meas_error_input)])
        else
            m_par_local = Flatten.reconstruct(m_par, dd)
        end

        # Update model solution w/ reduction
        lr_reduc = update_model(sr, lr_local, m_par_local)
        MX_reduc = [I; lr_reduc.State2Control] 
        # Fill OBSMAT_reduc
        for i = 1:n_shocks
            x0    = zeros(size(lr_reduc.LOMstate,1), 1) 
            x0[getfield(sr.indexes_r,shock_names[i])]   += getfield(m_par, Symbol("σ_", shock_names[i]))
            x     = x0 * ones(1, max_horizon + 1)
            IRFs  = zeros(sr.n_par.ntotal_r, max_horizon) 
            for t = 1:max_horizon
                IRFs[:, t]    = (MX_reduc * x[:, t])'
                x[:, t+1]     = lr_reduc.LOMstate * x[:, t]  
            end
            OBSMAT_reduc[:, :, i] = IRFs[selector_r, :]
        end
        for i = 1:n_shocks
            for d1 = 1:n_vars
                for d2 = 1:max_horizon
                        IRFMAT[d1, d2, i][s] = OBSMAT_reduc[d1, d2, i]
                end
            end
        end
    end
     IRF_lower  = quantile.(IRFMAT, percentile_bounds[1])
     IRF_upper  = quantile.(IRFMAT, percentile_bounds[2])
    #  L2_mean   = mean.(L2)
    return IRFMAT, IRF_lower, IRF_upper, draw_ind
end