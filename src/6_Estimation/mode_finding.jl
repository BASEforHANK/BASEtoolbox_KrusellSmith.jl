@doc raw"""
    mode_finding(XSSaggr, A, B, indexes, indexes_aggr, distrSS, compressionIndexes, m_par, n_par, e_set)

Given definition of observed variables and their transformation (level or growth rate) from `e_set`,
load the data, construct the observation equation, and maximize [`likeli()`](@ref) (the log-likelihood)
using the package `Optim`.

Save estimation results to `e_set.save_mode_file`.

# Returns
- `par_final`: parameter vector that maximizes the likelihood
- `hessian_final`: Hessian of the log-likelihood at `par_final`
- `posterior_mode`: log-likelihood at `par_final`
- `meas_error`,`meas_error_std`: returns from [`measurement_error()`](@ref)
- `parnames`: names of estimated parameters (including measurement error variances)
- `Data`,`Data_missing`: data from `e_set.data_file`; marker for missing data
- `H_sel`: selector matrix for states/controls that are observed
- `priors`: priors of parameters (including measurement error variances)
- `smoother_output`: output from the Kalman smoother
"""
function mode_finding_irf(sr, lr, m_par, e_set, par_start)

    # get names of estimated parameters and add measurement error params
    parnames = collect(fieldnameflatten(m_par))
    
    # initialize parameters at starting values
    par = copy(par_start)

    # Prior specification
    priors = collect(metaflatten(m_par, prior)) # model parameters

    if e_set.estimation_type == :likelihoodbased

        # Load data
        Data_temp = DataFrame(CSV.File(e_set.data_file; missingstring = "NaN"))
        data_names_temp = propertynames(Data_temp)

        # Rename observables that do not have matching model names
        for i in data_names_temp
            name_temp = get(e_set.data_rename, i, :none)
            if name_temp != :none
            rename!(Data_temp, Dict(i => name_temp))
            end
        end

        # Identify missing observations
        observed_vars = e_set.observed_vars_input
        Data = Matrix(Data_temp[:, observed_vars])
        Data_missing = ismissing.(Data)

        # Built selection matrix
        H_sel = zeros(e_set.nobservables, sr.n_par.nstates_r + sr.n_par.ncontrols_r)
        for i in eachindex(observed_vars)
            H_sel[i, getfield(sr.indexes_r, (observed_vars[i]))] = 1.0
        end

        # Set up measurement error
        meas_error, meas_error_prior, meas_error_std = measurement_error(Data, observed_vars, e_set)

        if e_set.me_treatment != :fixed
            m_par = Flatten.reconstruct(m_par, par[1:length(par)-length(meas_error)])
            append!(priors, meas_error_prior)          # add the meas. error priors
            for i in eachindex(e_set.meas_error_input)
                push!(parnames, Symbol(:σ_me_, e_set.meas_error_input[i]))
            end
        else
            m_par = Flatten.reconstruct(m_par, par)
        end

        # Optimization
        # Define objective function
        Laux(pp)  = -likeli(pp, Data, Data_missing, H_sel, priors, meas_error, 
                        meas_error_std, sr, lr, m_par, e_set)[3]

 
        # Code variant with box-constrained optimization, used for updating compression

        OptOpt  = Optim.Options(show_trace = true, show_every = 20, store_trace = true, x_tol = e_set.x_tol,
                                f_tol = e_set.f_tol, iterations = div(e_set.max_iter_mode, 10))
        opti    = optimize(Laux, par_start , e_set.optimizer, OptOpt)
        par_final = Optim.minimizer(opti)
        # Update estimated model parameters and resolve model
        if e_set.me_treatment != :fixed
            m_par = Flatten.reconstruct(m_par, par_final[1:length(par_final)-length(meas_error)])
        else
            m_par = Flatten.reconstruct(m_par, par_final)
        end
        println("updating model reduction after initial optimization")
        @set! sr.n_par.further_compress = false 
        sr_aux = model_reduction(sr, lr, m_par) # revert to full model
        lr_aux = update_model(sr_aux, lr, m_par) # solve full model
        @set! sr_aux.n_par.further_compress = true
        sr = model_reduction(sr_aux, lr_aux, m_par) # update model reduction
        lr = update_model(sr, lr_aux, m_par)   # solve new reduced model

        # Built selection matrix
        H_sel = zeros(e_set.nobservables, sr.n_par.nstates_r + sr.n_par.ncontrols_r)
        for i in eachindex(observed_vars)
            H_sel[i, getfield(sr.indexes_r, (observed_vars[i]))] = 1.0
        end

        # Define closure of objective function
        LL(pp)  = -likeli(pp, Data, Data_missing, H_sel, priors, meas_error, 
                        meas_error_std, sr, lr, m_par, e_set)[3]

        OptOpt  = Optim.Options(show_trace = true, show_every = 20, store_trace = true, x_tol = e_set.x_tol,
                                f_tol = e_set.f_tol, iterations = e_set.max_iter_mode)
        opti    = optimize(LL, Optim.minimizer(opti) , e_set.optimizer, OptOpt)
        par_final = Optim.minimizer(opti)
  
        # Update estimated model parameters and resolve model
        if e_set.me_treatment != :fixed
            m_par = Flatten.reconstruct(m_par, par_final[1:length(par_final)-length(meas_error)])
        else
            m_par = Flatten.reconstruct(m_par, par_final)
        end
        ll_old = -Optim.minimum(opti)
        
        println("updating model reduction after mode finding finished")
        @set! sr.n_par.further_compress = false 
        sr_aux = model_reduction(sr, lr, m_par) # revert to full model
        lr_aux = update_model(sr_aux, lr, m_par) # solve full model
        println("new reduction")
        @set! sr_aux.n_par.further_compress = true
        sr = model_reduction(sr_aux, lr_aux, m_par) # update model reduction
        lr = update_model(sr, lr_aux, m_par)   # solve new reduced model
        # Built selection matrix
        H_sel = zeros(e_set.nobservables, sr.n_par.nstates_r + sr.n_par.ncontrols_r)
        for i in eachindex(observed_vars)
            H_sel[i, getfield(sr.indexes_r, (observed_vars[i]))] = 1.0
        end

        LL_final(pp)  = -likeli(pp, Data, Data_missing, H_sel, priors, meas_error, 
                                meas_error_std, sr, lr, m_par, e_set)[3]
        
        posterior_mode = -LL_final(par_final)
        println("Likelihood at mode under ... reduction")
        println("old: ",ll_old, " new: ", posterior_mode)
        
        # Run Kalman smoother
        smoother_output = likeli(par_final, Data, Data_missing, H_sel, priors,
                                meas_error, meas_error_std, sr, lr, m_par, 
                                e_set; smoother = true)

        # Compute Hessian at posterior mode
        if e_set.compute_hessian == true
            if sr.n_par.verbose
            println("Computing Hessian. This might take a while...")
            end
            func          = TwiceDifferentiable(pp -> LL_final(pp), par_final)
            hessian_final = Optim.hessian!(func, par_final)
        else 
            if sr.n_par.verbose
            println("Assuming Hessian is I...")
            end
            hessian_final = Matrix{Float64}(I, length(par_final), length(par_final))
        end

        IRFtargets = []
        standard_errors = []
        
        return par_final, hessian_final, posterior_mode, meas_error, meas_error_std, parnames, 
          Data, Data_missing, IRFtargets, standard_errors, H_sel, priors, smoother_output, m_par, sr, lr

    elseif e_set.estimation_type == :irfmatching
        irf_horizon = e_set.irf_horizon
        # Load irf targets
        Data_temp = DataFrame(CSV.File(e_set.irfdata_file; missingstring = "NaN"))
        shock_names = Symbol.(unique(Data_temp[:, :shock]))
        shocks_selected = intersect(shock_names, e_set.shock_names)
        select_variables = intersect(Symbol.(propertynames(Data_temp)), e_set.observed_vars_input)
        
        IRFtargets = Array{Float64}(undef, irf_horizon, length(select_variables),length(shocks_selected))
        standard_errors = Array{Float64}(undef, irf_horizon, length(select_variables),length(shocks_selected))
        # standard_errors = ones(irf_horizon, length(select_variables),length(shocks_selected))

        counti = 0
        for i in shocks_selected
            counti += 1
            countj  = 0
            for j in select_variables
                countj += 1
                IRFtargets[:, countj, counti] = Data_temp[(Data_temp[:, :pointdum] .== 1) .& (Symbol.(Data_temp[:, :shock]).==i), j]
                standard_errors[:, countj, counti] = Data_temp[(Data_temp[:, :pointdum] .== 0) .& (Symbol.(Data_temp[:, :shock]).==i), j]
            end
        end
        IRFtargets=IRFtargets./maximum(Data_temp[:,:B])
        standard_errors=standard_errors./maximum(Data_temp[:,:B])

        weights = 1.0 ./ (standard_errors.^2)
        # weights = weights_temp ./ sum(weights_temp[:]) .* length(weights_temp[:])
        iter = 1
        indexes_sel_vars = []
        isstate = zeros(Bool, length(select_variables))
        for i in select_variables
            if i in Symbol.(sr.state_names)
                isstate[iter] = true
            end
            iter += 1
            append!(indexes_sel_vars, getfield(sr.indexes_r, i))
        end
    
        m_par = Flatten.reconstruct(m_par, par)
        
        # Optimization
        # Define objective function
        Lirfaux(pp)  = -irfmatch(pp, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)[3]

        # Code variant with box-constrained optimization, used for updating compression

        # OptOpt  = Optim.Options(show_trace = true, show_every = 20, store_trace = true, x_tol = e_set.x_tol,
        #                         f_tol = e_set.f_tol, iterations = div(e_set.max_iter_mode, 10))
        OptOpt  = Optim.Options(show_trace = true, show_every = 20, store_trace = true, x_tol = e_set.x_tol,
        f_tol = e_set.f_tol, iterations = e_set.max_iter_mode)
        opti    = optimize(Lirfaux, par_start , e_set.optimizer, OptOpt)
        par_final = Optim.minimizer(opti)
        # Update estimated model parameters and resolve model
        # m_par = Flatten.reconstruct(m_par, par_final)

        # println("updating model reduction after initial optimization")
        # @set! sr.n_par.further_compress = false 
        # sr_aux = model_reduction(sr, lr, m_par) # revert to full model
        # lr_aux = update_model(sr_aux, lr, m_par) # solve full model
        
        # @set! sr_aux.n_par.further_compress = true
        # sr = model_reduction(sr_aux, lr_aux, m_par) # update model reduction
        # lr = update_model(sr, lr_aux, m_par)   # solve new reduced model

        # iter = 1
        # indexes_sel_vars = []
        # isstate = zeros(Bool, length(select_variables))
        # for i in select_variables
        #     if i in Symbol.(sr.state_names)
        #         isstate[iter] = true
        #     end
        #     iter += 1
        #     append!(indexes_sel_vars, getfield(sr.indexes_r, i))
        # end
        # # Define closure of objective function
        # LLirf(pp)  = -irfmatch(pp, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)[3]

        # OptOpt  = Optim.Options(show_trace = true, show_every = 20, store_trace = true, x_tol = e_set.x_tol,
        #                         f_tol = e_set.f_tol, iterations = e_set.max_iter_mode)
        # opti    = optimize(LLirf, Optim.minimizer(opti) , e_set.optimizer, OptOpt)
        # par_final = Optim.minimizer(opti)
  
        # Update estimated model parameters and resolve model
        
        m_par = Flatten.reconstruct(m_par, par_final)
        ll_old = -Optim.minimum(opti)
        
        println("updating model reduction after mode finding finished")
        @set! sr.n_par.further_compress = false 
        sr_aux = model_reduction(sr, lr, m_par) # revert to full model
        lr_aux = update_model(sr_aux, lr, m_par) # solve full model
        
        println("new reduction")
        @set! sr_aux.n_par.further_compress = true
        sr = model_reduction(sr_aux, lr_aux, m_par) # update model reduction
        lr = update_model(sr, lr_aux, m_par)   # solve new reduced model

        iter = 1
        indexes_sel_vars = []
        isstate = zeros(Bool, length(select_variables))
        for i in select_variables
            if i in Symbol.(sr.state_names)
                isstate[iter] = true
            end
            iter += 1
            append!(indexes_sel_vars, getfield(sr.indexes_r, i))
        end
        
        LLirf_final(pp)  = -irfmatch(pp, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)[3]
        
        posterior_mode = -LLirf_final(par_final)
        println("L2 under ... reduction")
        println("old: ",ll_old, " new: ", posterior_mode)


        # Compute Hessian at posterior mode
        if e_set.compute_hessian == true
            if sr.n_par.verbose
            println("Computing Hessian. This might take a while...")
            end
            func          = TwiceDifferentiable(pp -> LLirf_final(pp), par_final)
            hessian_final = Optim.hessian!(func, par_final)
        else 
            if sr.n_par.verbose
            println("Assuming Hessian is I...")
            end
            hessian_final = Matrix{Float64}(I, length(par_final), length(par_final))
        end

        meas_error = []
        meas_error_std = []
        H_sel = []
        smoother_output = []
        Data = []
        Data_missing = []
        # hessian_final = []

        return par_final, hessian_final, posterior_mode, meas_error, meas_error_std, parnames, 
            Data, Data_missing, IRFtargets, standard_errors, H_sel, priors, smoother_output, m_par, sr, lr

    else
        error("estimation type not defined")
    end

end
