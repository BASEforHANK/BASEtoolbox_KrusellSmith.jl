@metadata prior nothing
@metadata label ""
@metadata latex_label L""
@doc raw"""
ModelParameters()

Collect all model parameters with calibrated values / priors for estimation in a `struct`.

Uses packages `Parameters`, `FieldMetadata`, `Flatten`. Boolean value denotes
whether parameter is estimated.

# Example
```jldoctest
julia> m_par = ModelParameters();
julia> # Obtain vector of prior distributions of parameters that are estimated.
julia> priors = collect(metaflatten(m_par, prior))
```
"""
@label @latex_label @prior @flattenable @with_kw struct ModelParameters{T}
    # variable = value  | ascii name 	| LaTex name 	| Prior distribution | estimated? # description

    # Household preference parameters
    ξ::T = 1.0 | "xi" | L"\xi" | _ | false # risk aversion
    γ::T = 2.0 | "gamma" | L"\gamma" | _ | false # inverse Frisch elasticity
    β::T = 0.99 | "beta" | L"\beta" | _ | false # discount factor

    # Individual income process
    ρ_h::T = 0.98 | "rho" | L"\rho" | _ | false # autocorrelation income shock
    σ_h::T = 0.12 | "sigma" | L"\sigma" | _ | false # std of income shocks (steady state)

    # Technological parameters
    α::T = 0.318 | "alpha" | L"\alpha" | _ | false # capital share
    δ_0::T = (0.07 + 0.016) / 4 | "delta" | L"\delta" | _ | false # depreciation rate

    # exogeneous aggregate "shocks"
    ρ_Z::T = 0.9 | "rho_Z" | L"\rho_Z" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of TFP
    σ_Z::T = 0.01 | "sigma_Z" | L"\sigma_Z" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std of TFP


end

@doc raw"""
NumericalParameters()

Collect parameters for the numerical solution of the model in a `struct`.

Use package `Parameters` to provide initial values.

# Example
```jldoctest
julia> n_par = NumericalParameters(mmin = -6.6, mmax = 1000)
```
"""
@with_kw struct NumericalParameters
    # Numerical Parameters to be set in advance
    m_par::ModelParameters = ModelParameters()
    ny::Int = 11      # ngrid income (4 is the coarse grid used initially in finding the StE)
    nm::Int = 80        # ngrid liquid assets (bonds) (10 is the coarse grid used initially in finding the StE)
    ny_copula::Int = 10        # ngrid income for refinement
    nm_copula::Int = 10        # ngrid liquid assets (bonds)
    mmin::Float64 = 0.0      # gridmin bonds
    mmax::Float64 = 200.0    # gridmax bonds
    ϵ::Float64 = 1.0e-13 # precision of solution 

    sol_algo::Symbol = :schur # options: :schur (Klein's method), :lit (linear time iteration), :litx (linear time iteration with Howard improvement)
    verbose::Bool = true   # verbose model
    reduc_value::Float64 = 5e-7   # Lost fraction of "energy" in the DCT compression for value functions
    reduc_marginal_value::Float64 = 1e-3   # Lost fraction of "energy" in the DCT compression for value functions

    further_compress::Bool = true   # run model-reduction step based on MA(∞) representation
    further_compress_critC = eps()  # critical value for eigenvalues for Value functions
    further_compress_critS = ϵ      # critical value for eigenvalues for copula

    # Parameters that will be overwritten in the code
    aggr_names::Array{String,1} = ["Something"] # Placeholder for names of aggregates
    distr_names::Array{String,1} = ["Something"] # Placeholder for names of distributions

    naggrstates::Int = 16 # (placeholder for the) number of aggregate states
    naggrcontrols::Int = 16 # (placeholder for the) number of aggregate controls
    nstates::Int = ny + nm + naggrstates - 2 # (placeholder for the) number of states + controls in total
    ncontrols::Int = 16 # (placeholder for the) number of controls in total
    ntotal::Int = nstates + ncontrols # (placeholder for the) number of states+ controls in total
    n_agg_eqn::Int = nstates + ncontrols     # (placeholder for the) number of aggregate equations
    naggr::Int = length(aggr_names)     # (placeholder for the) number of aggregate states + controls
    ntotal_r::Int = nstates + ncontrols# (placeholder for the) number of states + controls in total after reduction
    nstates_r::Int = nstates# (placeholder for the) number of states in total after reduction
    ncontrols_r::Int = ncontrols# (placeholder for the) number of controls in total after reduction

    PRightStates::AbstractMatrix = Diagonal(ones(nstates)) # (placeholder for the) Matrix used for second stage reduction (states only)
    PRightAll::AbstractMatrix = Diagonal(ones(ntotal))  # (placeholder for the) Matrix used for second stage reduction 

    # income grid
    grid_y::Array{Float64,1} =
        exp.(Tauchen(m_par.ρ_h, ny)[1] .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h .^ 2))  # income grid

    # Transition matrix from income in steady state
    Π::Matrix{Float64} = Tauchen(m_par.ρ_h, ny)[2]

    # bounds of income bins
    bounds_y::Array{Float64,1} = Tauchen(m_par.ρ_h, ny)[3]# (placeholder) bonds of income bins (overwritten by Tauchen)

    H::Float64 = ((Π^1000)[1, 1:end]' * grid_y[1:end]) # stationary equilibrium average human capital
    HW::Float64 = (1.0 / (1.0 - (Π^1000)[1, end]))     # stationary equilibrium fraction workers

    # initial guess for stationary distribution (needed if iterative procedure is used)
    dist_guess::Array{Float64,2} = ones(nm, ny) / (nm * ny)

    # grid liquid assets:
    grid_m::Array{Float64,1} =
        exp.(range(0, stop = log(mmax - mmin + 1.0), length = nm)) .+ mmin .- 1.0

    # meshes for income, bonds, capital
    mesh_y::Array{Float64,2} = repeat(reshape(grid_y, (1, ny)), outer = [nm, 1])
    mesh_m::Array{Float64,2} = repeat(reshape(grid_m, (nm, 1)), outer = [1, ny])

    # grid for copula marginal distributions
    copula_marginal_m::Array{Float64,1} =
        collect(range(0.0, stop = 1.0, length = nm_copula))
    copula_marginal_y::Array{Float64,1} =
        collect(range(0.0, stop = 1.0, length = ny_copula))

    # Storage for linearization results
    LOMstate_save::Array{Float64,2} = zeros(nstates, nstates)
    State2Control_save::Array{Float64,2} = zeros(ncontrols, nstates)
end


@doc raw"""
EstimationSettings()

Collect settings for the estimation of the model parameters in a `struct`.

Use package `Parameters` to provide initial values. Input and output file names are
stored in the fields `mode_start_file`, `data_file`, `save_mode_file` and `save_posterior_file`.
"""
@with_kw struct EstimationSettings
    shock_names::Array{Symbol,1} = shock_names # set in 1_Model/input_aggregate_names.jl
    observed_vars_input::Array{Symbol,1} = [:Igrowth, :Cgrowth, :wgrowth]

    nobservables = length(observed_vars_input)

    estimation_type::Symbol = :likelihoodbased # options: :likelihoodbased or :irfmatching

    data_rename::Dict{Symbol,Symbol} = Dict(:pi => :π, :sigma2 => :σ)

    me_treatment::Symbol = :unbounded
    me_std_cutoff::Float64 = 0.2

    meas_error_input::Array{Symbol,1} = [:Igrowth, :Cgrowth, :wgrowth]
    meas_error_distr::Array{InverseGamma{Float64},1} = [
        InverseGamma(ig_pars(0.0005, 0.001^2)...),
        InverseGamma(ig_pars(0.0005, 0.001^2)...),
        InverseGamma(ig_pars(0.0005, 0.001^2)...),
    ]

    # Leave empty to start with prior mode
    mode_start_file::String = ""  #"7_Saves/parameter_example.jld2" 

    irf_horizon::Int = 15
    prior_scale::Float64 = 1.0 # scales importance of prior in IRF matching; set to 0 if frequentist
    irfdata_file::String = "irf_data_0706_inclPC.csv"

    data_file::String = "bbl_data_inequality.csv"
    save_mode_file::String = "7_Saves/HANC_mode.jld2"
    save_posterior_file::String = "7_Saves/HANC_chain.jld2"

    estimate_model::Bool = true

    max_iter_mode::Int = 100
    optimizer::Optim.AbstractOptimizer = NelderMead()
    compute_hessian::Bool = true    # true: computes Hessian at posterior mode; false: sets Hessian to identity matrix
    f_tol::Float64 = 1.0e-4
    x_tol::Float64 = 1.0e-4

    multi_chain_init::Bool = false
    ndraws::Int = 400
    burnin::Int = 100
    mhscale::Float64 = 0.00015
    debug_print::Bool = true
    seed::Int = 778187

end
