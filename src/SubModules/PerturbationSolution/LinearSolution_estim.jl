@doc raw"""
    LinearSolution_estim(sr, m_par, A, B,;estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys`](@ref), while only differentiating with respect to the
aggregate part of the model, [`Fsys_agg()`](@ref).

The partials of the Jacobian belonging to the heterogeneous agent part of the model
are taken from the full-model derivatives provided as arguments, `A` and `B` (computed
by [`LinearSolution()`](@ref)).

# Arguments
- `sr`: steady-state structure (variable values, indexes, numerical parameters, ...)
- `A`,`B`: derivative of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
- `m_par`: model parameters

# Returns
as in [`LinearSolution()`](@ref)
"""
function LinearSolution_estim(
    sr::SteadyResults,
    m_par::ModelParameters,
    A::Array,
    B::Array;
    estim = true,
)

    ############################################################################
    # Calculate dericatives of non-lineear difference equation
    ############################################################################
    length_X0 = length(sr.XSSaggr)
    BA = ForwardDiff.jacobian(
        x -> Fsys_agg(
            x[1:length_X0],
            x[length_X0+1:end],
            sr.XSSaggr,
            sr.CDFSS,
            m_par,
            sr.n_par,
            sr.indexes_aggr,
        ),
        zeros(2 * length_X0),
    )

    for k in eachindex(aggr_names)
        if !(any(distr_names .== aggr_names[k]))
            j = getfield(sr.indexes, Symbol(aggr_names[k]))
            for h in eachindex(aggr_names)
                if !(any(distr_names .== aggr_names[h]))
                    i = getfield(sr.indexes, Symbol(aggr_names[h]))
                    A[j, i] = BA[k, h+length_X0]
                    B[j, i] = BA[k, h]
                end
            end
        end
    end

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    gx, hx, alarm_LinearSolution, nk = SolveDiffEq(A, B, sr.n_par, estim)

    return gx, hx, alarm_LinearSolution, nk, A, B
end
