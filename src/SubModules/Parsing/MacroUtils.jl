@doc raw"""
    @generate_equations()

Write out the expansions around steady state for all variables in `aggr_names`,
i.e. generate code that reads aggregate states/controls from steady state deviations.

Equations take the form of (with variable `r` as example):
- `r       = exp.(XSS[indexes.rSS] .+ X[indexes.r])`
- `rPrime  = exp.(XSS[indexes.rSS] .+ XPrime[indexes.r])`

# Requires
(module) global `aggr_names`
"""
macro generate_equations()
    ex = quote end # initialize expression to append to
    for j in aggr_names # loop over variables to generate
        i = Symbol(j)
        varnamePrime = Symbol(i, "Prime")
        varnameSS = Symbol(i, "SS")
        ex_aux = quote
            $i = exp.(XSS[indexes.$varnameSS] .+ X[indexes.$i])
            $varnamePrime = exp.(XSS[indexes.$varnameSS] .+ XPrime[indexes.$i])
        end

        append!(ex.args, ex_aux.args) # append to expression
    end

    return esc(ex)
end

"""
    @writeXSS()

Write all single steady state variables into vectors XSS / XSSaggr.

# Requires
(module) globals `state_names`, `control_names`, `aggr_names`
"""
macro writeXSS()
    # This macro writes all single steady state variables into a Vector XSS / XSSaggr
    # The variable names are expected to be in the GLOBALs state_names,
    # control_names, and aggr_names.
    #
    ex = quote
        XSS = [CDF_m[:]; PDF_y[:]; CDFSS[:]] # distrSS
    end
    for j in state_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSS, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end

    ex_aux = quote
        append!(XSS, VmSS[:]) # value function controls
    end
    append!(ex.args, ex_aux.args)
    for j in control_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSS, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end
    ex_aux = quote
        XSSaggr = [0.0]
    end
    append!(ex.args, ex_aux.args)
    for j in aggr_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSSaggr, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end
    ex_aux = quote
        deleteat!(XSSaggr, 1)
    end
    append!(ex.args, ex_aux.args)

    return esc(ex)
end

# macro include(filename::AbstractString)
#     path = joinpath(dirname(String(__source__.file)), filename)
#     return esc(Meta.parse("quote; " * read(path, String) * "; end").args[1])
# end

##########################################################
# Indexes
#---------------------------------------------------------

@doc raw"""
	@make_fnaggr(fn_name)

Create function `fn_name` that returns an instance of `struct` `IndexStructAggr`
(created by [`@make_struct_aggr`](@ref)), mapping aggregate states and controls to values
`1` to `length(aggr_names)` (both steady state and deviation from it).

# Requires
(module) global `aggr_names`
"""
macro make_fnaggr(fn_name)
    # state_names=Symbol.(aggr_names)
    n_states = length(aggr_names)

    fieldsSS_states = [:($i) for i = 1:n_states]
    fields_states = [:($i) for i = 1:n_states]
    esc(quote
        function $(fn_name)(n_par)
            indexes = IndexStructAggr($(fieldsSS_states...), $(fields_states...))
            return indexes
        end
    end)
end

@doc raw"""
	@make_fn(fn_name)

Create function `fn_name` that returns an instance of `struct` `IndexStruct`
(created by [`@make_struct`](@ref)), mapping states and controls to indexes
inferred from numerical parameters and compression indexes.

# Requires
(module) global `state_names`, `control_names`
"""
macro make_fn(fn_name)
    n_states = length(state_names)
    n_controls = length(control_names)

    fieldsSS_states = [:((tNo + tNo2) + $i) for i = 1:n_states]
    fields_states = [:(tNo + tNo4 - 2 + $i) for i = 1:n_states]
    fieldsSS_controls = [:(tNo + 2 * tNo2 + $i) for i in n_states .+ (1:n_controls)]
    fields_controls = [:(tNo + tNo3 + tNo4 - 2 + $i) for i in n_states .+ (1:n_controls)]
    esc(
        quote
            function $(fn_name)(n_par, compressionIndexesVm, compressionIndexesD)
                tNo = n_par.nm + n_par.ny
                tNo2 = n_par.nm * n_par.ny
                tNo3 = length(compressionIndexesVm)
                tNo4 = length(compressionIndexesD)
                indexes = IndexStruct(
                    1:n_par.nm, # distr_mSS
                    (n_par.nm+1):(tNo), # distr_ySS
                    (tNo+1):(tNo+tNo2), # VDSS
                    $(fieldsSS_states...),
                    ((tNo+tNo2)+$(n_states)+1):((tNo+tNo2)+tNo2+$(n_states)), # VmSS
                    $(fieldsSS_controls...),
                    1:(n_par.nm-1), # distr_m
                    n_par.nm:(tNo-2), # distr_y
                    (tNo-1):(tNo+tNo4-2), # VD
                    $(fields_states...),
                    (tNo+tNo4+$(n_states)-1):(tNo+tNo4+length(compressionIndexesVm)+$(
                        n_states
                    )-2), # Vm
                    $(fields_controls...),
                )
                return indexes
            end
        end,
    )
end
