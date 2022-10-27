# This file defines the sets of aggregate shocks, states (inluding shocks), and controls
# The code checks that the number of equations in the aggregate model is equal to the number 
# of aggregate variables excluding the distributional summary statistics. The latter are not 
# contained in the aggregate model code as they are parameter free but change whenever the 
# distribution changes and do not show up in any aggregate model equation.

shock_names = [:Z]

state_names = [
    "Z","Nlag","Ilag","Clag","wlag"
]

# List cross-sectional controls / distributional summary variables (no equations in aggregate model expected)
distr_names   = ["GiniC", "GiniX", "TOP10Ishare", "TOP10Wshare", "sdlogy"]

control_names = [
    "r", "w", "K", "Y" ,"C", "N", "I",
    "Ngrowth", "Igrowth", "Cgrowth", "wgrowth"
]

# All controls in one array
control_names       = [distr_names; control_names]
# All names in one array
aggr_names          = [state_names; control_names]

# ascii names used for cases where unicode doesn't work, e.g., file saves
unicode2ascii(x)    = replace.(replace.(replace.(replace.(replace.(x,"τ"=>"tau"), "σ" => "sigma"),"π"=>"pi"),"μ"=>"mu"),"ρ"=>"rho")

state_names_ascii   = unicode2ascii(state_names)
control_names_ascii = unicode2ascii(control_names)
aggr_names_ascii    = [state_names_ascii; control_names_ascii]