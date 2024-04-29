function MakeTransition(
    m_n_star::Array{Float64,2},
    Π::Array{Float64,2},
    n_par::NumericalParameters,
)

    # create linear interpolation weights from policy functions
    idm_n, weightright_m_n, weightleft_m_n = MakeWeights(m_n_star, n_par.grid_m)

    # Generate indexes
    blockindex = (0:n_par.ny-1) * n_par.nm
    runindex = 0

    # Generate weights
    weight = zeros(typeof(m_n_star[1]), 2, n_par.ny, n_par.nm * n_par.ny)
    targetindex = zeros(Int, 2, n_par.ny, n_par.nm * n_par.ny)
    startindex = zeros(Int, 2, n_par.ny, n_par.nm * n_par.ny)
    runindex = 0
    for zz = 1:n_par.ny # all current income states
        for mm = 1:n_par.nm # all current asset states
            runindex = runindex + 1
            WL = weightleft_m_n[mm, zz]
            WR = weightright_m_n[mm, zz]
            CI = idm_n[mm, zz]
            for jj = 1:n_par.ny
                pp = Π[zz, jj]
                weight[1, jj, runindex] = WL .* pp
                weight[2, jj, runindex] = WR .* pp
                targetindex[1, jj, runindex] = CI .+ blockindex[jj]
                targetindex[2, jj, runindex] = CI .+ 1 .+ blockindex[jj]
                startindex[1, jj, runindex] = runindex
                startindex[2, jj, runindex] = runindex
            end
        end
    end
    S_n = startindex[:]
    T_n = targetindex[:]
    W_n = weight[:]

    return S_n, T_n, W_n
end
