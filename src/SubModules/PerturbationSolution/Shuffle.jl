##########################################################
# Matrix to remove one degree of freedom from distribution
#---------------------------------------------------------
function shuffleMatrix(distr, n_par)
    distr_m = sum(distr, dims = 2) ./ sum(distr[:])
    distr_y = sum(distr, dims = 1) ./ sum(distr[:])
    Γ = Array{Array{Float64,2},1}(undef, 2)
    Γ[1] = zeros(Float64, (n_par.nm, n_par.nm - 1))
    Γ[2] = zeros(Float64, (n_par.ny, n_par.ny - 1))
    for j = 1:n_par.nm-1
        Γ[1][:, j] = -distr_m[:]
        Γ[1][j, j] = 1 - distr_m[j]
        Γ[1][j, j] = Γ[1][j, j] - sum(Γ[1][:, j])
    end
    for j = 1:n_par.ny-1
        Γ[2][:, j] = -distr_y[:]
        Γ[2][j, j] = 1 - distr_y[j]
        Γ[2][j, j] = Γ[2][j, j] - sum(Γ[2][:, j])
    end

    return Γ
end

function shuffleMatrix_1dim(distr_i, n_i)
    Γ = zeros(Float64, (n_i, n_i - 1))
    for j = 1:n_i-1
        Γ[:, j] = -distr_i[:]
        Γ[j, j] = 1 - distr_i[j]
        Γ[j, j] = Γ[j, j] - sum(Γ[:, j])
    end
    return Γ
end
