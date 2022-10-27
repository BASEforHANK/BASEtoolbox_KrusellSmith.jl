#---------------------------------------------------------
# Discrete Cosine Transform
#---------------------------------------------------------
function  mydctmx(n::Int)
    DC::Array{Float64,2} =zeros(n,n);
    for j=0:n-1
        DC[1,j+1]=float(1/sqrt(n));
        for i=1:n-1
            DC[i+1,j+1]=(pi*float((j+1/2)*i/n));
            DC[i+1,j+1] = sqrt(2/n).*cos.(DC[i+1,j+1])
        end
    end
    return DC
end


function uncompress(compressionIndexes, XC, DC,IDC)
    nm = size(DC[1],1)
    ny = size(DC[2],1)
    # POTENTIAL FOR SPEEDUP BY SPLITTING INTO DUAL AND REAL PART AND USE BLAS
    θ1 =zeros(eltype(XC),nm,ny)
    for j  =1:length(XC)
        θ1[compressionIndexes[j]] = copy(XC[j])
    end
    θ1 = IDC[1]*θ1*DC[2]
    θ = reshape(θ1,(nm)*(ny))
    return θ
end

function compress(compressionIndexes::AbstractArray, XU::AbstractArray,
    DC::AbstractArray,IDC::AbstractArray)
    θ   = zeros(eltype(XU),length(compressionIndexes))
    XU2 = zeros(eltype(XU),size(XU))
    XU2 = DC[1]*XU*IDC[2]
    θ = XU2[compressionIndexes]
    return θ
end

function select_comp_ind(V,reduc_value)
    Theta             = dct(V)[:]                          # Discrete cosine transformation of marginal liquid asset value
    ind                 = sortperm(abs.(Theta[:]);rev=true)   # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                                     # Container to store the number of retained coefficients
    # Find the important basis functions (discrete cosine) for VmSS
    while norm(Theta[ind[1:coeffs]])/norm(Theta) < 1 - reduc_value 
            coeffs      += 1                                    # add retained coefficients until only n_par.reduc_value hare of energy is lost
    end
    select_ind  = ind[1:coeffs]  
    return select_ind
end
