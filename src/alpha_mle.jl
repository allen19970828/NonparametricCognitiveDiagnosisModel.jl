# src/alpha_mle.jl

"""
    alpha_mle(Y::Matrix{Int}, Q::Matrix{Int}, par::Union{ParMLEResult, Dict{Symbol, Any}}, model::Symbol; undefined_flag::Union{Nothing, Vector{Int}}=nothing) -> AlphaMLEResult

在已知題目參數的情況下，使用極大似然法 (MLE) 估計所有受試者的能力屬性組型。
"""
function alpha_mle(Y::Matrix{Int}, Q::Matrix{Int}, par::Union{ParMLEResult, Dict{Symbol, Any}}, model::Symbol; undefined_flag::Union{Nothing, Vector{Int}}=nothing)
    check_input(Y, Q)
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    M = 2^natt
    pattern = alpha_permute(natt)
    
    # 轉換參數結構為 Dict 以便於 cdl 函數處理
    par_dict = if par isa ParMLEResult
        d = Dict{Symbol, Any}()
        if model in (:DINA, :DINO, :NIDA, :GNIDA)
            d[:slip] = par.slip
            d[:guess] = par.guess
        elseif model == :RRUM
            d[:pi] = par.pi
            d[:r] = par.r
        end
        d
    else
        par
    end
    
    if isnothing(undefined_flag)
        undefined_flag = zeros(Int, nitem)
    end
    
    loglike_matrix = Matrix{Float64}(undef, M, nperson)
    alpha_est = Matrix{Int}(undef, nperson, natt)
    est_class = Vector{Int}(undef, nperson)
    n_tie = zeros(Int, nperson)
    class_tie = zeros(Int, nperson, M)
    
    for i in 1:nperson
        yi = Y[i, :]
        loglike = Vector{Float64}(undef, M)
        for m in 1:M
            loglike[m] = cdl(yi, Q, par_dict, pattern[m, :], model, undefined_flag)
        end
        loglike_matrix[:, i] = loglike
        
        # 尋找極大似然的組型
        max_val = maximum(loglike)
        max_indices = findall(x -> abs(x - max_val) < 1e-9, loglike)
        
        if length(max_indices) == 1
            est_class[i] = max_indices[1]
        else
            n_tie[i] = length(max_indices)
            for idx in 1:length(max_indices)
                class_tie[i, idx] = max_indices[idx]
            end
            # 平手時隨機選擇一個組型
            est_class[i] = rand(max_indices)
        end
        
        alpha_est[i, :] = pattern[est_class[i], :]
    end
    
    return AlphaMLEResult(
        alpha_est,
        est_class,
        pattern,
        n_tie,
        class_tie,
        loglike_matrix,
        Y,
        Q,
        par,
        model
    )
end
