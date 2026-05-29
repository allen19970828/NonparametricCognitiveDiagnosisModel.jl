# src/probability.jl

"""
    clamp_prob(x::Real) -> Float64

輔助函數：將機率值為 0 或 1 的邊界值進行微小修正（與 R 語言中的 CDL / CDP 實現完全一致），以確保對數似然值計算的精度穩定。
"""
function clamp_prob(x::Real)
    if x == 0.0
        return 0.001
    elseif x == 1.0
        return 0.999
    else
        return Float64(x)
    end
end

"""
    cdp(Q_j::AbstractVector{Int}, par::Dict{Symbol, Any}, alpha::AbstractVector{Int}, model::Symbol) -> Float64

計算特定受試者在給定能力屬性組型下，答對某一個試題的機率 P(X_j = 1 | alpha)。
"""
function cdp(Q_j::AbstractVector{Int}, par::Dict{Symbol, Any}, alpha::AbstractVector{Int}, model::Symbol)
    natt = length(Q_j)
    
    # 複製參數以避免修改原始資料
    slip = copy(par[:slip])
    guess = copy(par[:guess])
    
    if model in (:DINA, :DINO, :NIDA, :GNIDA)
        slip .= clamp_prob.(slip)
        guess .= clamp_prob.(guess)
    elseif model == :RRUM
        pi_val = clamp_prob(par[:pi])
        r_val = clamp_prob.(par[:r])
    end
    
    if model == :DINA
        # ita = prod(alpha ^ Q_j)
        ita = prod(alpha .^ Q_j)
        P = (1.0 - slip[1])^ita * guess[1]^(1.0 - ita)
        
    elseif model == :DINO
        # omega = 1 - prod((1 - alpha) ^ Q_j)
        omega = 1 - prod((1 .- alpha) .^ Q_j)
        P = (1.0 - slip[1])^omega * guess[1]^(1.0 - omega)
        
    elseif model in (:NIDA, :GNIDA)
        # P = prod(((1 - slip)^alpha * guess^(1 - alpha))^Q_j)
        term = ((1.0 .- slip) .^ alpha) .* (guess .^ (1 .- alpha))
        P = prod(term .^ Q_j)
        
    elseif model == :RRUM
        # P = pi * prod(r ^ (Q_j * (1 - alpha)))
        term = r_val .^ (Q_j .* (1 .- alpha))
        P = pi_val * prod(term)
        
    else
        error("未知的模型類型: $model")
    end
    
    return P
end

# 為了方便傳入陣列的同名多載函數
function cdp(Q_j::AbstractVector{Int}, slip::Real, guess::Real, alpha::AbstractVector{Int}, model::Symbol)
    p_dict = Dict{Symbol, Any}(:slip => [slip], :guess => [guess])
    return cdp(Q_j, p_dict, alpha, model)
end

function cdp(Q_j::AbstractVector{Int}, slip::AbstractVector{<:Real}, guess::AbstractVector{<:Real}, alpha::AbstractVector{Int}, model::Symbol)
    p_dict = Dict{Symbol, Any}(:slip => slip, :guess => guess)
    return cdp(Q_j, p_dict, alpha, model)
end

function cdp(Q_j::AbstractVector{Int}, pi_val::Real, r_val::AbstractVector{<:Real}, alpha::AbstractVector{Int}, model::Symbol)
    p_dict = Dict{Symbol, Any}(:pi => pi_val, :r => r_val)
    return cdp(Q_j, p_dict, alpha, model)
end


"""
    cdl(Y::AbstractVector{Int}, Q::Matrix{Int}, par::Dict{Symbol, Any}, alpha::AbstractVector{Int}, model::Symbol, undefined_flag::Union{Nothing, Vector{Int}}=nothing) -> Float64

計算單一受試者（答對反應向量為 Y，屬性組型為 alpha）的總對數似然值 (Log-Likelihood)。
"""
function cdl(Y::AbstractVector{Int}, Q::Matrix{Int}, par::Dict{Symbol, Any}, alpha::AbstractVector{Int}, model::Symbol, undefined_flag::Union{Nothing, Vector{Int}}=nothing)
    nitem = length(Y)
    natt = size(Q, 2)
    
    if isnothing(undefined_flag)
        undefined_flag = zeros(Int, nitem)
    end
    
    # 取得本試題的模型參數並克隆，以防修改原引數
    slip_orig = get(par, :slip, nothing)
    guess_orig = get(par, :guess, nothing)
    pi_orig = get(par, :pi, nothing)
    r_orig = get(par, :r, nothing)
    
    # 進行機率的邊界 clamping 以利數值精度
    if model in (:DINA, :DINO, :NIDA, :GNIDA)
        slip = clamp_prob.(slip_orig)
        guess = clamp_prob.(guess_orig)
    elseif model == :RRUM
        pi_val = clamp_prob.(pi_orig)
        r_val = clamp_prob.(r_orig)
    end
    
    loglike = 0.0
    
    if model == :DINA
        # ita[j] = prod(alpha ^ Q[j, :])
        for j in 1:nitem
            if undefined_flag[j] == 0
                ita = prod(alpha .^ Q[j, :])
                prob_correct = (1.0 - slip[j])^ita * guess[j]^(1.0 - ita)
                # 為確保精度直接使用公式
                term1 = Y[j] * ita * log(1.0 - slip[j])
                term2 = (1 - Y[j]) * ita * log(slip[j])
                term3 = Y[j] * (1 - ita) * log(guess[j])
                term4 = (1 - Y[j]) * (1 - ita) * log(1.0 - guess[j])
                loglike += term1 + term2 + term3 + term4
            end
        end
        
    elseif model == :DINO
        # omega[j] = 1 - prod((1 - alpha) ^ Q[j, :])
        for j in 1:nitem
            if undefined_flag[j] == 0
                omega = 1 - prod((1 .- alpha) .^ Q[j, :])
                term1 = Y[j] * omega * log(1.0 - slip[j])
                term2 = (1 - Y[j]) * omega * log(slip[j])
                term3 = Y[j] * (1 - omega) * log(guess[j])
                term4 = (1 - Y[j]) * (1 - omega) * log(1.0 - guess[j])
                loglike += term1 + term2 + term3 + term4
            end
        end
        
    elseif model == :NIDA
        for j in 1:nitem
            if undefined_flag[j] == 0
                # P = prod(((1 - s) ^ alpha * g ^ (1 - alpha)) ^ Q_j)
                term = ((1.0 .- slip) .^ alpha) .* (guess .^ (1 .- alpha))
                P = prod(term .^ Q[j, :])
                
                # 計算精準的對數
                logP = sum(alpha .* Q[j, :] .* log.(1.0 .- slip) .+ (1 .- alpha) .* Q[j, :] .* log.(guess))
                
                # 累加對數似然值
                loglike += Y[j] * logP + (1 - Y[j]) * log(1.0 - P)
            end
        end
        
    elseif model == :GNIDA
        for j in 1:nitem
            if undefined_flag[j] == 0
                # slip[j, :] 與 guess[j, :] 是矩陣中第 j 題的參數
                s_j = slip[j, :]
                g_j = guess[j, :]
                
                term = ((1.0 .- s_j) .^ alpha) .* (g_j .^ (1 .- alpha))
                P = prod(term .^ Q[j, :])
                
                logP = sum(alpha .* Q[j, :] .* log.(1.0 .- s_j) .+ (1 .- alpha) .* Q[j, :] .* log.(g_j))
                loglike += Y[j] * logP + (1 - Y[j]) * log(1.0 - P)
            end
        end
        
    elseif model == :RRUM
        for j in 1:nitem
            if undefined_flag[j] == 0
                pi_j = pi_val[j]
                r_j = r_val[j, :]
                
                P = pi_j * prod(r_j .^ (Q[j, :] .* (1 .- alpha)))
                logP = log(pi_j) + sum(Q[j, :] .* (1 .- alpha) .* log.(r_j))
                
                loglike += Y[j] * logP + (1 - Y[j]) * log(1.0 - P)
            end
        end
    else
        error("未知的模型類型: $model")
    end
    
    return loglike
end
