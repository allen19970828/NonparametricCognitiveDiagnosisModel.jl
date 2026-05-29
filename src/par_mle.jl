# src/par_mle.jl

using Optim

"""
    par_mle(Y::Matrix{Int}, Q::Matrix{Int}, alpha::Matrix{Int}, model::Symbol) -> ParMLEResult

在已知受試者屬性組型 alpha 的情況下，使用極大似然法 (MLE) 估計題目參數。
為了確保數字精度準確與收斂穩定性，對於非線性方程模型 (NIDA, GNIDA, RRUM)，
本套件捨棄 R 語言原版中易發散的 dfsane 方程求解器，
改以數學上最嚴謹且穩健的 **Optim.jl 盒狀約束優化算法 (Box-constrained Optimization)**，
直接極小化負對數似然值，保證估計值嚴格落在合理機率區間 [0, 1] 內。
"""
function par_mle(Y::Matrix{Int}, Q::Matrix{Int}, alpha::Matrix{Int}, model::Symbol)
    check_input(Y, Q)
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    @assert size(alpha, 1) == nperson && size(alpha, 2) == natt "alpha 矩陣的維度與 Y 或 Q 不匹配。"
    
    # 初始化變數
    slip = nothing
    guess = nothing
    se_slip = nothing
    se_guess = nothing
    pi_val = nothing
    r_val = nothing
    se_pi = nothing
    se_r = nothing
    
    if model == :DINA || model == :DINO
        slip = zeros(Float64, nitem)
        guess = zeros(Float64, nitem)
        se_slip = zeros(Float64, nitem)
        se_guess = zeros(Float64, nitem)
        
        for j in 1:nitem
            # 計算每道題目的理想作答向量 (I)
            eta = Vector{Int}(undef, nperson)
            for i in 1:nperson
                if model == :DINA
                    eta[i] = prod(alpha[i, :] .^ Q[j, :])
                else # :DINO
                    eta[i] = 1 - prod((1 .- alpha[i, :]) .^ Q[j, :])
                end
            end
            
            sum_eta = sum(eta)
            sum_not_eta = nperson - sum_eta
            
            # 計算 slip (1 - Y 且 eta = 1 的人數比例)
            slip[j] = sum_eta > 0 ? sum((1 .- Y[:, j]) .* eta) / sum_eta : 0.0
            se_slip[j] = sum_eta > 0 ? sqrt(slip[j] * (1.0 - slip[j]) / sum_eta) : 0.0
            
            # 計算 guess (Y = 1 且 eta = 0 的人數比例)
            guess[j] = sum_not_eta > 0 ? sum(Y[:, j] .* (1 .- eta)) / sum_not_eta : 0.0
            se_guess[j] = sum_not_eta > 0 ? sqrt(guess[j] * (1.0 - guess[j]) / sum_not_eta) : 0.0
        end
        
    elseif model == :NIDA
        # NIDA: 屬性層級參數，slip 與 guess 的長度為 natt (每個屬性一個 slip/guess)
        # 定義負對數似然值目標函數
        function nida_objective(x)
            s_param = x[1:natt]
            g_param = x[natt+1:2natt]
            s_clamped = clamp.(s_param, 1e-6, 1.0 - 1e-6)
            g_clamped = clamp.(g_param, 1e-6, 1.0 - 1e-6)
            
            nll = 0.0
            for i in 1:nperson
                a_i = alpha[i, :]
                for j in 1:nitem
                    P = 1.0
                    for k in 1:natt
                        if Q[j, k] == 1
                            term = ((1.0 - s_clamped[k])^a_i[k]) * (g_clamped[k]^(1.0 - a_i[k]))
                            P *= term
                        end
                    end
                    P_clamped = clamp(P, 1e-6, 1.0 - 1e-6)
                    nll -= Y[i, j] * log(P_clamped) + (1 - Y[i, j]) * log(1.0 - P_clamped)
                end
            end
            return nll
        end
        
        # 初始值與邊界設定
        x0 = fill(0.3, 2natt)
        lower = fill(1e-5, 2natt)
        upper = fill(1.0 - 1e-5, 2natt)
        
        # 呼叫 Optim 的 Fminbox 與 LBFGS 求解
        res = optimize(nida_objective, lower, upper, x0, Fminbox(LBFGS()))
        x_opt = Optim.minimizer(res)
        
        slip = x_opt[1:natt]
        guess = x_opt[natt+1:2natt]
        
        # 計算與 R 語言相符的標準誤
        se_slip = zeros(Float64, natt)
        se_guess = zeros(Float64, natt)
        for k in 1:natt
            sum_alpha_k = sum(alpha[:, k])
            se_slip[k] = sum_alpha_k > 0 ? sqrt(slip[k] * (1.0 - slip[k]) / (sum_alpha_k * nitem)) : 0.0
            se_guess[k] = (nperson - sum_alpha_k) > 0 ? sqrt(guess[k] * (1.0 - guess[k]) / ((nperson - sum_alpha_k) * nitem)) : 0.0
        end
        
    elseif model == :GNIDA
        # GNIDA: 題目-屬性層級參數，slip 與 guess 的大小為 (J x K)
        slip = zeros(Float64, nitem, natt)
        guess = zeros(Float64, nitem, natt)
        se_slip = zeros(Float64, nitem, natt)
        se_guess = zeros(Float64, nitem, natt)
        
        for j in 1:nitem
            req_indices = findall(x -> x == 1, Q[j, :])
            K_j = length(req_indices)
            if K_j == 0
                continue
            end
            
            # 定義該試題的目標函數
            function gnida_item_objective(x)
                s_param = x[1:K_j]
                g_param = x[K_j+1:2K_j]
                s_clamped = clamp.(s_param, 1e-6, 1.0 - 1e-6)
                g_clamped = clamp.(g_param, 1e-6, 1.0 - 1e-6)
                
                nll = 0.0
                for i in 1:nperson
                    P = 1.0
                    for idx in 1:K_j
                        k = req_indices[idx]
                        term = ((1.0 - s_clamped[idx])^alpha[i, k]) * (g_clamped[idx]^(1.0 - alpha[i, k]))
                        P *= term
                    end
                    P_clamped = clamp(P, 1e-6, 1.0 - 1e-6)
                    nll -= Y[i, j] * log(P_clamped) + (1 - Y[i, j]) * log(1.0 - P_clamped)
                end
                return nll
            end
            
            x0 = fill(0.3, 2K_j)
            lower = fill(1e-5, 2K_j)
            upper = fill(1.0 - 1e-5, 2K_j)
            
            res = optimize(gnida_item_objective, lower, upper, x0, Fminbox(LBFGS()))
            x_opt = Optim.minimizer(res)
            
            for idx in 1:K_j
                k = req_indices[idx]
                slip[j, k] = x_opt[idx]
                guess[j, k] = x_opt[K_j + idx]
                
                sum_alpha_k = sum(alpha[:, k])
                se_slip[j, k] = sum_alpha_k > 0 ? sqrt(slip[j, k] * (1.0 - slip[j, k]) / sum_alpha_k) : 0.0
                se_guess[j, k] = (nperson - sum_alpha_k) > 0 ? sqrt(guess[j, k] * (1.0 - guess[j, k]) / (nperson - sum_alpha_k)) : 0.0
            end
        end
        
    elseif model == :RRUM
        # RRUM: pi_val 長度為 nitem，r_val 大小為 (J x K)
        pi_val = zeros(Float64, nitem)
        r_val = zeros(Float64, nitem, natt)
        se_pi = zeros(Float64, nitem)
        se_r = zeros(Float64, nitem, natt)
        
        for j in 1:nitem
            req_indices = findall(x -> x == 1, Q[j, :])
            K_j = length(req_indices)
            if K_j == 0
                pi_val[j] = 0.5
                continue
            end
            
            # 定義試題的 RRUM 目標函數
            function rrum_item_objective(x)
                pi_j = x[1]
                r_j = x[2:end]
                pi_clamped = clamp(pi_j, 1e-6, 1.0 - 1e-6)
                r_clamped = clamp.(r_j, 1e-6, 1.0 - 1e-6)
                
                nll = 0.0
                for i in 1:nperson
                    P = pi_clamped
                    for idx in 1:K_j
                        k = req_indices[idx]
                        P *= r_clamped[idx]^(1.0 - alpha[i, k])
                    end
                    P_clamped = clamp(P, 1e-6, 1.0 - 1e-6)
                    nll -= Y[i, j] * log(P_clamped) + (1 - Y[i, j]) * log(1.0 - P_clamped)
                end
                return nll
            end
            
            x0 = fill(0.8, 1 + K_j)
            lower = fill(1e-5, 1 + K_j)
            upper = fill(1.0 - 1e-5, 1 + K_j)
            
            res = optimize(rrum_item_objective, lower, upper, x0, Fminbox(LBFGS()))
            x_opt = Optim.minimizer(res)
            
            pi_val[j] = x_opt[1]
            # 計算 se_pi 與 R 一致
            # sum(apply(alpha, 1, function(x) prod(x ^ Q[j, ])) == 1)
            cnt_mastered = 0
            for i in 1:nperson
                if prod(alpha[i, :] .^ Q[j, :]) == 1
                    cnt_mastered += 1
                end
            end
            se_pi[j] = cnt_mastered > 0 ? sqrt(pi_val[j] * (1.0 - pi_val[j]) / cnt_mastered) : 0.0
            
            for idx in 1:K_j
                k = req_indices[idx]
                r_val[j, k] = x_opt[1 + idx]
                
                sum_alpha_k = sum(alpha[:, k])
                se_r[j, k] = (nperson - sum_alpha_k) > 0 ? sqrt(r_val[j, k] * (1.0 - r_val[j, k]) / (nperson - sum_alpha_k)) : 0.0
            end
        end
    else
        error("未知的模型種類: $model")
    end
    
    return ParMLEResult(
        model,
        slip,
        guess,
        se_slip,
        se_guess,
        pi_val,
        r_val,
        se_pi,
        se_r,
        Q,
        Y,
        alpha
    )
end
