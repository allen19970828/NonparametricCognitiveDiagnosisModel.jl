# src/nwcdm.jl

"""
    nwcdm(Y::Matrix{Int}, Q::Matrix{Int}; max_iter::Int=50, tol::Float64=1e-6) -> NWCDMResult

實作論文《無參數加權認知診斷模式》中的加權中心迭代分類演算法 (NWCDM)。
此演算法利用「疏忽權重」與「猜測權重」計算連續型的加權中心，並通過 L1 距離進行受試者認知反應組型的分類。

# 參數說明
* `Y`: 作答反應矩陣 (I x J)
* `Q`: Q 矩陣 (J x K)
* `max_iter`: 最大迭代次數
* `tol`: 收斂容忍度 (本演算法是以受試者分類是否改變為停止指標，因此當無受試者分類變動時即宣告收斂)
"""
function nwcdm(Y::Matrix{Int}, Q::Matrix{Int}; max_iter::Int=50, tol::Float64=1e-6)
    # 1. 檢查輸入維度與正確性
    check_input(Y, Q)
    
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    M = 2^natt
    pattern = alpha_permute(natt)
    
    # 2. 預先計算 DINA 理想作答反應 Eta (M x J)
    # 論文中主要以 DINA (AND gate) 作為理想反應基礎
    Eta = Matrix{Int}(undef, M, nitem)
    for m in 1:M
        for j in 1:nitem
            Eta[m, j] = prod(pattern[m, :] .^ Q[j, :])
        end
    end
    
    # 3. 初始化權重
    s = zeros(Float64, nitem)   # 題目疏忽權重 (Slip weights)
    g = zeros(Float64, nitem)   # 題目猜測權重 (Guess weights)
    
    est_class = zeros(Int, nperson)
    est_class_old = copy(est_class)
    
    iter_count = 0
    converged = false
    
    # 4. 迭代更新加權中心與分類
    while iter_count < max_iter && !converged
        iter_count += 1
        
        # 4a. 計算 2^K 個加權中心 (M x J)
        C = Matrix{Float64}(undef, M, nitem)
        for m in 1:M
            for j in 1:nitem
                C[m, j] = ((1.0 - s[j])^Eta[m, j]) * (g[j]^(1.0 - Eta[m, j]))
            end
        end
        
        # 4b. 受試者分類：計算每個 examinee 與 C 的 L1 距離
        for i in 1:nperson
            yi = Y[i, :]
            best_dist = Inf
            best_idx = 1
            candidates = Int[]
            
            for m in 1:M
                # 計算 L1 距離 (相當於漢明距離的實數推廣)
                dist = sum(abs.(yi .- C[m, :]))
                if dist < best_dist
                    best_dist = dist
                    best_idx = m
                    empty!(candidates)
                    push!(candidates, m)
                elseif abs(dist - best_dist) < 1e-9
                    push!(candidates, m)
                end
            end
            
            # 若有多個最優解平手，隨機挑選一個分類
            est_class[i] = length(candidates) > 1 ? rand(candidates) : best_idx
        end
        
        # 4c. 收斂條件檢查：若全體受試者的分類與上一次相比完全沒有變動，即代表收斂
        if est_class == est_class_old
            converged = true
            break
        end
        copyto!(est_class_old, est_class)
        
        # 4d. 依據新的分類結果，重新估計各題目的疏忽權重與猜測權重
        for j in 1:nitem
            sum_eta = 0.0
            sum_slip = 0.0
            sum_not_eta = 0.0
            sum_guess = 0.0
            
            for i in 1:nperson
                eta_ij = Eta[est_class[i], j]
                x_ij = Y[i, j]
                
                if eta_ij == 1
                    sum_eta += 1.0
                    sum_slip += (1.0 - x_ij)
                else
                    sum_not_eta += 1.0
                    sum_guess += x_ij
                end
            end
            
            # 計算更新權重（加入極小值 1e-6 避免分母為 0 造成 NaN）
            s[j] = sum_eta > 0.0 ? sum_slip / sum_eta : 0.0
            g[j] = sum_not_eta > 0.0 ? sum_guess / sum_not_eta : 0.0
            
            # 將數值限縮在合理的區間內 [1e-4, 1.0 - 1e-4] 確保數值精度穩定
            s[j] = clamp(s[j], 1e-4, 1.0 - 1e-4)
            g[j] = clamp(g[j], 1e-4, 1.0 - 1e-4)
        end
    end
    
    alpha_est = pattern[est_class, :]
    est_ideal = Eta[est_class, :]
    
    # 重新計算最終的加權中心矩陣 C (2^K x J)
    C_final = Matrix{Float64}(undef, M, nitem)
    for m in 1:M
        for j in 1:nitem
            C_final[m, j] = ((1.0 - s[j])^Eta[m, j]) * (g[j]^(1.0 - Eta[m, j]))
        end
    end
    
    return NWCDMResult(
        alpha_est,
        est_class,
        est_ideal,
        C_final,
        s,
        g,
        pattern,
        iter_count,
        Q,
        Y
    )
end
