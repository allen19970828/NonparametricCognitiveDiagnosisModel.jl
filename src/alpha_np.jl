# src/alpha_np.jl

"""
    alpha_np(Y::Matrix{Int}, Q::Matrix{Int}; gate::Symbol=:AND, method::Symbol=:Hamming, wg::Real=1.0, ws::Real=1.0) -> AlphaNPResult

標準無參數認知診斷分類法 (NCDM / AlphaNP)。

# 參數說明
* `Y`: 作答反應矩陣 (I x J)
* `Q`: Q 矩陣 (J x K)
* `gate`: `:AND` (DINA 閘) 或 `:OR` (DINO 閘)
* `method`: `:Hamming` (平坦漢明距離), `:Weighted` (以題目變異數倒數為權重), `:Penalized` (考慮 guess 與 slip 的懲罰漢明距離)
* `wg`: 猜測 (guess) 的懲罰權重
* `ws`: 疏忽 (slip) 的懲罰權重
"""
function alpha_np(Y::Matrix{Int}, Q::Matrix{Int}; gate::Symbol=:AND, method::Symbol=:Hamming, wg::Real=1.0, ws::Real=1.0)
    # 1. 檢查輸入
    check_input(Y, Q)
    
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    M = 2^natt
    pattern = alpha_permute(natt)
    
    # 2. 計算全體 2^K 種能力組型對所有試題的二元理想作答反應 Ideal (M x J)
    Ideal = Matrix{Int}(undef, M, nitem)
    for m in 1:M
        for j in 1:nitem
            if gate == :AND
                Ideal[m, j] = prod(pattern[m, :] .^ Q[j, :])
            elseif gate == :OR
                Ideal[m, j] = 1 - prod((1 .- pattern[m, :]) .^ Q[j, :])
            else
                error("gate 參數必須是 :AND 或 :OR")
            end
        end
    end
    
    # 3. 計算題目權重 (Weight)
    weight = ones(Float64, nitem)
    if method == :Weighted || method == :Penalized
        p_bar = mean(Y, dims=1) |> vec
        weight = 1.0 ./ (p_bar .* (1.0 .- p_bar))
        
        # 避免分母為零，將權重限制在 1 / (0.95 * 0.05) 之內
        cap = 1.0 / (0.95 * 0.05)
        for j in 1:nitem
            if weight[j] > cap || isnan(weight[j]) || isinf(weight[j])
                weight[j] = cap
            end
        end
    end
    
    if method == :Penalized && ws == wg
        @warn "Penalizing weights for guess and slip are the same --> equivalent with the \"Weighted\" method."
    end
    
    # 4. 計算各能力組型對於每個受試者的損失
    loss_matrix = Matrix{Float64}(undef, M, nperson)
    est_class = Vector{Int}(undef, nperson)
    n_tie = zeros(Int, nperson)
    
    for i in 1:nperson
        yi = Y[i, :]
        for m in 1:M
            ideal_m = Ideal[m, :]
            
            # 漢明距離差值
            diff = abs.(yi .- ideal_m)
            
            # 若為 penalized，要分別考慮 guess (yi=1, ideal=0) 與 slip (yi=0, ideal=1) 的權重
            # 否則 ws_val, wg_val 為 1.0
            if method == :Penalized
                term = wg * diff .* yi .+ ws * diff .* (1 .- yi)
                loss_matrix[m, i] = sum(weight .* term)
            else
                # Hamming 或 Weighted
                loss_matrix[m, i] = sum(weight .* diff)
            end
        end
        
        # 尋找損失最小的組型
        min_loss = minimum(loss_matrix[:, i])
        min_indices = findall(x -> x == min_loss, loss_matrix[:, i])
        
        if length(min_indices) > 1
            n_tie[i] = length(min_indices)
            # 遭遇多個最優解平手時，以均勻機率隨機抽取一個分類類別
            est_class[i] = rand(min_indices)
        else
            est_class[i] = min_indices[1]
        end
    end
    
    alpha_est = pattern[est_class, :]
    est_ideal = Ideal[est_class, :]
    
    return AlphaNPResult(
        alpha_est,
        est_ideal,
        est_class,
        n_tie,
        pattern,
        loss_matrix,
        string(method),
        gate,
        Q,
        Y
    )
end
