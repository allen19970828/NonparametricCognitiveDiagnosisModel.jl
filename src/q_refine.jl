# src/q_refine.jl

"""
    q_refine(Y::Matrix{Int}, Q::Matrix{Int}; gate::Symbol=:AND, max_iter::Int=50) -> QrefineResult

Q 矩陣修正演算法 (Q-matrix Refinement)。
基於殘差平方和 (RSS) 最小化準則，透過最近鄰二元分類交替優化每道題目的 q 向量。
"""
function q_refine(Y::Matrix{Int}, Q::Matrix{Int}; gate::Symbol=:AND, max_iter::Int=50)
    check_input(Y, Q)
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    initial_Q = copy(Q)
    working_Q = copy(Q)
    
    M = 2^natt
    pattern = alpha_permute(natt)
    
    # 候選 q 向量集合 (排除全 0 向量，共 2^K - 1 種候選向量)
    pattern_Q = pattern[2:end, :] 
    num_candidates = size(pattern_Q, 1)
    
    initial_class = Int[]
    terminal_class = Int[]
    
    # 用於儲存各步驟的 RSS。每一列儲存：[rss_item_1, rss_item_2, ..., rss_item_J, total_rss]
    RSS_history = Matrix{Float64}(undef, 0, nitem + 1)
    
    for m in 1:max_iter
        @info "Qrefine Iteration: $m"
        
        max_rss_item = Int[]
        
        for k in 1:nitem
            # 1. 使用 Hamming 距離的 AlphaNP 對受試者進行最新分類
            classification = alpha_np(Y, working_Q; gate=gate, method=:Hamming, wg=1.0, ws=1.0)
            est_class = classification.est_class
            est_ideal = classification.est_ideal
            
            # 紀錄初始分類結果
            if m == 1 && k == 1
                initial_class = copy(est_class)
            end
            
            # 2. 計算目前各題目的 RSS
            diff = Y .- est_ideal
            rss = sum(diff .^ 2, dims=1) |> vec
            total_rss = sum(rss)
            
            # 記錄歷史 RSS
            row_rss = vcat(rss, total_rss)'
            RSS_history = vcat(RSS_history, row_rss)
            
            # 3. 挑選目前 RSS 最大且尚未在此迭代中被修正的題目
            max_rss = 1
            if k == 1
                max_rss = argmax(rss)
            else
                # 排除已經被修正的題目
                rss_unvisited = Float64.(rss)
                for idx in max_rss_item
                    rss_unvisited[idx] = -Inf
                end
                max_rss = argmax(rss_unvisited)
            end
            push!(max_rss_item, max_rss)
            
            # 4. 在 2^K - 1 個候選 q 向量中進行窮舉搜索，尋找能使該題 RSS 最小的 q 向量
            update_rss = Vector{Float64}(undef, num_candidates)
            for cand_idx in 1:num_candidates
                q_cand = pattern_Q[cand_idx, :]
                
                # 計算全體 examinee 在此候選 q 向量下的理想作答
                u = Vector{Int}(undef, nperson)
                for i in 1:nperson
                    profile = pattern[est_class[i], :]
                    if gate == :AND
                        u[i] = prod(profile .^ q_cand)
                    else # :OR
                        u[i] = 1 - prod((1 .- profile) .^ q_cand)
                    end
                end
                
                # 計算該候選向量與 Y 觀測值的殘差平方和
                update_rss[cand_idx] = sum((Y[:, max_rss] .- u) .^ 2)
            end
            
            # 尋找最小 RSS 的候選向量
            min_cand_val = minimum(update_rss)
            min_cand_indices = findall(x -> abs(x - min_cand_val) < 1e-9, update_rss)
            
            best_cand_idx = length(min_cand_indices) > 1 ? rand(min_cand_indices) : min_cand_indices[1]
            update_q = pattern_Q[best_cand_idx, :]
            
            # 更新 Q 矩陣中的對應題目
            working_Q[max_rss, :] = update_q
        end
        
        # 5. 檢查停止條件：如果在一個完整迭代（nitem 次調整）中，總 RSS 未發生任何改變，則代表完全收斂
        start_row = (m - 1) * nitem + 1
        end_row = m * nitem
        rss_totals = RSS_history[start_row:end_row, nitem + 1]
        
        if sum(abs.(rss_totals .- rss_totals[1])) < 1e-9
            classification_final = alpha_np(Y, working_Q; gate=gate, method=:Hamming, wg=1.0, ws=1.0)
            terminal_class = classification_final.est_class
            break
        end
        
        # 若是最後一次迭代，也需要更新 terminal_class
        if m == max_iter
            classification_final = alpha_np(Y, working_Q; gate=gate, method=:Hamming, wg=1.0, ws=1.0)
            terminal_class = classification_final.est_class
        end
    end
    
    # 6. 計算被修改的 entries
    # 判斷有無修改：若兩矩陣完全相同，回傳 "NA"，否則回傳包含 [Item, Attribute] 的二維矩陣
    diff_Q = (working_Q .- initial_Q) .!= 0
    modified = "NA"
    if sum(diff_Q) > 0
        indices = findall(diff_Q)
        mod_matrix = Matrix{Int}(undef, length(indices), 2)
        for idx in 1:length(indices)
            mod_matrix[idx, 1] = indices[idx][1] # Item
            mod_matrix[idx, 2] = indices[idx][2] # Attribute
        end
        modified = mod_matrix
    end
    
    return QrefineResult(
        working_Q,
        modified,
        initial_class,
        terminal_class,
        pattern,
        initial_Q,
        Y,
        gate
    )
end
