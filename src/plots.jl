# src/plots.jl

using RecipesBase

# Helper function to convert binary vector to string like "010"
function profile_to_string(vec::AbstractVector)
    return join(string.(vec))
end

"""
    Plot recipe for AlphaNPResult
    用法: plot(result, nperson=1)
"""
@recipe function f(res::AlphaNPResult; nperson=1)
    loss = res.loss_matrix[:, nperson]
    perm = sortperm(loss)
    sorted_loss = loss[perm]
    
    # 建立 X 軸的標籤 (能力組型字串)
    npattern = size(res.pattern, 1)
    labels = [profile_to_string(res.pattern[i, :]) for i in 1:npattern]
    sorted_labels = labels[perm]
    
    # 決定顏色或填色樣式，將估計的組型著色 highlight
    est_m = res.est_class[nperson]
    colors = Symbol[]
    for idx in perm
        if idx == est_m
            push!(colors, :crimson) # Highlight
        else
            push!(colors, :steelblue)
        end
    end
    
    # 圖表設定
    title --> "Loss Function of Examinee $nperson (NCDM: $(res.method))"
    xguide --> "Attribute Profile"
    yguide --> "Loss"
    seriestype --> :bar
    legend --> :topleft
    label --> "Candidate Profile"
    xrotation --> 45
    fillcolor --> colors
    
    sorted_labels, sorted_loss
end

"""
    Plot recipe for AlphaMLEResult
    用法: plot(result, nperson=1)
"""
@recipe function f(res::AlphaMLEResult; nperson=1)
    nll = -res.loglike_matrix[:, nperson]
    perm = sortperm(nll)
    sorted_nll = nll[perm]
    
    npattern = size(res.pattern, 1)
    labels = [profile_to_string(res.pattern[i, :]) for i in 1:npattern]
    sorted_labels = labels[perm]
    
    est_m = res.est_class[nperson]
    colors = Symbol[]
    for idx in perm
        if idx == est_m
            push!(colors, :forestgreen) # Highlight
        else
            push!(colors, :steelblue)
        end
    end
    
    title --> "Negative Log-Likelihood of Examinee $nperson (MLE)"
    xguide --> "Attribute Profile"
    yguide --> "-Log-Likelihood"
    seriestype --> :bar
    legend --> :topleft
    label --> "Candidate Profile"
    xrotation --> 45
    fillcolor --> colors
    
    sorted_labels, sorted_nll
end

"""
    Plot recipe for JMLEResult
    用法: plot(result, nperson=1)
    會自動切分為上下兩個面板子圖：
    1. 初始無參數估計的 Loss
    2. JMLE 的 Negative Log-Likelihood
"""
@recipe function f(res::JMLEResult; nperson=1)
    # 版面設定為 2x1 面板
    layout --> (2, 1)
    size --> (800, 600)
    
    npattern = size(res.pattern, 1)
    labels = [profile_to_string(res.pattern[i, :]) for i in 1:npattern]
    
    # 1. 第一子圖 (Loss)
    @series begin
        subplot := 1
        loss = res.np_loss_matrix[:, nperson]
        est_m_np = res.np_est_class[nperson]
        colors_np = [i == est_m_np ? :crimson : :steelblue for i in 1:npattern]
        
        title := "Initial Loss Function of Examinee $nperson (NCDM: $(res.np_method))"
        xguide := ""
        yguide := "Loss"
        seriestype := :bar
        xrotation := 45
        fillcolor := colors_np
        label := "NP Candidate Profile"
        
        labels, loss
    end
    
    # 2. 第二子圖 (Negative Log-Likelihood)
    @series begin
        subplot := 2
        nll = -res.loglike_matrix[:, nperson]
        est_m_jmle = res.est_class[nperson]
        colors_jmle = [i == est_m_jmle ? :forestgreen : :steelblue for i in 1:npattern]
        
        title := "Negative Log-Likelihood of Examinee $nperson (JMLE)"
        xguide := "Attribute Profile"
        yguide := "-Log-Likelihood"
        seriestype := :bar
        xrotation := 45
        fillcolor := colors_jmle
        label := "JMLE Candidate Profile"
        
        labels, nll
    end
end

"""
    Plot recipe for QrefineResult
    用法: plot(result, item_idx=1)
    顯示給定題目在各個候選 q 向量下的 RSS 值，並 highlight 原始 q 與修正後 q
"""
@recipe function f(res::QrefineResult; item_idx=1)
    # 取得屬性組型 (排除全 0)
    num_candidates = size(res.patterns, 1) - 1
    pattern_Q = res.patterns[2:end, :]
    
    # 重算該題在所有 2^K - 1 種 q 向量下的 RSS
    nperson = size(res.Y, 1)
    RSS = Vector{Float64}(undef, num_candidates)
    
    for m in 1:num_candidates
        q_cand = pattern_Q[m, :]
        u = Vector{Int}(undef, nperson)
        for i in 1:nperson
            profile = res.patterns[res.terminal_class[i], :]
            if res.gate == :AND
                u[i] = prod(profile .^ q_cand)
            else
                u[i] = 1 - prod((1 .- profile) .^ q_cand)
            end
        end
        RSS[m] = sum((res.Y[:, item_idx] .- u) .^ 2)
    end
    
    perm = sortperm(RSS)
    sorted_rss = RSS[perm]
    
    labels = [profile_to_string(pattern_Q[m, :]) for m in 1:num_candidates]
    sorted_labels = labels[perm]
    
    # Highlight 顏色與標籤
    # 尋找原始與修正後的 q 向量在 pattern_Q 中的索引
    init_q = res.initial_Q[item_idx, :]
    final_q = res.modified_Q[item_idx, :]
    
    colors = Symbol[]
    for idx in perm
        q_cand = pattern_Q[idx, :]
        if all(q_cand .== final_q)
            push!(colors, :forestgreen) # 修正後
        elseif all(q_cand .== init_q)
            push!(colors, :crimson)     # 原始
        else
            push!(colors, :steelblue)   # 其他
        end
    end
    
    title --> "RSS of Candidate Q-vectors for Item $item_idx"
    xguide --> "Q-Vector"
    yguide --> "RSS"
    seriestype --> :bar
    xrotation --> 45
    fillcolor --> colors
    legend --> :topright
    label --> "Candidate Q-vector (Red: Initial, Green: Refined)"
    
    sorted_labels, sorted_rss
end
