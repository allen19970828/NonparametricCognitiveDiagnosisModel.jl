# src/utils.jl

"""
    check_input(Y::Matrix{Int}, Q::Matrix{Int})

檢查作答反應矩陣 Y 與 Q 矩陣的維度與元素值是否符合二元認知診斷的要求。
"""
function check_input(Y::Matrix{Int}, Q::Matrix{Int})
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    if nitem != nitem_q
        error("作答反應矩陣 Y 中的試題數與 Q 矩陣中的試題數不一致！(Y: $nitem 題, Q: $nitem_q 題)")
    end
    
    # 檢查二元值
    for y in Y
        if y != 0 && y != 1
            error("作答反應矩陣 Y 應只包含二元值：1 = 答對，0 = 答錯。偵測到異常值: $y")
        end
    end
    
    for q in Q
        if q != 0 && q != 1
            error("Q 矩陣應只包含二元值：1 = 需要該屬性，0 = 不需要該屬性。偵測到異常值: $q")
        end
    end
    
    return nothing
end

"""
    alpha_permute(K::Int) -> Matrix{Int}

給定屬性個數 K，遞迴產生 2^K x K 的二元屬性組型矩陣。
其排序規則與 R 語言中的 `AlphaPermute` 保持完全一致：
- K = 1: [0; 1]
- K = 2: [0 0; 1 0; 0 1; 1 1]...
此順序為交替排列，便於結果與 R 版本進行數值一致性比對。
"""
function alpha_permute(K::Int)
    if K <= 0
        error("屬性個數 K 必須大於 0")
    end
    
    alpha = [0; 1]
    
    for i in 2:K
        # 複製原本的 alpha 矩陣
        alpha_dup = vcat(alpha, alpha)
        # 產生對應的新一列屬性
        new_col = vcat(zeros(Int, 2^(i-1)), ones(Int, 2^(i-1)))
        alpha = hcat(alpha_dup, new_col)
    end
    
    return alpha
end
