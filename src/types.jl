# src/types.jl

abstract type AbstractCDMResult end

const AbstractCognitiveModel = AbstractCDMResult


"""
    AlphaNPResult

無參數認知診斷分類法 (NCDM / AlphaNP) 的結果結構。
"""
struct AlphaNPResult <: AbstractCDMResult
    alpha_est::Matrix{Int}          # 估計的 examinee 屬性組型 (I x K)
    est_ideal::Matrix{Int}          # 估計的理想作答反應 (I x J)
    est_class::Vector{Int}          # 估計的組型索引，對應到 pattern 的行數 (I)
    n_tie::Vector{Int}              # 平手次數 (I)
    pattern::Matrix{Int}            # 全體 2^K 種可能的二元屬性組型 (2^K x K)
    loss_matrix::Matrix{Float64}    # 損失矩陣 (2^K x I)
    method::String                  # 分類方法: "Hamming", "Weighted", "Penalized"
    gate::Symbol                    # 閘類型: :AND 或 :OR
    Q::Matrix{Int}                  # Q 矩陣 (J x K)
    Y::Matrix{Int}                  # 作答反應矩陣 (I x J)
end

"""
    NWCDMResult

無參數加權認知診斷分類法 (NWCDM) 的結果結構（源自於本論文的全新迭代方法）。
"""
struct NWCDMResult <: AbstractCDMResult
    alpha_est::Matrix{Int}          # 估計的 examinee 屬性組型 (I x K)
    est_class::Vector{Int}          # 估計的組型索引，對應到 pattern 的行數 (I)
    est_ideal::Matrix{Int}          # 理想作答反應 (I x J)
    weighted_centers::Matrix{Float64} # 加權中心矩陣 (2^K x J)
    slip_weights::Vector{Float64}   # 估計出的題目疏忽權重 (J)
    guess_weights::Vector{Float64}  # 估計出的題目猜測權重 (J)
    pattern::Matrix{Int}            # 全體 2^K 種可能的二元屬性組型 (2^K x K)
    iterations::Int                 # 收斂時迭代的次數
    Q::Matrix{Int}                  # Q 矩陣 (J x K)
    Y::Matrix{Int}                  # 作答反應矩陣 (I x J)
end

"""
    ParMLEResult

參數 MLE 估計題目參數的結果結構。
"""
struct ParMLEResult <: AbstractCDMResult
    model::Symbol                   # 模型類型: :DINA, :DINO, :NIDA, :GNIDA, :RRUM
    # 以下為 DINA, DINO, NIDA, GNIDA 專用
    slip::Any                       # Vector{Float64} (DINA, DINO, NIDA) 或 Matrix{Float64} (GNIDA)
    guess::Any                      # Vector{Float64} 或 Matrix{Float64}
    se_slip::Any                    # 標準誤
    se_guess::Any                   # 標準誤
    # 以下為 RRUM 專用
    pi::Any                         # Vector{Float64}
    r::Any                          # Matrix{Float64} (J x K)
    se_pi::Any                      # 標準誤
    se_r::Any                       # 標準誤
    
    Q::Matrix{Int}                  # Q 矩陣 (J x K)
    Y::Matrix{Int}                  # 作答反應矩陣 (I x J)
    alpha::Matrix{Int}              # Examinee 屬性組型 (I x K)
end

"""
    AlphaMLEResult

在給定題目參數下，估計 Examinee 屬性組型的 MLE 結果結構。
"""
struct AlphaMLEResult <: AbstractCDMResult
    alpha_est::Matrix{Int}          # 估計的屬性組型 (I x K)
    est_class::Vector{Int}          # 分類類別 (I)
    pattern::Matrix{Int}            # 全體 2^K 種可能的二元屬性組型 (2^K x K)
    n_tie::Vector{Int}              # 平手次數 (I)
    class_tie::Matrix{Int}          # 平手時紀錄的候選分類 (I x 2^K)
    loglike_matrix::Matrix{Float64} # 對數似然值矩陣 (2^K x I)
    Y::Matrix{Int}                  # 作答反應矩陣
    Q::Matrix{Int}                  # Q 矩陣
    par::Any                        # 傳入的題目參數 (ParMLEResult 或 Dict)
    model::Symbol                   # 模型類型
end

"""
    JMLEResult

聯合極大似然估計 (Joint MLE) 的結果結構。
"""
struct JMLEResult <: AbstractCDMResult
    alpha_est::Matrix{Int}          # 估計的屬性組型 (I x K)
    par_est::ParMLEResult           # 估計的題目參數
    n_tie::Vector{Int}              # 平手次數 (I)
    undefined_flag::Vector{Int}     # 標記題目參數估計是否為 undefined
    loglike::Float64                # 總對數似然值
    convergence::String             # 收斂狀態說明
    iterations::Int                 # 實際迭代次數
    aic::Float64                    # AIC
    bic::Float64                    # BIC
    loglike_matrix::Matrix{Float64} # 對數似然值矩陣 (2^K x I)
    est_class::Vector{Int}          # 分類類別 (I)
    
    # 初始無參數估計的資訊
    np_loss_matrix::Matrix{Float64}
    np_alpha_est::Matrix{Int}
    np_method::String
    np_est_class::Vector{Int}
    
    pattern::Matrix{Int}
    model::Symbol
    Q::Matrix{Int}
    Y::Matrix{Int}
end

"""
    QrefineResult

Q 矩陣修正結果的結構。
"""
struct QrefineResult <: AbstractCDMResult
    modified_Q::Matrix{Int}         # 修正後的 Q 矩陣 (J x K)
    modified_entries::Any           # 修正項目列表。無修改為 "NA"；否則為 Matrix{Int} (N x 2)，包含 [Item, Attribute]
    initial_class::Vector{Int}      # 初始受試者分類類別
    terminal_class::Vector{Int}     # 最終受試者分類類別
    patterns::Matrix{Int}           # 屬性組型空間
    initial_Q::Matrix{Int}          # 原始 Q 矩陣
    Y::Matrix{Int}                  # 作答反應矩陣
    gate::Symbol                    # 閘類型
end
