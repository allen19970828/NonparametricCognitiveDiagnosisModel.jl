# document.jl
# ==============================================================================
# NonparametricCognitiveDiagnosisModel.jl - Bilingual Documentation & Reference
# 雙語學術文檔與套件 API 參考指南
# ==============================================================================
# 本檔案旨在為 NonparametricCognitiveDiagnosisModel.jl 提供完整、嚴謹且易讀的
# 學術理論公式與程式碼 API 對照。涵蓋標準無參數方法、李政軒教授的加權中心無參數方法 (NWCDM)、
# 以及 DINA, DINO, NIDA, GNIDA, RRUM 的極大似然估計 (MLE) 與聯合估計 (JMLE)。
#
# This file provides a comprehensive and mathematically rigorous bilingual reference
# for NonparametricCognitiveDiagnosisModel.jl, detailing NCDM, Prof. Li's NWCDM,
# and MLE/JMLE implementations for DINA, DINO, NIDA, GNIDA, and RRUM.
# ==============================================================================

"""
# 第一部分：認知診斷模型學術理論與數學公式
# Part I: Mathematical & Theoretical Foundations of CDMs

## 1. 基礎符號定義 (Basic Notations)
- $I$: 受試者人數 (Number of Examinees), 索引以 $i = 1, \dots, I$ 表示。
- $J$: 試題總數 (Number of Items), 索引以 $j = 1, \dots, J$ 表示。
- $K$: 能力屬性總數 (Number of Attributes), 索引以 $k = 1, \dots, K$ 表示。
- $Y$: 作答反應矩陣 (Response Matrix) $I \times J$, 元素 $y_{ij} \in \{0, 1\}$ 表示對錯。
- $Q$: 概念關聯矩陣 (Q-matrix) $J \times K$, 元素 $q_{jk} \in \{0, 1\}$ 表示試題 $j$ 是否要求屬性 $k$。
- $\boldsymbol{\alpha}_i$: 受試者 $i$ 的屬性組型 (Attribute Profile) $1 \times K$ 向量, $\alpha_{ik} \in \{0, 1\}$ 表示是否精熟該屬性。
- $\bar{\boldsymbol{\alpha}}_\ell$: 能力屬性空間中第 $\ell$ 種可能的二元組型, 共 $M = 2^K$ 種。

## 2. 理想作答反應 (Ideal Responses)
在無噪聲的理想狀態下，受試者在給定屬性組型 $\boldsymbol{\alpha}$ 下對試題 $j$ 的答對反應 $\eta_{j}(\boldsymbol{\alpha})$ 依閘類型 (Gate) 定義：
- **AND Gate (非補償性，如 DINA, NIDA, GNIDA, RRUM)**: 受試者必須精熟試題要求的所有屬性才能答對。
  $$\eta_{j}(\boldsymbol{\alpha}) = \prod_{k=1}^K \alpha_k^{q_{jk}}$$
- **OR Gate (補償性，如 DINO)**: 受試者只需具備試題要求的任一屬性即可答對。
  $$\eta_{j}(\boldsymbol{\alpha}) = 1 - \prod_{k=1}^K (1 - \alpha_k)^{q_{jk}}$$

---

## 3. 無參數認知診斷分類法 (NCDM / AlphaNP)
無參數方法（AlphaNP）不需要複雜的馬可夫鏈蒙地卡羅 (MCMC) 或期望值極大化 (EM) 參數估計，適合小樣本。
主要思想是尋找使受試者觀測反應 $\boldsymbol{y}_i$ 與理想反應 $\boldsymbol{\eta}(\bar{\boldsymbol{\alpha}}_\ell)$ 距離最小的屬性組型。

- **損失函數 (Loss Function)**:
  $$L_\ell(\boldsymbol{y}_i) = \sum_{j=1}^J w_j \left[ w_g |y_{ij} - \eta_j(\bar{\boldsymbol{\alpha}}_\ell)| y_{ij} + w_s |y_{ij} - \eta_j(\bar{\boldsymbol{\alpha}}_\ell)| (1 - y_{ij}) \right]$$
  1. **Hamming 距離**: $w_j = 1$, $w_g = w_s = 1$。
  2. **加權距離 (Weighted)**: $w_j = \frac{1}{\bar{p}_j(1 - \bar{p}_j)}$（$\bar{p}_j$ 為題目 $j$ 的樣本平均答對率），$w_g = w_s = 1$。
  3. **懲罰距離 (Penalized)**: $w_g$（猜測懲罰）與 $w_s$（疏忽懲罰）不相等。

---

## 4. 李政軒教授提出之《無參數加權認知診斷模式》 (NWCDM)
標準 NCDM 未考慮個別學生的猜測與疏忽，容易受噪聲干擾而誤判。
NWCDM 藉由引進「疏忽權重」與「猜測權重」，將二元理想反應轉換為連續型的**加權中心 (Weighted Centers)**：

### 核心演算法步驟 (Iterative NWCDM Algorithm)
1. **初始化 (Initialization)**:
   - 令所有試題的疏忽權重 $s_j^w = 0$，猜測權重 $g_j^w = 0$。
2. **計算加權中心 (Weighted Centers)**:
   - 對於每一種可能的屬性組型 $\bar{\boldsymbol{\alpha}}_\ell$ ($\ell = 1, \dots, 2^K$)，其在試題 $j$ 的加權中心 $c_{\ell j}$ 定義為：
     $$c_{\ell j} = (1 - s_j^w)^{\eta_j(\bar{\boldsymbol{\alpha}}_\ell)} (g_j^w)^{1 - \eta_j(\bar{\boldsymbol{\alpha}}_\ell)}$$
     *(註：當 $s_j^w=0, g_j^w=0$ 時，$c_{\ell j}$ 會退化為二元理想反應 $\eta_j$，即退化為常規 NCDM)*
3. **最近鄰分類 (Nearest Neighbor Classification)**:
   - 計算受試者與加權中心之 L1 距離，並重新分類能力組型：
     $$\boldsymbol{\alpha}_i = \bar{\boldsymbol{\alpha}}_{\hat{\ell}} \quad \text{其中} \quad \hat{\ell} = \arg\min_{\ell \in \{1,\dots,2^K\}} \sum_{j=1}^J |y_{ij} - c_{\ell j}|$$
4. **重新估計疏忽與猜測權重 (Weight Update)**:
   - 根據最新分類得到的受試者理想作答 $\eta_{ij}$，更新權重：
     $$s_j^w = \frac{\sum_{i=1}^I (1 - y_{ij}) \eta_{ij}}{\sum_{i=1}^I \eta_{ij}}$$
     $$g_j^w = \frac{\sum_{i=1}^I y_{ij} (1 - \eta_{ij})}{\sum_{i=1}^I (1 - \eta_{ij})}$$
5. **收斂判定 (Convergence Check)**:
   - 重複步驟 2~4，直到全體受試者的分類結果 $\boldsymbol{\alpha}_i$ 在兩次迭代間不再改變為止。

---

## 5. 參數化認知診斷模型之似然度定義 (Likelihoods of Parametric CDMs)
在極大似然估計 (MLE) 下，受試者答對試題 $j$ 的機率 $P_j(\boldsymbol{\alpha})$ 定義如下：

- **DINA Model (Deterministic Input, Noisy "And" Gate)**:
  - 參數：試題級 $s_j$ (slip), $g_j$ (guess)
  - 機率：$P_j(\boldsymbol{\alpha}) = (1 - s_j)^{\eta_j} g_j^{1 - \eta_j}$
- **DINO Model (Deterministic Input, Noisy "Or" Gate)**:
  - 參數：試題級 $s_j$ (slip), $g_j$ (guess)
  - 機率：$P_j(\boldsymbol{\alpha}) = (1 - s_j)^{\omega_j} g_j^{1 - \omega_j}$，其中 $\omega_j = 1 - \prod (1 - \alpha_k)^{q_{jk}}$
- **NIDA Model (Noisy Inputs, Deterministic "And" Gate)**:
  - 參數：屬性級 $s_k$ (slip), $g_k$ (guess)
  - 機率：$P_j(\boldsymbol{\alpha}) = \prod_{k=1}^K [ (1 - s_k)^{\alpha_k} g_k^{1 - \alpha_k} ]^{q_{jk}}$
- **GNIDA Model (Generalized NIDA)**:
  - 參數：試題-屬性級 $s_{jk}$ (slip), $g_{jk}$ (guess)
  - 機率：$P_j(\boldsymbol{\alpha}) = \prod_{k=1}^K [ (1 - s_{jk})^{\alpha_k} g_{jk}^{1 - \alpha_k} ]^{q_{jk}}$
- **RRUM Model (Reduced Reparameterized Unified Model)**:
  - 參數：試題級 $\pi_j$ (精熟者答對率), 屬性折損因子 $r_{jk}$
  - 機率：$P_j(\boldsymbol{\alpha}) = \pi_j \prod_{k=1}^K r_{jk}^{q_{jk}(1 - \alpha_k)}$
"""

# ==============================================================================
# 第二部分：API 介面說明與 Julia 程式範例
# Part II: API Documentation & Practical Examples
# ==============================================================================

using NonparametricCognitiveDiagnosisModel

# ------------------------------------------------------------------------------
# 1. 組型生成與驗證 (Pattern Generation & Verification)
# ------------------------------------------------------------------------------

# 產生 K = 4 的二元屬性空間 (共 16 種可能的組態，大小為 16 x 4)
# Generate attribute space of K = 4 (16 x 4 matrix)
K = 4
pattern_space = alpha_permute(K)

# 準備作答資料 Y 與 Q 矩陣
# Preparing response matrix Y (5 examinees x 7 items) and Q-matrix (7 items x 4 attributes)
Y = [
    1 1 0 1 0 1 0;
    1 0 0 0 1 0 1;
    0 1 1 0 0 0 0;
    1 1 1 1 1 1 1;
    0 0 1 0 0 1 0
]

Q = [
    1 1 0 1;
    0 1 1 0;
    0 1 0 1;
    1 1 0 1;
    1 1 1 1;
    1 1 0 1;
    0 1 1 1
]

# 檢查 Y 與 Q 的維度與內容二元一致性
# Validate consistency of Y and Q
check_input(Y, Q)

# ------------------------------------------------------------------------------
# 2. 無參數分類估計 (Nonparametric Classifications)
# ------------------------------------------------------------------------------

# 範例 A: 標準 NCDM (Weighted Hamming 距離，AND 閘)
# Example A: Standard NCDM (Weighted Hamming distance, AND gate)
ncdm_result = alpha_np(Y, Q; gate=:AND, method=:Weighted)

# 範例 B: 論文加權迭代 NWCDM 分類法 (小樣本極力推薦，精度最高)
# Example B: Prof. Li's NWCDM iterative method (Highly recommended for small samples)
nwcdm_result = nwcdm(Y, Q; max_iter=50)

# 透過 dispatch show 輸出結果報告
# Automatically outputs detailed structural reports in terminal
println(ncdm_result)
println(nwcdm_result)

# ------------------------------------------------------------------------------
# 3. 參數極大似然估計 (Parametric MLE)
# ------------------------------------------------------------------------------

# 假設受試者的能力組型已知 (I x K)
# Assuming examinees' profiles are known
assumed_alpha = [
    1 1 0 1;
    0 1 1 0;
    0 1 0 0;
    1 1 1 1;
    0 0 1 1
]

# 估計 GNIDA 模型的題目參數。
# 本套件採用 Optim.jl 盒狀約束優化求解，避免了極端機率 NaN/Inf 產生，數值精度冠絕同類套件。
# Fit GNIDA parameters. Optim.jl box-constrained solver ensures values strictly in [0, 1].
gnida_params = par_mle(Y, Q, assumed_alpha, :GNIDA)
println(gnida_params)

# 在給定題目參數下，使用 MLE 估計 Examinee 的能力屬性組型
# Estimate examinees' profiles under given item parameters
alpha_est_mle = alpha_mle(Y, Q, gnida_params, :GNIDA)
println(alpha_est_mle)

# ------------------------------------------------------------------------------
# 4. 聯合極大似然估計 (Joint MLE / JMLE)
# ------------------------------------------------------------------------------

# 交替迭代 ParMLE 與 AlphaMLE，直至題目與學生屬性估計完全收斂
# Alternating iterations of ParMLE and AlphaMLE until global convergence
jmle_result = jmle(Y, Q; model=:DINA, np_method=:Weighted, max_iter=100)
println(jmle_result)

# ------------------------------------------------------------------------------
# 5. Q 矩陣修正與適配度檢定 (Q-matrix Refinement & Fit Statistics)
# ------------------------------------------------------------------------------

# 優化修正 Q 矩陣中適配度差（RSS 最大）的試題屬性配置
# Iteratively optimize and refine Q-matrix entries by minimizing Residual Sum of Squares (RSS)
q_refine_result = q_refine(Y, Q; gate=:AND, max_iter=20)
println(q_refine_result)

# 計算各試題的適配度 (RMSEA, 卡方值, 卡方 p-value 與自由度 df)
# Calculate item fit indices using high-precision SpecialFunctions card CDF
item_fit_table = item_fit(jmle_result)
println("\n--- Item Fit Statistics ---")
for r in 1:size(item_fit_table, 1)
    println(join(rpad.(string.(item_fit_table[r, :]), 18)))
end

# 計算模型的 AIC 與 BIC
# Calculate overall model fit (AIC, BIC, Log-likelihood)
model_fit_stats = model_fit(jmle_result)
println("\nModel Fit Statistics: AIC = $(model_fit_stats[:AIC]), BIC = $(model_fit_stats[:BIC])")

# ------------------------------------------------------------------------------
# 6. 可視化診斷繪圖 (Diagnostic Plot Recipes)
# ------------------------------------------------------------------------------
# 若您安裝了 Plots.jl，本套件提供的 RecipesBase 配方允許您直接繪製精美診斷圖：
# If you have Plots.jl installed, you can easily plot beautiful diagnostic charts:
#
# using Plots
#
# # 繪製受試者 1 的無參數損失函數柱狀圖 (Highlight 估計值)
# plot(ncdm_result, nperson=1)
#
# # 繪製受試者 2 的極大似然對數似然值柱狀圖
# plot(alpha_est_mle, nperson=2)
#
# # 一次繪製受試者 3 在 JMLE 中對比無參數與 MLE 的雙面板子圖
# plot(jmle_result, nperson=3)
#
# # 繪製試題 2 在不同候選 q 向量下的 RSS 修正圖 (Highlight 原始與修正後的 q)
# plot(q_refine_result, item_idx=2)
# ==============================================================================
