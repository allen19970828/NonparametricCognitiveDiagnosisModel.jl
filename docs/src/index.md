# NonparametricCognitiveDiagnosisModel.jl

[**繁體中文**](#繁體中文) | [**English**](#english)

---

## 繁體中文

歡迎來到 `NonparametricCognitiveDiagnosisModel.jl` 的官方文檔！此套件專為教育心理學、心理計量學與認知診斷（Cognitive Diagnosis Models, CDMs）開發，提供現代化且卓越的高效能 Julia 實作。

### 📌 功能模組
本套件深度涵蓋以下演算法與認知診斷方法：
- **無參數分類法 (NCDM / AlphaNP)**：漢明距離、加權漢明距離、懲罰漢明距離，並支援 DINA (`:AND` gate) 與 DINO (`:OR` gate)。
- **無參數加權分類法 (NWCDM)**：李政軒教授發表的全新專利學術成果。透過自適應更新學生的「猜測」與「疏忽」機率以建立連續型加權中心，解決了常規 NCDM 在小樣本與小班級下易受作答噪聲干擾的缺陷。
- **題目參數極大似然估計 (ParMLE)**：支援 DINA, DINO, NIDA, GNIDA 以及 RRUM 模型。使用盒狀約束（Box-constrained）極大似然優化算法，避免傳統方程求解器的數值發散，提供最高數值精度。
- **能力組型極大似然估計 (AlphaMLE)**：在已知題目參數下估計 Examinee 的能力屬性組型。
- **聯合極大似然估計 (JMLE)**：透過 ParMLE 與 AlphaMLE 交替迭代計算，估計全模型參數，提供 AIC 與 BIC 模型指標。
- **Q 矩陣修正優化 (Qrefine)**：透過交替最近鄰分類與 RSS 殘差平方和最小化，自動優化與修正 Q 矩陣中的不當屬性配置。
- **模型與試題適配指標 (ItemFit & ModelFit)**：計算精確的卡方 CDF p-value、RMSEA、AIC 以及 BIC 指標。

### 🚀 API 快速導覽
- `alpha_permute(K)`：生成 $2^K$ 種二元屬性空間。
- `alpha_np(Y, Q; gate=:AND, method=:Weighted)`：標準無參數分類。
- `nwcdm(Y, Q; max_iter=50)`：論文提出之迭代加權無參數分類。
- `par_mle(Y, Q, alpha, model)`：高精度題目參數 MLE 估計。
- `alpha_mle(Y, Q, par, model)`：學生能力組型 MLE 估計。
- `jmle(Y, Q; model=:DINA)`：交替聯合極大似然估計。
- `q_refine(Y, Q)`：Q 矩陣優化與修正。
- `item_fit(result)`：試題適配指標計算（RMSEA、卡方值、卡方 p-value）。
- `model_fit(result)`：模型適配度指標（AIC, BIC, Log-likelihood）。

---

## English

Welcome to the official documentation of `NonparametricCognitiveDiagnosisModel.jl`! This package is a modern, premium psychometrics toolkit developed for Cognitive Diagnostic Models (CDMs), offering exceptional mathematical precision and high-performance Julia implementations.

### 📌 Core Features
This toolkit covers:
- **Nonparametric Classification (NCDM / AlphaNP)**: Plain Hamming, Variance-Weighted, and Penalized distance metrics, supporting DINA (`:AND` gate) and DINO (`:OR` gate).
- **Nonparametric Weighted CDM (NWCDM)**: Full implementation of Prof. Cheng-Hsuan Li's innovative academic method. By adaptively estimating slipping and guessing weights of items, it constructs continuous weighted centers, resolving NCDM's diagnostic errors in noisy or small-class teaching scenarios.
- **Parametric Item MLE (ParMLE)**: Supports DINA, DINO, NIDA, GNIDA, and RRUM models. Bounded box-constrained likelihood optimization ensures supreme precision and mathematical stability.
- **Parametric Examinee MLE (AlphaMLE)**: Estimates examinee profiles given item parameters.
- **Joint MLE (JMLE)**: Alternating MLE steps for global model estimation and fit statistics (AIC, BIC).
- **Q-matrix Refinement (Qrefine)**: Adaptively optimizes Q-matrix entries by minimizing Residual Sum of Squares (RSS).
- **Model & Item Fit (ItemFit & ModelFit)**: Computes high-precision RMSEA, Chi-square p-values, AIC, and BIC.

### 🚀 Quick API Reference
- `alpha_permute(K)`: Generate all $2^K$ binary attribute configurations.
- `alpha_np(Y, Q; gate=:AND, method=:Weighted)`: Classic nonparametric CDM classification.
- `nwcdm(Y, Q; max_iter=50)`: Professor Li's iterative weighted centers classification.
- `par_mle(Y, Q, alpha, model)`: Box-constrained Maximum Likelihood parameter estimation.
- `alpha_mle(Y, Q, par, model)`: Examinee attribute profile MLE estimation.
- `jmle(Y, Q; model=:DINA)`: Iterative Joint MLE parameter estimation.
- `q_refine(Y, Q)`: Optimize and refine Q-matrix entries.
- `item_fit(result)`: Compute RMSEA, Chi-square value, card p-value, and degrees of freedom.
- `model_fit(result)`: Compute global fit indicators (AIC, BIC, Log-likelihood).
