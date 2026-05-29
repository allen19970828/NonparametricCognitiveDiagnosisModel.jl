# src/fit.jl

using SpecialFunctions

"""
    item_fit(x::AbstractCognitiveModel) -> Matrix{Any}

針對無參數 (AlphaNP)、極大似然法 (AlphaMLE, ParMLE) 或聯合極大似然法 (JMLE) 的估計結果，
計算各個試題的適配度指標，包括 RMSEA、卡方值 (Chi-square)、p 值以及自由度 (df)。
"""
function item_fit(x::AbstractCognitiveModel)
    Q = x.Q
    Y = x.Y
    
    nitem, natt = size(Q)
    nperson = size(Y, 1)
    
    pattern = alpha_permute(natt)
    npattern = size(pattern, 1)
    
    # 1. 根據輸入結構的類型，提取對應的分類結果 est_class
    est_class = Vector{Int}(undef, nperson)
    if x isa AlphaNPResult || x isa AlphaMLEResult || x isa JMLEResult || x isa NWCDMResult
        est_class = x.est_class
    elseif x isa ParMLEResult
        # 若傳入的是題目估計結果，需利用屬性組型重新映射分類類別
        for i in 1:nperson
            # 尋找與 alpha[i, :] 完全一致的 pattern 索引
            match_idx = 1
            for m in 1:npattern
                if all(x.alpha[i, :] .== pattern[m, :])
                    match_idx = m
                    break
                end
            end
            est_class[i] = match_idx
        end
    end
    
    # 2. 提取模型類型與參數
    model = :DINA
    par = Dict{Symbol, Any}()
    
    if x isa AlphaMLEResult || x isa ParMLEResult || x isa JMLEResult
        model = x.model
    end
    
    if x isa AlphaMLEResult
        # 轉成 Dictionary
        par_input = x.par
        if par_input isa ParMLEResult
            if model in (:DINA, :DINO, :NIDA, :GNIDA)
                par[:slip] = par_input.slip
                par[:guess] = par_input.guess
            else
                par[:pi] = par_input.pi
                par[:r] = par_input.r
            end
        else
            par = par_input
        end
    elseif x isa ParMLEResult
        if model in (:DINA, :DINO, :NIDA, :GNIDA)
            par[:slip] = x.slip
            par[:guess] = x.guess
        else
            par[:pi] = x.pi
            par[:r] = x.r
        end
    elseif x isa JMLEResult
        par_input = x.par_est
        if model in (:DINA, :DINO, :NIDA, :GNIDA)
            par[:slip] = par_input.slip
            par[:guess] = par_input.guess
        else
            par[:pi] = par_input.pi
            par[:r] = par_input.r
        end
    elseif x isa NWCDMResult
        # NWCDM 本質上是 DINA-AND gate 的加權中心推廣，將其 s 與 g 轉換為參數形式
        model = :DINA
        par[:slip] = x.slip_weights
        par[:guess] = x.guess_weights
    elseif x isa AlphaNPResult
        # 標準 AlphaNP 無參數，使用預設 slip=0, guess=0 作為極端狀況
        model = :DINA
        par[:slip] = zeros(Float64, nitem)
        par[:guess] = zeros(Float64, nitem)
    end
    
    # 3. 計算卡方自由度 (df)
    df = 1
    if model in (:DINA, :DINO)
        df = npattern - 2
    elseif model in (:NIDA, :GNIDA)
        df = npattern - natt
    elseif model == :RRUM
        df = npattern - natt - 1
    end
    
    # 4. 計算各能力組型的頻率與比例
    class_freq = zeros(Float64, npattern)
    for i in 1:nperson
        class_freq[est_class[i]] += 1.0
    end
    class_prop = class_freq ./ nperson
    
    # 5. 計算各試題的適配度
    RMSEA = zeros(Float64, nitem)
    Chisq = zeros(Float64, nitem)
    Chisq_p = zeros(Float64, nitem)
    
    for j in 1:nitem
        # 提取第 j 題的參數
        par_j = Dict{Symbol, Any}()
        if model in (:DINA, :DINO)
            par_j[:slip] = [par[:slip][j]]
            par_j[:guess] = [par[:guess][j]]
        elseif model == :NIDA
            par_j[:slip] = par[:slip]
            par_j[:guess] = par[:guess]
        elseif model == :GNIDA
            par_j[:slip] = par[:slip][j, :]
            par_j[:guess] = par[:guess][j, :]
        elseif model == :RRUM
            par_j[:pi] = par[:pi][j]
            par_j[:r] = par[:r][j, :]
        end
        
        rmsea_tmp = 0.0
        chisq_val = 0.0
        
        for m in 1:npattern
            # 理想答對機率 E
            P = cdp(Q[j, :], par_j, pattern[m, :], model)
            
            # 觀測答對機率 O
            indices_in_class_m = findall(x -> x == m, est_class)
            P_obs = if isempty(indices_in_class_m)
                0.0
            else
                mean(Y[indices_in_class_m, j])
            end
            
            rmsea_tmp += (P - P_obs)^2 * class_prop[m]
            
            O = P_obs * class_freq[m]
            E = P * class_freq[m]
            N = class_freq[m]
            
            # 當 E 與 N-E 大於 0 時累加卡方項，確保數值精度安全
            if E > 1e-9 && (N - E) > 1e-9
                chisq_val += N * (O - E)^2 / (E * (N - E))
            end
        end
        
        RMSEA[j] = sqrt(rmsea_tmp)
        Chisq[j] = chisq_val
        
        # 使用 SpecialFunctions 中的正規化 incomplete gamma 函數精準計算卡方分佈的累積機率 (CDF)
        # p_val = 1.0 - CDF(Chisq, df)
        if chisq_val <= 0.0
            Chisq_p[j] = 1.0
        else
            Chisq_p[j] = 1.0 - gamma_inc(df / 2.0, chisq_val / 2.0)[1]
        end
    end
    
    # 6. 包裝為 Matrix{Any} 格式以對齊輸出
    out = Matrix{Any}(undef, nitem + 1, 5)
    out[1, :] = ["Item", "RMSEA", "Chisq", "Chisq p-value", "Chisq df"]
    for j in 1:nitem
        out[j + 1, 1] = "Item $j"
        out[j + 1, 2] = RMSEA[j]
        out[j + 1, 3] = Chisq[j]
        out[j + 1, 4] = Chisq_p[j]
        out[j + 1, 5] = df
    end
    
    return out
end

"""
    model_fit(x::AbstractCognitiveModel) -> Dict{Symbol, Float64}

針對聯合極大似然法 (JMLE) 或極大似然法題目參數估計 (ParMLE)，計算全模型的對數似然值 (-2LL)、AIC 與 BIC 等適配度指標。
"""
function model_fit(x::AbstractCognitiveModel)
    if x isa JMLEResult
        return Dict(:AIC => x.aic, :BIC => x.bic, :loglike => x.loglike)
    elseif x isa ParMLEResult
        Q = x.Q
        Y = x.Y
        nitem, natt = size(Q)
        nperson = size(Y, 1)
        pattern = alpha_permute(natt)
        npattern = size(pattern, 1)
        
        # 1. 映射分類類別
        est_class = Vector{Int}(undef, nperson)
        for i in 1:nperson
            match_idx = 1
            for m in 1:npattern
                if all(x.alpha[i, :] .== pattern[m, :])
                    match_idx = m
                    break
                end
            end
            est_class[i] = match_idx
        end
        
        # 2. 計算總對數似然值
        # 建立 undefined_flag
        undefined_flag = zeros(Int, nitem)
        if x.model in (:DINA, :DINO)
            for j in 1:nitem
                if isnan(x.slip[j]) || isinf(x.slip[j]) || isnan(x.guess[j]) || isinf(x.guess[j])
                    undefined_flag[j] = 1
                end
            end
        end
        
        # 轉換 par 為字典
        par = Dict{Symbol, Any}()
        if x.model in (:DINA, :DINO, :NIDA, :GNIDA)
            par[:slip] = x.slip
            par[:guess] = x.guess
        else
            par[:pi] = x.pi
            par[:r] = x.r
        end
        
        loglike = 0.0
        for i in 1:nperson
            loglike += cdl(Y[i, :], Q, par, pattern[est_class[i], :], x.model, undefined_flag)
        end
        
        # 3. 計算模型參數個數 npar
        npar = 0
        if x.model in (:DINA, :DINO)
            npar = 2 * nitem
        elseif x.model == :NIDA
            npar = 2 * natt
        elseif x.model == :GNIDA
            npar = 2 * natt * nitem
        elseif x.model == :RRUM
            npar = nitem * (natt + 1)
        end
        
        aic = -2 * loglike + 2 * npar
        bic = -2 * loglike + npar * log(nperson)
        
        return Dict(:AIC => aic, :BIC => bic, :loglike => loglike)
    else
        @warn "Model fit statistics are not appropriate for this class of object."
        return Dict(:AIC => NaN, :BIC => NaN, :loglike => NaN)
    end
end
