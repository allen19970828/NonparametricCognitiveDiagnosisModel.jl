# src/jmle.jl

"""
    jmle(Y::Matrix{Int}, Q::Matrix{Int}; model::Symbol=:DINA, np_method::Symbol=:Weighted, conv_crit_par::Float64=0.001, conv_crit_att::Float64=0.01, max_iter::Int=100) -> JMLEResult

聯合極大似然估計 (Joint MLE / JMLE)。
交替進行 `par_mle` (估計題目參數) 與 `alpha_mle` (估計受試者能力組型)，直至收斂或達到最大迭代次數。
"""
function jmle(Y::Matrix{Int}, Q::Matrix{Int}; model::Symbol=:DINA, np_method::Symbol=:Weighted, conv_crit_par::Float64=0.001, conv_crit_att::Float64=0.01, max_iter::Int=100)
    check_input(Y, Q)
    nperson, nitem = size(Y)
    nitem_q, natt = size(Q)
    
    # 1. 根據模型種類選擇對應的無參數 gate
    gate = (model in (:DINA, :NIDA, :GNIDA, :RRUM)) ? :AND : :OR
    
    # 2. 步驟 1：初始無參數估計 (AlphaNP)
    np_result = alpha_np(Y, Q; gate=gate, method=np_method, wg=1.0, ws=1.0)
    alpha_est = copy(np_result.alpha_est)
    
    # 3. 步驟 2：MLE 迭代更新
    d_par = [1.0]
    d_att = 1.0
    d_undefined = 1.0
    
    ite = 0
    conv = "Convergence criteria met."
    
    # 用於比對變動的變數
    par_est_old = nothing
    alpha_out_mle_old = nothing
    alpha_est_old = copy(alpha_est)
    undefined_flag_old = zeros(Int, nitem)
    
    # 當前迭代估計值
    par_est = nothing
    alpha_out_mle = nothing
    undefined_flag = zeros(Int, nitem)
    loglike = 0.0
    loglike_matrix = nothing
    
    while (((maximum(d_par) > conv_crit_par && d_att > 0.0) || d_undefined > 0.0 || (d_att > conv_crit_att && maximum(d_par) > 0.0)) && ite < max_iter)
        ite += 1
        
        # a) 估計題目參數 MLE
        par_est = par_mle(Y, Q, alpha_est, model)
        
        # 建立 undefined_flag (當 slip 或 guess 為 NaN 或 Inf 時設為 1，在 Optim.jl 優化下通常為 0)
        undefined_flag = zeros(Int, nitem)
        if model in (:DINA, :DINO, :NIDA, :GNIDA)
            for j in 1:length(par_est.slip)
                if isnan(par_est.slip[j]) || isinf(par_est.slip[j]) || isnan(par_est.guess[j]) || isinf(par_est.guess[j])
                    undefined_flag[j] = 1
                end
            end
        end
        
        # b) 估計 Examinee 屬性組型 MLE
        alpha_out_mle = alpha_mle(Y, Q, par_est, model; undefined_flag=undefined_flag)
        alpha_est = alpha_out_mle.alpha_est
        loglike_matrix = alpha_out_mle.loglike_matrix
        
        # c) 計算總似然值
        loglike = 0.0
        for i in 1:nperson
            loglike += loglike_matrix[alpha_out_mle.est_class[i], i]
        end
        
        # d) 計算與前一次迭代的變動量
        if ite > 1
            # 計算題目參數變動
            # 展平當前參數與舊參數
            v_curr = Float64[]
            v_old = Float64[]
            if model in (:DINA, :DINO, :NIDA, :GNIDA)
                append!(v_curr, par_est.slip)
                append!(v_curr, par_est.guess)
                append!(v_old, par_est_old.slip)
                append!(v_old, par_est_old.guess)
            else # :RRUM
                append!(v_curr, par_est.pi)
                append!(v_curr, par_est.r)
                append!(v_old, par_est_old.pi)
                append!(v_old, par_est_old.r)
            end
            
            # 排除非數值與無效項
            d_par = abs.(v_curr .- v_old)
            filter!(x -> !isnan(x) && !isinf(x), d_par)
            if isempty(d_par)
                d_par = [0.0]
            end
            
            d_undefined = sum(abs.(undefined_flag .- undefined_flag_old))
            
            # 計算屬性組型分類變動
            if alpha_out_mle_old.class_tie == alpha_out_mle.class_tie
                d_att = 0.0
            else
                changed_tie = [any(alpha_out_mle_old.class_tie[i, :] .!= alpha_out_mle.class_tie[i, :]) for i in 1:nperson]
                changed_est = [any(alpha_est[i, :] .!= alpha_est_old[i, :]) for i in 1:nperson]
                d_att = sum(changed_tie .& changed_est) / nperson
            end
        end
        
        # 存檔本次迭代以備下次對比
        par_est_old = par_est
        alpha_out_mle_old = alpha_out_mle
        alpha_est_old = copy(alpha_est)
        undefined_flag_old = copy(undefined_flag)
        
        @info "Iteration $ite: loglike = $(round(loglike, digits=5)), max_diff_par = $(round(maximum(d_par), digits=5)), diff_att = $(round(d_att, digits=5))"
    end
    
    if ite == max_iter
        conv = "Maximum iteration reached."
    end
    if maximum(d_par) > conv_crit_par && d_att == 0.0
        conv = "Examinee profile has stabilized except for randomness from ties."
    end
    
    # 4. 計算 AIC 與 BIC 模型適配度指標
    M_patterns = 2^natt
    npar = 0
    if model == :DINA || model == :DINO
        npar = 2 * nitem + M_patterns - 1
    elseif model == :NIDA
        npar = 2 * natt + M_patterns - 1
    elseif model == :GNIDA
        npar = 2 * natt * nitem + M_patterns - 1
    elseif model == :RRUM
        npar = nitem * (natt + 1) + M_patterns - 1
    end
    
    aic = -2 * loglike + 2 * npar
    bic = -2 * loglike + npar * log(nperson)
    
    return JMLEResult(
        alpha_est,
        par_est,
        alpha_out_mle.n_tie,
        undefined_flag,
        loglike,
        conv,
        ite,
        aic,
        bic,
        loglike_matrix,
        alpha_out_mle.est_class,
        np_result.loss_matrix,
        np_result.alpha_est,
        string(np_method),
        np_result.est_class,
        np_result.pattern,
        model,
        Q,
        Y
    )
end
