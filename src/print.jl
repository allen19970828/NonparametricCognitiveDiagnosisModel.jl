# src/print.jl

import Base: show

"""
輔助函數：格式化二維矩陣輸出，加上列與行標籤。
"""
function print_matrix_with_labels(io::IO, mat::AbstractMatrix, rownames::Vector{String}, colnames::Vector{String})
    # 決定每列的最大寬度以對齊
    col_widths = [maximum(length.(vcat(colnames[j], string.(mat[:, j])))) for j in 1:size(mat, 2)]
    rowname_width = maximum(length.(rownames))
    
    # 預留行標籤寬度
    print(io, lpad("", rowname_width + 2))
    for j in 1:size(mat, 2)
        print(io, lpad(colnames[j], col_widths[j] + 2))
    end
    println(io)
    
    for i in 1:size(mat, 1)
        print(io, lpad(rownames[i], rowname_width + 2))
        for j in 1:size(mat, 2)
            val = mat[i, j]
            val_str = val isa Real ? (isnan(val) ? "NA" : string(round(val, digits=4))) : string(val)
            print(io, lpad(val_str, col_widths[j] + 2))
        end
        println(io)
    end
end


function show(io::IO, ::MIME"text/plain", res::AlphaNPResult)
    println(io, "==================================================")
    println(io, "  NCDM (AlphaNP) Examinee Attribute Profiles      ")
    println(io, "==================================================")
    println(io, "Classification Method : ", res.method)
    println(io, "Gate Type             : ", res.gate)
    println(io, "Number of Examinees   : ", size(res.Y, 1))
    println(io, "Number of Attributes  : ", size(res.Q, 2))
    println(io, "--------------------------------------------------")
    println(io, "Estimated Examinee Attribute Profiles (sample):")
    
    I, K = size(res.alpha_est)
    display_rows = min(I, 10)
    rownames = ["Examinee $i" for i in 1:display_rows]
    colnames = ["Attr $k" for k in 1:K]
    
    print_matrix_with_labels(io, res.alpha_est[1:display_rows, :], rownames, colnames)
    if I > 10
        println(io, "  ... ($I examinees total) ...")
    end
    println(io, "==================================================")
end


function show(io::IO, ::MIME"text/plain", res::NWCDMResult)
    println(io, "==================================================")
    println(io, "  NWCDM (Nonparametric Weighted) Estimation       ")
    println(io, "==================================================")
    println(io, "Number of Examinees   : ", size(res.Y, 1))
    println(io, "Number of Attributes  : ", size(res.Q, 2))
    println(io, "Number of Items       : ", size(res.Q, 1))
    println(io, "Iterations to Converge: ", res.iterations)
    println(io, "--------------------------------------------------")
    
    # 輸出題目估計的加權權重
    println(io, "Estimated Item Weights (Slip & Guess):")
    J = length(res.slip_weights)
    item_rownames = ["Item $j" for j in 1:J]
    item_colnames = ["Slip Weight", "Guess Weight"]
    item_mat = hcat(res.slip_weights, res.guess_weights)
    print_matrix_with_labels(io, item_mat, item_rownames, item_colnames)
    
    println(io, "--------------------------------------------------")
    println(io, "Estimated Examinee Attribute Profiles (sample):")
    I, K = size(res.alpha_est)
    display_rows = min(I, 10)
    rownames = ["Examinee $i" for i in 1:display_rows]
    colnames = ["Attr $k" for k in 1:K]
    
    print_matrix_with_labels(io, res.alpha_est[1:display_rows, :], rownames, colnames)
    if I > 10
        println(io, "  ... ($I examinees total) ...")
    end
    println(io, "==================================================")
end


function show(io::IO, ::MIME"text/plain", res::AlphaMLEResult)
    println(io, "==================================================")
    println(io, "  Conditional MLE Examinee Attribute Profiles    ")
    println(io, "==================================================")
    println(io, "Model Type            : ", res.model)
    println(io, "Number of Examinees   : ", size(res.Y, 1))
    println(io, "Number of Attributes  : ", size(res.Q, 2))
    println(io, "--------------------------------------------------")
    println(io, "Estimated Examinee Attribute Profiles (sample):")
    
    I, K = size(res.alpha_est)
    display_rows = min(I, 10)
    rownames = ["Examinee $i" for i in 1:display_rows]
    colnames = ["Attr $k" for k in 1:K]
    
    print_matrix_with_labels(io, res.alpha_est[1:display_rows, :], rownames, colnames)
    if I > 10
        println(io, "  ... ($I examinees total) ...")
    end
    println(io, "==================================================")
end


function show(io::IO, ::MIME"text/plain", res::ParMLEResult)
    println(io, "==================================================")
    println(io, "  Conditional MLE Item Parameter Estimates        ")
    println(io, "==================================================")
    println(io, "Model Type            : ", res.model)
    println(io, "Estimation Method     : Box-constrained Likelihood Optimization")
    println(io, "--------------------------------------------------")
    
    if res.model in (:DINA, :DINO, :NIDA)
        names = res.model == :NIDA ? ["Attribute $k" for k in 1:length(res.slip)] : ["Item $j" for j in 1:length(res.slip)]
        colnames = ["Slip", "SE.Slip", "Guess", "SE.Guess"]
        mat = hcat(res.slip, res.se_slip, res.guess, res.se_guess)
        print_matrix_with_labels(io, mat, names, colnames)
        
    elseif res.model == :GNIDA
        J, K = size(res.slip)
        names = ["Item $j" for j in 1:J]
        colnames = String[]
        for k in 1:K
            push!(colnames, "Slip.A$k", "SE.Slip.A$k", "Guess.A$k", "SE.Guess.A$k")
        end
        # 交錯排列列向量
        mat = Matrix{Any}(undef, J, 4K)
        for j in 1:J
            for k in 1:K
                if res.Q[j, k] == 1
                    mat[j, 4*(k-1) + 1] = res.slip[j, k]
                    mat[j, 4*(k-1) + 2] = res.se_slip[j, k]
                    mat[j, 4*(k-1) + 3] = res.guess[j, k]
                    mat[j, 4*(k-1) + 4] = res.se_guess[j, k]
                else
                    mat[j, 4*(k-1) + 1] = "NA"
                    mat[j, 4*(k-1) + 2] = "NA"
                    mat[j, 4*(k-1) + 3] = "NA"
                    mat[j, 4*(k-1) + 4] = "NA"
                end
            end
        end
        print_matrix_with_labels(io, mat, names, colnames)
        
    elseif res.model == :RRUM
        J, K = size(res.r)
        names = ["Item $j" for j in 1:J]
        colnames = ["Pi", "SE.Pi"]
        for k in 1:K
            push!(colnames, "r.A$k", "SE.r.A$k")
        end
        mat = Matrix{Any}(undef, J, 2 + 2K)
        for j in 1:J
            mat[j, 1] = res.pi[j]
            mat[j, 2] = res.se_pi[j]
            for k in 1:K
                if res.Q[j, k] == 1
                    mat[j, 2 + 2*(k-1) + 1] = res.r[j, k]
                    mat[j, 2 + 2*(k-1) + 2] = res.se_r[j, k]
                else
                    mat[j, 2 + 2*(k-1) + 1] = "NA"
                    mat[j, 2 + 2*(k-1) + 2] = "NA"
                end
            end
        end
        print_matrix_with_labels(io, mat, names, colnames)
    end
    println(io, "==================================================")
end


function show(io::IO, ::MIME"text/plain", res::JMLEResult)
    println(io, "==================================================")
    println(io, "  Joint MLE (JMLE) Estimation Summary             ")
    println(io, "==================================================")
    println(io, "Model Type            : ", res.model)
    println(io, "Convergence Status    : ", res.convergence)
    println(io, "Number of Iterations  : ", res.iterations)
    println(io, "Total Log-Likelihood  : ", round(res.loglike, digits=4))
    println(io, "AIC                   : ", round(res.aic, digits=4))
    println(io, "BIC                   : ", round(res.bic, digits=4))
    println(io, "--------------------------------------------------")
    
    # 輸出題目參數
    println(io, "Estimated Item Parameters:")
    show(io, MIME("text/plain"), res.par_est)
    
    println(io, "--------------------------------------------------")
    println(io, "Estimated Examinee Attribute Profiles (sample):")
    I, K = size(res.alpha_est)
    display_rows = min(I, 10)
    rownames = ["Examinee $i" for i in 1:display_rows]
    colnames = ["Attr $k" for k in 1:K]
    
    print_matrix_with_labels(io, res.alpha_est[1:display_rows, :], rownames, colnames)
    if I > 10
        println(io, "  ... ($I examinees total) ...")
    end
    println(io, "==================================================")
end


function show(io::IO, ::MIME"text/plain", res::QrefineResult)
    println(io, "==================================================")
    println(io, "  Q-matrix Refinement Results                     ")
    println(io, "==================================================")
    println(io, "Gate Type             : ", res.gate)
    println(io, "Number of Items       : ", size(res.modified_Q, 1))
    println(io, "Number of Attributes  : ", size(res.modified_Q, 2))
    println(io, "--------------------------------------------------")
    
    # 輸出修正後的 Q 矩陣
    println(io, "Modified Q-matrix:")
    J, K = size(res.modified_Q)
    rownames = ["Item $j" for j in 1:J]
    colnames = ["Attr $k" for k in 1:K]
    print_matrix_with_labels(io, res.modified_Q, rownames, colnames)
    
    println(io, "--------------------------------------------------")
    println(io, "Modified Entries (Item & Attribute indices):")
    if res.modified_entries isa String
        println(io, "  ", res.modified_entries, " (No changes were made)")
    else
        N = size(res.modified_entries, 1)
        mod_rownames = ["Change $n" for n in 1:N]
        mod_colnames = ["Item Index", "Attribute Index"]
        print_matrix_with_labels(io, res.modified_entries, mod_rownames, mod_colnames)
    end
    println(io, "==================================================")
end
