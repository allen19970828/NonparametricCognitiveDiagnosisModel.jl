module NonparametricCognitiveDiagnosisModel

# 核心套件依賴
using LinearAlgebra
using Statistics
using SpecialFunctions
using Optim
using RecipesBase

# 1. 包含模組檔案
include("types.jl")
include("utils.jl")
include("alpha_np.jl")
include("nwcdm.jl")
include("probability.jl")
include("par_mle.jl")
include("alpha_mle.jl")
include("jmle.jl")
include("q_refine.jl")
include("fit.jl")
include("plots.jl")
include("print.jl")

# 2. 匯出結果結構與類型
export AbstractCDMResult,
       AbstractCognitiveModel,
       AlphaNPResult,
       NWCDMResult,
       ParMLEResult,
       AlphaMLEResult,
       JMLEResult,
       QrefineResult

# 3. 匯出核心函數
export check_input,
       alpha_permute,
       alpha_np,
       nwcdm,
       cdp,
       cdl,
       par_mle,
       alpha_mle,
       jmle,
       q_refine,
       item_fit,
       model_fit

end # module NonparametricCognitiveDiagnosisModel
