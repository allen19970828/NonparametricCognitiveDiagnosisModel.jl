# docs/make.jl

using Documenter
using NonparametricCognitiveDiagnosisModel

# 若要將本套件放入環境路徑，先將套件根目錄加入 LOAD_PATH
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

makedocs(
    sitename = "NonparametricCognitiveDiagnosisModel.jl",
    modules = [NonparametricCognitiveDiagnosisModel],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://yourusername.github.io/NonparametricCognitiveDiagnosisModel.jl",
        assets = String[],
    ),
    pages = [
        "首頁 / Home" => "index.md",
    ]
)

# 部署指令 (當在 GitHub Actions CI 中時會自動觸發)
# deploydocs(
#     repo = "github.com/yourusername/NonparametricCognitiveDiagnosisModel.jl.git",
#     devbranch = "main",
# )
