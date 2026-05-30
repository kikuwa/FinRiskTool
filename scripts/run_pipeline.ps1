# finRiskTool 最小流水线示例（Windows PowerShell）
# 用法: .\scripts\run_pipeline.ps1 [-LabelCol RevolverStatus]

param(
    [string]$LabelCol = "RevolverStatus"
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python cli.py dataset split --label-col $LabelCol --test-size 0.3
python cli.py pu train --dataset split --label-col $LabelCol --rate 10 --timeout-minutes 20
python cli.py fe run --dataset split --label-col $LabelCol --rate 10
python cli.py fe confirm --default
python cli.py mlbase compare --dataset split --recall-target 0.6
python cli.py mlbase test --variant top_features --dataset split

Write-Host "流水线完成。详见 docs/CLI.md"
