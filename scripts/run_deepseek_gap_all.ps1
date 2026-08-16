#Requires -Version 5.1
<#
.SYNOPSIS
  Resume-friendly DeepSeek review for all aozora generation_gap rows.
#>
[CmdletBinding()]
param(
  [string]$InputComparisons = "data\interim\mozc_batch\aozora\comparisons.jsonl",
  [string]$Out = "data\review\aozora\reviews_generation_gap_all.jsonl",
  [string]$Model = "deepseek-v4-flash",
  [double]$InputPrice = 0.14,
  [double]$OutputPrice = 0.28,
  [double]$MaxCostUsd = 10.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host ">>> deepseek-review resume -> $Out"
$sw = [Diagnostics.Stopwatch]::StartNew()
& python -u -m tools.dataset.main deepseek-review `
  --input $InputComparisons `
  --out $Out `
  --model $Model `
  --input-price-per-million $InputPrice `
  --output-price-per-million $OutputPrice `
  --actions generation_gap `
  --max-cost-usd $MaxCostUsd `
  --execute
$code = $LASTEXITCODE
$sw.Stop()
Write-Host "EXIT=$code ELAPSED_SEC=$([math]::Round($sw.Elapsed.TotalSeconds,1))"
if (Test-Path $Out) {
  & python scripts\_review_status.py
}
exit $code
