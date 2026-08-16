#Requires -Version 5.1
<#
.SYNOPSIS
  Chunked DeepSeek review for wikidata generation_gap (resume-friendly).

.DESCRIPTION
  Filters comparisons to generation_gap once, then runs deepseek-review with
  --limit so a failure or Ctrl+C only loses the current chunk. Re-running
  resumes from --out via comparison_key.

  Full corpus is ~29k unique gaps; at aozora rates (~$0.0006/item) that is
  above the default $10 budget (~15k items). Use -MaxItems / -MaxCostUsd to
  stage expansion after a pilot sample.
#>
[CmdletBinding()]
param(
  [string]$Comparisons = "data\interim\mozc_batch\wikidata\comparisons.jsonl",
  [string]$GapInput = "data\review\wikidata\generation_gap.jsonl",
  [string]$Out = "data\review\wikidata\reviews_generation_gap.jsonl",
  [string]$Model = "deepseek-v4-flash-0731",
  [double]$InputPrice = 0.14,
  [double]$OutputPrice = 0.28,
  [double]$MaxCostUsd = 10.0,
  [int]$BatchSize = 200,
  [int]$MaxItems = 500,
  [int]$Workers = 1,
  [switch]$DryRun,
  [switch]$SkipFilter
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Qwen Cloud / OpenAI-compatible gateway (optional override).
if (-not $env:DEEPSEEK_BASE_URL) {
  $env:DEEPSEEK_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
}
Write-Host ">>> API base: $($env:DEEPSEEK_BASE_URL)"
Write-Host ">>> model: $Model"

New-Item -ItemType Directory -Force -Path (Split-Path $GapInput) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

if (-not $SkipFilter -or -not (Test-Path $GapInput)) {
  Write-Host ">>> filter generation_gap -> $GapInput"
  & python -c @"
import json
from pathlib import Path
from tools.dataset.deepseek_review import comparison_key
src = Path(r'$Comparisons')
dst = Path(r'$GapInput')
seen = set()
n_in = n_out = 0
with src.open(encoding='utf-8-sig') as fin, dst.open('w', encoding='utf-8') as fout:
    for line in fin:
        row = json.loads(line)
        if row.get('action') != 'generation_gap':
            continue
        n_in += 1
        key = comparison_key(row)
        if key in seen:
            continue
        seen.add(key)
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
        n_out += 1
print(f'filtered gap_rows={n_in} unique={n_out} -> {dst}')
"@
}

Write-Host ">>> dry-run status"
& python -m tools.dataset.main deepseek-review `
  --input $GapInput `
  --out $Out `
  --model $Model `
  --actions generation_gap `
  --input-price-per-million $InputPrice `
  --output-price-per-million $OutputPrice `
  --max-cost-usd $MaxCostUsd `
  --limit $BatchSize `
  --workers $Workers

if ($DryRun) {
  Write-Host "DryRun set; exiting before execute"
  exit 0
}

$remainingBudgetItems = $MaxItems
$totalWritten = 0
$batchIndex = 0
while ($remainingBudgetItems -gt 0) {
  $batchIndex += 1
  $take = [Math]::Min($BatchSize, $remainingBudgetItems)
  Write-Host ">>> batch $batchIndex execute limit=$take workers=$Workers (session cap remaining=$remainingBudgetItems)"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & python -u -m tools.dataset.main deepseek-review `
    --input $GapInput `
    --out $Out `
    --model $Model `
    --actions generation_gap `
    --input-price-per-million $InputPrice `
    --output-price-per-million $OutputPrice `
    --max-cost-usd $MaxCostUsd `
    --limit $take `
    --workers $Workers `
    --execute
  $code = $LASTEXITCODE
  $sw.Stop()
  Write-Host "batch EXIT=$code ELAPSED_SEC=$([math]::Round($sw.Elapsed.TotalSeconds,1))"
  if ($code -ne 0) {
    Write-Host "Stopping on batch failure; re-run the same script to resume."
    exit $code
  }

  # Detect whether anything new remains / was written via dry-run pending.
  $status = & python -m tools.dataset.main deepseek-review `
    --input $GapInput `
    --out $Out `
    --model $Model `
    --actions generation_gap `
    --input-price-per-million $InputPrice `
    --output-price-per-million $OutputPrice `
    --max-cost-usd $MaxCostUsd `
    --limit 1 `
    --workers $Workers 2>&1 | Out-String
  Write-Host $status.Trim()
  if ($status -match 'pending=0') {
    Write-Host "All unique pending gaps reviewed."
    break
  }
  if ($status -match 'batch=0') {
    Write-Host "No work left in this batch window."
    break
  }

  $remainingBudgetItems -= $take
  $totalWritten += $take
}

Write-Host ">>> session done (approx capped items=$totalWritten). Out=$Out"
exit 0
