#Requires -Version 5.1
<#
.SYNOPSIS
  Run Mozc candidate generation for a records JSONL (keys → mozc_batch → merge → classify).

.EXAMPLE
  .\scripts\run_mozc_candidates.ps1 -Records data\interim\aozora_ruby.jsonl

.EXAMPLE
  .\scripts\run_mozc_candidates.ps1 -Records data\interim\aozora_ruby.jsonl -Limit 200 -SkipClassify
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Records,

  [string]$OutDir = "data\interim\mozc_batch",

  [string]$EnvFile = "config\mozc_batch.env",

  [int]$MaxCandidates = 0,

  [int]$Limit = 0,

  [switch]$SkipClassify,

  [string]$Name = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path $EnvFile)) {
  throw "Missing $EnvFile — copy config\mozc_batch.env.example and set MOZC_BATCH_EXE / MOZC_ENGINE_DATA_PATH"
}

$stem = if ($Name) { $Name } else { [IO.Path]::GetFileNameWithoutExtension($Records) }
$work = Join-Path $OutDir $stem
$classifyIn = Join-Path $work "classify_in.jsonl"
$comparisons = Join-Path $work "comparisons.jsonl"

$mozcArgs = @(
  "-m", "tools.dataset.main", "mozc-run",
  "--records", $Records,
  "--out", $classifyIn,
  "--work-dir", $work,
  "--env-file", $EnvFile
)
if ($MaxCandidates -gt 0) { $mozcArgs += @("--max-candidates", "$MaxCandidates") }
if ($Limit -gt 0) { $mozcArgs += @("--limit", "$Limit") }

# mozc_batch (AI-patched tree) may write AIRewriter / alive-beacon lines to stderr.
# With $ErrorActionPreference=Stop, PowerShell turns those into terminating NativeCommandError
# even when python/mozc_batch exit 0 — so merge/classify never runs. Keep stderr visible
# but only fail on non-zero exit codes.
Write-Host ">>> python $($mozcArgs -join ' ')"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& python @mozcArgs 2>&1 | ForEach-Object {
  if ($_ -is [System.Management.Automation.ErrorRecord]) {
    Write-Host $_.ToString()
  } else {
    Write-Host $_
  }
}
$mozcExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($mozcExit -ne 0) { throw "mozc-run failed with exit $mozcExit" }

if (-not $SkipClassify) {
  Write-Host ">>> classify -> $comparisons"
  $ErrorActionPreference = "Continue"
  & python -m tools.dataset.main classify --input $classifyIn --out $comparisons 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
      Write-Host $_.ToString()
    } else {
      Write-Host $_
    }
  }
  $classifyExit = $LASTEXITCODE
  $ErrorActionPreference = $prevEap
  if ($classifyExit -ne 0) { throw "classify failed with exit $classifyExit" }
}

Write-Host "Done."
Write-Host "  classify_in: $((Resolve-Path $classifyIn).Path)"
if (-not $SkipClassify) {
  Write-Host "  comparisons: $((Resolve-Path $comparisons).Path)"
}
