# Mode A: run mozc_batch.exe natively on Windows (NTFS mmap for mozc.data).
# Usage (from repo root or any cwd):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mozc_batch_mode_a.ps1
# Optional:
#   -WorkDir data\rerank_ctx\work\mozc_smoke200
param(
    [string]$RepoRoot = "",
    [string]$WorkDir = "data\rerank_ctx\work\mozc",
    [string]$EnvFile = "config\mozc_batch.env",
    [int]$MaxCandidates = 80
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot
$work = Join-Path $RepoRoot $WorkDir
$keys = Join-Path $work "keys.txt"
$cands = Join-Path $work "candidates.tsv"
if (-not (Test-Path $keys)) { throw "keys.txt missing: $keys" }

$exe = $null
$data = $null
$max = $MaxCandidates
Get-Content (Join-Path $RepoRoot $EnvFile) | ForEach-Object {
    if ($_ -match '^\s*MOZC_BATCH_EXE=(.*)$') { $exe = $Matches[1].Trim().Trim('"').Trim("'") }
    if ($_ -match '^\s*MOZC_ENGINE_DATA_PATH=(.*)$') { $data = $Matches[1].Trim().Trim('"').Trim("'") }
    if ($_ -match '^\s*MOZC_MAX_CANDIDATES=(.*)$') { $max = [int]$Matches[1].Trim() }
}
if ($MaxCandidates -gt 0) { $max = $MaxCandidates }
if (-not $exe -or -not (Test-Path $exe)) { throw "MOZC_BATCH_EXE missing: $exe" }
if (-not $data -or -not (Test-Path $data)) { throw "MOZC_ENGINE_DATA_PATH missing: $data" }

New-Item -ItemType Directory -Force -Path $work | Out-Null
Write-Host "Mode A mozc_batch keys=$keys out=$cands max=$max"
& $exe --engine_data_path=$data --input=$keys --output=$cands --max_candidates=$max
if ($LASTEXITCODE -ne 0) { throw "mozc_batch failed rc=$LASTEXITCODE" }
Write-Host "wrote $cands bytes=$((Get-Item $cands).Length)"
