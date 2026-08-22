$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Artifacts = Join-Path $Root "artifacts"
if (-not (Test-Path -LiteralPath $Artifacts)) {
    Write-Host "No artifacts directory exists."
    exit 0
}

$Rows = Get-ChildItem -LiteralPath $Artifacts -Directory |
    Where-Object { $_.Name -ne "_trash" } |
    ForEach-Object {
    $Bytes = (Get-ChildItem -LiteralPath $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Directory = $_.Name
        MiB = [math]::Round($Bytes / 1MB, 1)
    }
} | Sort-Object MiB -Descending

$Rows | Format-Table -AutoSize
$Total = ($Rows | Measure-Object -Property MiB -Sum).Sum
$TrashBytes = if (Test-Path -LiteralPath (Join-Path $Artifacts "_trash")) {
    (Get-ChildItem -LiteralPath (Join-Path $Artifacts "_trash") -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
} else { 0 }
Write-Host "Active artifacts: $([math]::Round($Total, 1)) MiB"
Write-Host "Archived in _trash: $([math]::Round($TrashBytes / 1MB, 1)) MiB"
Write-Host "Read-only audit: no files were deleted or modified."
