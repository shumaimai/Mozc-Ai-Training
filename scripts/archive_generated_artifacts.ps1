param(
    [switch]$Restore,
    [string]$Archive
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Artifacts = Join-Path $Root "artifacts"
$Trash = Join-Path $Artifacts "_trash"

if ($Restore) {
    if (-not $Archive) { throw "Specify -Archive with the archive directory name." }
    $ArchiveRoot = Join-Path $Trash $Archive
    $ManifestPath = Join-Path $ArchiveRoot "manifest.json"
    if (-not (Test-Path -LiteralPath $ManifestPath)) { throw "Manifest not found: $ManifestPath" }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($Item in $Manifest.items) {
        if (Test-Path -LiteralPath $Item.original) {
            throw "Restore target already exists: $($Item.original)"
        }
        $Parent = Split-Path -Parent $Item.original
        if (-not (Test-Path -LiteralPath $Parent)) {
            New-Item -ItemType Directory -Path $Parent | Out-Null
        }
        Move-Item -LiteralPath $Item.archived -Destination $Item.original
    }
    Write-Host "Restored archive: $ArchiveRoot"
    exit 0
}

if (-not (Test-Path -LiteralPath $Artifacts)) { throw "Artifacts directory not found: $Artifacts" }
if (-not (Test-Path -LiteralPath $Trash)) {
    New-Item -ItemType Directory -Path $Trash | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ArchiveRoot = Join-Path $Trash "generated-$Stamp"
New-Item -ItemType Directory -Path $ArchiveRoot | Out-Null

$GeneratedDirectories = @(
    "plamo_hip_ssd_len256_actual",
    "plamo_hip_ssd_len256",
    "plamo_native_final_offload",
    "plamo_native_hip_ssd_conv",
    "plamo_native_hip_ssd_final",
    "plamo_native_hip_ssd",
    "plamo_native_offload_smoke",
    "plamo_native_optimized_no_offload",
    "plamo_native_qlora_smoke",
    "plamo_native_ssm_optimized",
    "plamo_native_v9_final",
    "plamo_torch_ssd_len256_actual",
    "qlora_native_smoke_final",
    "qlora_native_smoke",
    "plamo-ssd-extension-v2",
    "plamo-ssd-extension-v3",
    "plamo-ssd-extension-v4",
    "plamo-ssd-extension-v5",
    "plamo-ssd-extension-v6",
    "plamo-ssd-extension-v7",
    "plamo-ssd-extension-v8",
    "plamo-ssd-extension-v9"
)

$LegacyBuildFiles = @(
    ".ninja_log",
    "build.ninja",
    "plamo_ssd_hip_ext.exp",
    "plamo_ssd_hip_ext.lib",
    "plamo_ssd_hip_ext.pyd",
    "plamo_ssd.cuda.o",
    "plamo_ssd.o"
)

$Items = @()
foreach ($Name in $GeneratedDirectories) {
    $Source = Join-Path $Artifacts $Name
    if (-not (Test-Path -LiteralPath $Source)) { continue }
    $Destination = Join-Path $ArchiveRoot $Name
    Move-Item -LiteralPath $Source -Destination $Destination
    $Items += [PSCustomObject]@{ original = $Source; archived = $Destination }
}

$LegacyRoot = Join-Path $ArchiveRoot "plamo-ssd-extension-legacy-root"
foreach ($Name in $LegacyBuildFiles) {
    $Source = Join-Path (Join-Path $Artifacts "plamo-ssd-extension") $Name
    if (-not (Test-Path -LiteralPath $Source)) { continue }
    if (-not (Test-Path -LiteralPath $LegacyRoot)) {
        New-Item -ItemType Directory -Path $LegacyRoot | Out-Null
    }
    $Destination = Join-Path $LegacyRoot $Name
    Move-Item -LiteralPath $Source -Destination $Destination
    $Items += [PSCustomObject]@{ original = $Source; archived = $Destination }
}

$Manifest = [PSCustomObject]@{
    created_at = (Get-Date).ToString("o")
    reason = "Generated ROCm smoke outputs and superseded extension builds"
    data_modified = $false
    items = $Items
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ArchiveRoot "manifest.json") -Encoding UTF8

Write-Host "Archived $($Items.Count) generated items to: $ArchiveRoot"
Write-Host "No input data, retained comparison artifacts, or current hash build were modified."
Write-Host "Restore with: .\scripts\archive_generated_artifacts.ps1 -Restore -Archive $([IO.Path]::GetFileName($ArchiveRoot))"
