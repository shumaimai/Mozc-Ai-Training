$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$KeyPath = Join-Path $env:USERPROFILE ".config\vastai\vast_api_key"
if (-not (Get-Command vastai -ErrorAction SilentlyContinue)) {
    throw "Vast.ai CLI is not installed. Install it with: pip install vastai"
}
if (-not (Test-Path -LiteralPath $KeyPath)) {
    throw "Vast.ai API key file was not found: $KeyPath"
}

$env:VAST_API_KEY = ([System.IO.File]::ReadAllText($KeyPath)).Trim()
try {
    if (-not $env:VAST_API_KEY) { throw "Vast.ai API key file is empty." }
    $UserRaw = & vastai show user --raw 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Vast.ai authentication failed." }
    $User = ($UserRaw -join "`n") | ConvertFrom-Json
    $Balance = if ($null -ne $User.credit) { $User.credit } elseif ($null -ne $User.balance) { $User.balance } else { $null }

    # -n disables the implicit external=false rentable=true verified=true filters.
    $OfferRaw = & vastai search offers -n 'gpu_arch=amd rented=any' --raw --limit 100 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Vast.ai AMD offer search failed." }
    $Parsed = ($OfferRaw -join "`n") | ConvertFrom-Json
    $Offers = if ($null -ne $Parsed.offers) { @($Parsed.offers) } else { @($Parsed) }
    $Sanitized = @($Offers | ForEach-Object {
        [PSCustomObject]@{
            id = $_.id
            gpu_name = $_.gpu_name
            gpu_arch = $_.gpu_arch
            num_gpus = $_.num_gpus
            dph_total = $_.dph_total
            rentable = $_.rentable
            verified = $_.verified
            geolocation = $_.geolocation
        }
    })
    $Report = [PSCustomObject]@{
        checked_at = (Get-Date).ToString("o")
        authenticated = $true
        balance = $Balance
        search_query = "vastai search offers -n 'gpu_arch=amd rented=any'"
        default_filters_disabled = $true
        amd_offer_count = $Sanitized.Count
        offers = $Sanitized
        instance_created = $false
        billing_action_performed = $false
    }
    $Artifacts = Join-Path $Root "artifacts"
    if (-not (Test-Path -LiteralPath $Artifacts)) {
        New-Item -ItemType Directory -Path $Artifacts | Out-Null
    }
    $Report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Artifacts "vastai-amd-availability.json") -Encoding UTF8
    Write-Host "Vast.ai authentication: OK"
    Write-Host "Balance: `$$Balance"
    Write-Host "AMD offers: $($Sanitized.Count)"
    Write-Host "No instance was created and no billing action was performed."
} finally {
    Remove-Item Env:VAST_API_KEY -ErrorAction SilentlyContinue
}
