param([string]$RuntimeRoot = $env:MOZCAI_RUNTIME_ROOT)

$ErrorActionPreference = "Stop"
if (-not $RuntimeRoot) { throw "Set RuntimeRoot to the paired Mozc-Ai checkout" }
$scriptPath = Join-Path $RuntimeRoot "scripts/windows_smoke.ps1"
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
$function = $ast.Find({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Assert-RerankModelSha256"
}, $true)
if ($null -eq $function) { throw "Runtime smoke has no model identity validator" }
# Load only the actual validator, without invoking process/IME/installation code.
Invoke-Expression $function.Extent.Text

$fixture = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString() + ".onnx")
try {
    [IO.File]::WriteAllBytes($fixture, [Text.Encoding]::UTF8.GetBytes("synthetic model identity fixture"))
    $installedSha = (Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant()
    $otherSha = "a" * 64
    if ($installedSha -eq $otherSha) { throw "Fixture hash unexpectedly matches alternate model" }
    $cases = @(
        @{ name = "matching model"; ping = $installedSha; scored = $installedSha; pass = $true },
        @{ name = "wrong ping model"; ping = $otherSha; scored = $installedSha; pass = $false },
        @{ name = "wrong scored model"; ping = $installedSha; scored = $otherSha; pass = $false },
        @{ name = "same wrong model in both responses"; ping = $otherSha; scored = $otherSha; pass = $false },
        @{ name = "missing ping hash"; ping = $null; scored = $installedSha; pass = $false },
        @{ name = "missing scored hash"; ping = $installedSha; scored = $null; pass = $false },
        @{ name = "malformed scored hash"; ping = $installedSha; scored = "not-a-sha"; pass = $false },
        @{ name = "case-insensitive hex"; ping = $installedSha.ToUpperInvariant(); scored = $installedSha; pass = $true }
    )
    foreach ($case in $cases) {
        $passed = $true
        try {
            Assert-RerankModelSha256 $installedSha $case.ping "ping"
            Assert-RerankModelSha256 $installedSha $case.scored "scored response"
        } catch {
            $passed = $false
            if ($_.Exception.Message -notlike "SMOKE FAIL:*") { throw }
        }
        if ($passed -ne $case.pass) { throw "Unexpected result: $($case.name)" }
        Write-Host "PASS: $($case.name)"
    }
    Write-Host "Model identity contract: $($cases.Count) cases passed"
} finally {
    Remove-Item -LiteralPath $fixture -ErrorAction SilentlyContinue
}
