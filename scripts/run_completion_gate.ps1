param([switch]$IncludeModel)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context

$Arguments = @((Join-Path $PSScriptRoot "run_completion_gate.py"))
if ($IncludeModel) { $Arguments += "--include-model" }
& $Context.Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Native ROCm completion gate failed." }
