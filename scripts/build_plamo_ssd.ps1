$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context
$Python = $Context.Python
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path -LiteralPath $VcVars)) {
    throw "Visual Studio 2022 C++ Build Tools were not found."
}

$Command = "call `"$VcVars`" >nul && `"$Python`" -c `"from tools.train.plamo_ssd_hip import load_extension; print(load_extension())`""
cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) { throw "PLaMo SSD HIP extension build failed." }
