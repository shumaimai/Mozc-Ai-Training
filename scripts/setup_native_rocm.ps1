$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context
$Python = $Context.Python

& $Python -c "import torch; assert torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "The native ROCm PyTorch environment is not healthy." }

& uv pip install --python $Python -r (Join-Path $Root "requirements-rocm-windows.txt")
if ($LASTEXITCODE -ne 0) { throw "Training dependency installation failed." }

& $Python -c "import torch; assert '+rocm7.14.0' in torch.__version__, torch.__version__; print('ROCm torch preserved:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The AMD ROCm torch wheel was replaced unexpectedly." }

& (Join-Path $PSScriptRoot "run_native_rocm_tests.ps1")
if ($LASTEXITCODE -ne 0) { throw "Native ROCm validation suite failed." }
