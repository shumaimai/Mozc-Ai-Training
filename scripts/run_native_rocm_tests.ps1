param([switch]$Benchmarks)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context
$Python = $Context.Python

& $Python -c "import torch; assert torch.version.hip; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "ROCm PyTorch health check failed." }

& $Python (Join-Path $PSScriptRoot "probe_bnb_rocm.py")
if ($LASTEXITCODE -ne 0) { throw "bitsandbytes NF4 test failed." }

& $Python (Join-Path $PSScriptRoot "probe_paged_optimizer_rocm.py")
if ($LASTEXITCODE -ne 0) { throw "PagedAdamW8bit resume test failed." }

& (Join-Path $PSScriptRoot "build_plamo_ssd.ps1")
if ($LASTEXITCODE -ne 0) { throw "PLaMo HIP extension build failed." }

& $Python (Join-Path $PSScriptRoot "test_plamo_fallback_math.py")
if ($LASTEXITCODE -ne 0) { throw "PLaMo FP32 parity test failed." }

& $Python (Join-Path $PSScriptRoot "test_plamo_low_precision_oracle.py")
if ($LASTEXITCODE -ne 0) { throw "PLaMo FP16/BF16 independent oracle failed." }

if ($Benchmarks) {
    & $Python (Join-Path $PSScriptRoot "benchmark_plamo_ssd.py") --iterations 20
    if ($LASTEXITCODE -ne 0) { throw "PLaMo SSD benchmark failed." }
    & $Python (Join-Path $PSScriptRoot "benchmark_plamo_conv.py")
    if ($LASTEXITCODE -ne 0) { throw "PLaMo causal-conv benchmark failed." }
}

Write-Host "Native ROCm GPU tests passed."
