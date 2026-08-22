param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [string]$Model = "pfnet/plamo-2-1b",
    [string]$Revision = "92c75fd6eea9018bcb9c33ee8921589febe071fa",
    [string]$Out = "artifacts\plamo2_1b_lora_native_rocm",
    [int]$MaxSteps = 2,
    [int]$Limit = 16,
    [int]$MaxLength = 64,
    [switch]$QLoRA,
    [switch]$ActivationOffload,
    [switch]$PagedOptimizer,
    [switch]$UseHipKernels,
    [switch]$UseExperimentalHipSsd
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context
$Python = $Context.Python
$DataPath = if ([System.IO.Path]::IsPathRooted($Data)) { $Data } else { Join-Path $Root $Data }
$OutPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $Root $Out }

if (-not (Test-Path -LiteralPath $DataPath)) { throw "Training data not found: $DataPath" }

if ($QLoRA -and $Model -eq "pfnet/plamo-2-1b") {
    & $Python (Join-Path $PSScriptRoot "plan_model_memory.py") $Model `
        --revision $Revision --parameters 1291448320
    if ($LASTEXITCODE -ne 0) { throw "The model does not fit the supported QLoRA memory path." }
}

$Arguments = @(
    "-m", "tools.train.lora_sft",
    "--data", $DataPath,
    "--model", $Model,
    "--trust-remote-code",
    "--revision", $Revision,
    "--out", $OutPath,
    "--epochs", "1",
    "--max-steps", "$MaxSteps",
    "--batch-size", "1",
    "--grad-accum", "1",
    "--lr", "2e-4",
    "--max-length", "$MaxLength",
    "--lora-r", "16",
    "--lora-alpha", "32",
    "--limit", "$Limit"
)
if ($QLoRA) { $Arguments += "--load-in-4bit" }
if ($ActivationOffload) { $Arguments += "--activation-offload" }
if ($PagedOptimizer) { $Arguments += @("--optimizer", "paged_adamw_8bit") }
if ($UseHipKernels) { $Arguments += "--use-hip-kernels" }
if ($UseExperimentalHipSsd) { $Arguments += "--use-experimental-hip-ssd" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Native ROCm PLaMo training failed." }
