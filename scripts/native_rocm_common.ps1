function Get-NativeRocmContext {
    param([string]$ProjectRoot)

    $Workspace = Split-Path -Parent (Split-Path -Parent $ProjectRoot)
    $Python = if ($env:ROCM_PYTHON) {
        $env:ROCM_PYTHON
    } else {
        Join-Path $Workspace "ROCmforwindows\.venv\Scripts\python.exe"
    }
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Native ROCm Python was not found: $Python. Set ROCM_PYTHON explicitly."
    }
    [PSCustomObject]@{
        Root = $ProjectRoot
        Workspace = $Workspace
        Python = $Python
        VenvScripts = Split-Path -Parent $Python
    }
}

function Initialize-NativeRocmEnvironment {
    param([Parameter(Mandatory = $true)]$Context)

    $env:PYTHONPATH = $Context.Root
    $env:PATH = "$($Context.VenvScripts);$env:PATH"
    $env:HF_HOME = Join-Path $env:USERPROFILE ".cache\huggingface"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:PYTHONUNBUFFERED = "1"
    Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    Remove-Item Env:PYTORCH_ALLOC_CONF -ErrorAction SilentlyContinue
    Remove-Item Env:PYTORCH_HIP_ALLOC_CONF -ErrorAction SilentlyContinue
}
