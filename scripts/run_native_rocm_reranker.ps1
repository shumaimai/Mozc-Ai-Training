param(
    [string]$Model = "sbintuitions/modernbert-ja-70m",
    [int]$BatchSize = 128,
    [int]$Epochs = 2
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "native_rocm_common.ps1")
$Context = Get-NativeRocmContext -ProjectRoot $Root
Initialize-NativeRocmEnvironment -Context $Context
$Python = $Context.Python
$Train = Join-Path $Root "data\rerank_v2\train.jsonl"
$Eval = Join-Path $Root "data\rerank_v2\holdout.jsonl"
$Out = Join-Path $Root "artifacts\rerank\modernbert70m_ce_native_rocm"

if (-not (Test-Path -LiteralPath $Train)) { throw "Training data not found: $Train" }
if (-not (Test-Path -LiteralPath $Eval)) { throw "Evaluation data not found: $Eval" }

& $Python -m tools.rerank.train_cross_encoder train `
    --train $Train `
    --eval $Eval `
    --model $Model `
    --out $Out `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --max-len 128 `
    --max-neg 15 `
    --num-workers 0 `
    --fp16 `
    --grad-checkpointing `
    --require-cuda `
    --require-gold-in-nbest `
    --save-every 100 `
    --auto-resume
if ($LASTEXITCODE -ne 0) { throw "Native ROCm reranker training failed." }
