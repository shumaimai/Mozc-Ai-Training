#!/usr/bin/env bash
set -euo pipefail
cd /work/mozc-ai-training
export PYTHONPATH=/work/mozc-ai-training
export HF_HOME=/root/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

MODE="${1:-smoke}"
OUT_BASE="/work/mozc-ai-training/artifacts"
mkdir -p "$OUT_BASE"

echo "=== PLaMo LoRA mode=$MODE device probe ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

if [[ "$MODE" == "smoke" ]]; then
  python3 -m tools.train.lora_sft \
    --data /work/mozc-ai-training/train_mixed.jsonl \
    --model pfnet/plamo-2-1b \
    --trust-remote-code \
    --out "$OUT_BASE/plamo2_1b_lora_smoke" \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --max-length 512 \
    --lora-r 16 \
    --lora-alpha 32 \
    --limit 64
elif [[ "$MODE" == "poc" ]]; then
  python3 -m tools.train.lora_sft \
    --data /work/mozc-ai-training/train_mixed.jsonl \
    --model pfnet/plamo-2-1b \
    --trust-remote-code \
    --out "$OUT_BASE/plamo2_1b_lora" \
    --epochs 1 \
    --max-steps 100 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --max-length 256 \
    --lora-r 16 \
    --lora-alpha 32 \
    --limit 2048
else
  python3 -m tools.train.lora_sft \
    --data /work/mozc-ai-training/train_mixed.jsonl \
    --model pfnet/plamo-2-1b \
    --trust-remote-code \
    --out "$OUT_BASE/plamo2_1b_lora" \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --max-length 512 \
    --lora-r 16 \
    --lora-alpha 32
fi

echo "DONE mode=$MODE"
