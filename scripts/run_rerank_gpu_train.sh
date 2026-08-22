#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WIN="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${MOZC_TRAIN_WORK_ROOT:-$HOME/work/mozc-ai-training}"
MODEL="${1:-sbintuitions/modernbert-ja-70m}"
BS="${2:-1536}"

docker start rocm-torch >/dev/null
docker exec rocm-torch bash -lc 'pkill -f train_cross_encoder || true; pkill -f lora_sft || true' || true
sleep 2

cp -f "$WIN/tools/rerank/train_cross_encoder.py" "$ROOT/tools/rerank/"
sed -i 's/\r$//' "$ROOT/tools/rerank/train_cross_encoder.py"

# Wait until a trivial CUDA alloc works (ROCm/WSL can be flaky right after churn).
for i in 1 2 3 4 5 6 7 8; do
  if docker exec rocm-torch bash -lc 'python3 -c "import torch; x=torch.zeros(8,device=\"cuda\"); torch.cuda.synchronize(); print(\"cuda_ok\", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))"'; then
    break
  fi
  echo "cuda not ready; attempt $i"
  sleep 3
done

docker exec rocm-torch bash -lc ': > /work/mozc-ai-training/artifacts/rerank/train_run.log'
docker exec -d rocm-torch bash -lc "
set -euo pipefail
cd /work/mozc-ai-training
export PYTHONPATH=/work/mozc-ai-training
export HF_HOME=/root/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF
python3 -m tools.rerank.train_cross_encoder train \
  --train /work/mozc-ai-training/data/rerank_v2/train.jsonl \
  --eval /work/mozc-ai-training/data/rerank_v2/holdout.jsonl \
  --model $MODEL \
  --out /work/mozc-ai-training/artifacts/rerank/modernbert70m_ce \
  --epochs 2 \
  --batch-size $BS \
  --max-len 128 \
  --max-neg 15 \
  --num-workers 0 \
  --fp16 \
  --require-cuda \
  --require-gold-in-nbest \
  > /work/mozc-ai-training/artifacts/rerank/train_run.log 2>&1
"

sleep 25
echo "=== procs ==="
docker exec rocm-torch bash -lc 'ps -eo pid,etime,args | grep -F train_cross_encoder | grep -v grep || echo NONE'
echo "=== log ==="
docker exec rocm-torch bash -lc 'tail -n 80 /work/mozc-ai-training/artifacts/rerank/train_run.log'
