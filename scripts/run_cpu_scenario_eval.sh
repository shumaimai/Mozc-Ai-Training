#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WIN="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${MOZC_TRAIN_WORK_ROOT:-$HOME/work/mozc-ai-training}"
sed -i 's/\r$//' "$WIN/scripts/cpu_scenario_eval.py"
cp "$WIN/scripts/cpu_scenario_eval.py" "$ROOT/cpu_scenario_eval.py"
cp "$WIN/tools/train/infer.py" "$ROOT/tools/train/infer.py"
cp "$WIN/tools/train/benchmark.py" "$ROOT/tools/train/benchmark.py"

docker start rocm-torch >/dev/null
docker exec rocm-torch bash -lc '
  export PYTHONPATH=/work/mozc-ai-training
  export HF_HOME=/root/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES=
  cd /work/mozc-ai-training
  python3 /work/mozc-ai-training/cpu_scenario_eval.py
' 2>&1 | tee "$ROOT/cpu_scenario_eval.log"
