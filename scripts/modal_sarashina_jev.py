"""Run Sarashina-JEV training on Modal and detach safely.

One-time dataset staging example:
    modal volume create mozc-training-data
    modal volume put mozc-training-data data/public/rerank_ctx/train_v2.jsonl /train_v2.jsonl
    modal volume put mozc-training-data data/public/rerank_ctx/eval_unseen_v2.jsonl /eval_unseen_v2.jsonl

Launch:
    modal run scripts/modal_sarashina_jev.py \
      --limit 2000 \
      --keep-layers 12 \
      --train-last-n-layers 4 \
      --out /artifacts/sarashina_jev/12l_smoke
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

app = modal.App("mozc-sarashina-jev")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48,<5",
        "tokenizers>=0.21,<0.23",
        "sentencepiece>=0.2",
        "safetensors>=0.4",
        "accelerate>=0.28",
    )
    .add_local_dir("tools", "/root/repo/tools")
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
training_data = modal.Volume.from_name("mozc-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=6 * 60 * 60,
    volumes={
        "/artifacts": artifacts,
        "/root/.cache/huggingface": hf_cache,
        "/data": training_data,
    },
)
def train(
    train_path: str = "/data/train_v2.jsonl",
    eval_path: str = "/data/eval_unseen_v2.jsonl",
    model: str = "sbintuitions/sarashina2.2-0.5b",
    out: str = "/artifacts/sarashina_jev/12l_smoke",
    keep_layers: int = 12,
    page_size: int = 5,
    train_last_n_layers: int = 4,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 128,
    anchor_weight: float = 0.25,
    limit: int = 2000,
    fp16: bool = False,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"training data not found: {train_path}; upload it to mozc-training-data"
        )
    if eval_path and not os.path.isfile(eval_path):
        raise FileNotFoundError(
            f"eval data not found: {eval_path}; upload it to mozc-training-data"
        )

    cmd = [
        sys.executable,
        "-m",
        "tools.sarashina_jev.train",
        "--train",
        train_path,
        "--model",
        model,
        "--out",
        out,
        "--keep-layers",
        str(keep_layers),
        "--page-size",
        str(page_size),
        "--train-last-n-layers",
        str(train_last_n_layers),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--max-length",
        str(max_length),
        "--anchor-weight",
        str(anchor_weight),
        "--gradient-checkpointing",
    ]
    if eval_path:
        cmd += ["--eval", eval_path]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    cmd += ["--fp16" if fp16 else "--bf16"]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/repo"
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)

    artifacts.commit()
    hf_cache.commit()
    print(f"DONE rc={proc.returncode} out={out}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Sarashina-JEV training failed rc={proc.returncode}")

    return {"out": out, "returncode": proc.returncode}


@app.local_entrypoint()
def main(
    train_path: str = "/data/train_v2.jsonl",
    eval_path: str = "/data/eval_unseen_v2.jsonl",
    model: str = "sbintuitions/sarashina2.2-0.5b",
    out: str = "/artifacts/sarashina_jev/12l_smoke",
    keep_layers: int = 12,
    page_size: int = 5,
    train_last_n_layers: int = 4,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 128,
    anchor_weight: float = 0.25,
    limit: int = 2000,
    fp16: bool = False,
):
    call = train.spawn(
        train_path=train_path,
        eval_path=eval_path,
        model=model,
        out=out,
        keep_layers=keep_layers,
        page_size=page_size,
        train_last_n_layers=train_last_n_layers,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_length=max_length,
        anchor_weight=anchor_weight,
        limit=limit,
        fp16=fp16,
    )
    print(
        f"SPAWNED function_call_id={call.object_id} out={out}",
        flush=True,
    )
