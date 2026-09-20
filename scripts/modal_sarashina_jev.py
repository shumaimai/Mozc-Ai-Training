"""Run Sarashina-JEV fully in Modal, including optional public-data bootstrap.

Cloud-only smoke:
    modal run scripts/modal_sarashina_jev.py \
      --bootstrap \
      --limit 2000 \
      --keep-layers 12 \
      --train-last-n-layers 4 \
      --out /artifacts/sarashina_jev/12l_public_smoke
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

app = modal.App("mozc-sarashina-jev")

base_deps = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=3.0",
        "SudachiPy>=0.6",
        "sudachidict_core",
    )
)

base_image = base_deps.add_local_dir("tools", "/root/repo/tools")

train_image = (
    base_deps
    .pip_install(
        "torch",
        "transformers>=4.48,<5",
        "tokenizers>=0.21,<0.23",
        "sentencepiece>=0.2",
        "protobuf>=4.25",
        "safetensors>=0.4",
        "accelerate>=0.28",
    )
    .add_local_dir("tools", "/root/repo/tools")
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
training_data = modal.Volume.from_name("mozc-training-data", create_if_missing=True)


@app.function(
    image=base_image,
    timeout=2 * 60 * 60,
    volumes={
        "/data": training_data,
        "/root/.cache/huggingface": hf_cache,
    },
)
def bootstrap_public_data(
    out_dir: str = "/data/sarashina_jev_public_proxy",
    scan_articles: int = 6000,
    example_articles: int = 6000,
    max_examples: int = 6000,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    from tools.sarashina_jev.bootstrap_public import build_public_proxy

    meta = build_public_proxy(
        out_dir,
        scan_articles=scan_articles,
        example_articles=example_articles,
        max_examples=max_examples,
        eval_ratio=0.1,
    )
    training_data.commit()
    hf_cache.commit()
    print("BOOTSTRAP_DONE", meta, flush=True)
    return {
        "train_path": f"{out_dir}/train.jsonl",
        "eval_path": f"{out_dir}/eval.jsonl",
        "meta": meta,
    }


@app.function(
    image=train_image,
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
    training_data.reload()

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"training data not found: {train_path}")
    if eval_path and not os.path.isfile(eval_path):
        raise FileNotFoundError(f"eval data not found: {eval_path}")

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
    bootstrap: bool = False,
    bootstrap_scan_articles: int = 6000,
    bootstrap_example_articles: int = 6000,
    bootstrap_max_examples: int = 6000,
):
    if bootstrap:
        staged = bootstrap_public_data.remote(
            scan_articles=bootstrap_scan_articles,
            example_articles=bootstrap_example_articles,
            max_examples=bootstrap_max_examples,
        )
        train_path = staged["train_path"]
        eval_path = staged["eval_path"]
        print(
            f"USING_BOOTSTRAP train={train_path} eval={eval_path} "
            f"rows={staged['meta']['train_rows']}/{staged['meta']['eval_rows']}",
            flush=True,
        )

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
    print(f"SPAWNED function_call_id={call.object_id} out={out}", flush=True)
