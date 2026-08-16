"""Modal で リランカー学習を回す（投げて離席できる）。

途中チェックポイント:
  - 既定で --save-every 200（checkpoint_latest.pt を Volume に commit）
  - 中断後は同じ --out で投げ直すと --auto-resume で続きから

使い方（リポジトリのルートで）:
    modal run --detach scripts/modal_train.py \\
      --train-path data/rerank_ctx/train_v2.jsonl \\
      --eval-path data/rerank_ctx/eval_unseen_v2.jsonl \\
      --model sbintuitions/modernbert-ja-70m \\
      --out /artifacts/trackA_v2_modernbert70m \\
      --epochs 2 --batch-size 512

Track B:
    modal run --detach scripts/modal_train.py \\
      --init-from /current_ce/cross_encoder.pt \\
      --model sbintuitions/modernbert-ja-70m \\
      --out /artifacts/trackB_v2_continue \\
      --train-path data/rerank_ctx/train_v2.jsonl \\
      --eval-path data/rerank_ctx/eval_unseen_v2.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import modal

app = modal.App("mozc-reranker")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48,<5",
        "tokenizers>=0.21",
        "sentencepiece",
        "accelerate",
    )
    .add_local_dir("tools", "/root/repo/tools")
    .add_local_dir("data/public/rerank_ctx", "/root/repo/data/rerank_ctx")
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def train(
    train_path: str = "data/rerank_ctx/train_v2.jsonl",
    eval_path: str = "data/rerank_ctx/eval_unseen_v2.jsonl",
    model: str = "sbintuitions/modernbert-ja-70m",
    out: str = "/artifacts/trackA_v2_modernbert70m",
    epochs: int = 2,
    batch_size: int = 512,
    init_from: str = "",
    max_len: int = 128,
    save_every: int = 200,
    auto_resume: bool = True,
):
    """Train in-process so mid-run Volume commits persist after cancel."""
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(train_path, eval_path, datasets=True)
    ensure_public_modal_paths(out, init_from)
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    from tools.rerank.train_cross_encoder import command_train

    def _commit(_path: Path | None = None) -> None:
        artifacts.commit()
        print(f"volume_committed path={_path}", flush=True)

    args = argparse.Namespace(
        train=train_path,
        eval=eval_path,
        model=model,
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        max_neg=15,
        lr=2e-5,
        num_workers=2,
        log_every=20,
        save_every=save_every,
        fp16=True,
        grad_checkpointing=True,
        require_cuda=True,
        require_gold_in_nbest=True,
        init_ckpt=init_from or "",
        resume="",
        auto_resume=auto_resume,
        on_checkpoint=_commit,
    )
    print(
        "RUN in-process "
        f"model={model} out={out} save_every={save_every} "
        f"auto_resume={auto_resume} init_from={init_from or '-'}",
        flush=True,
    )
    rc = command_train(args)
    artifacts.commit()
    print("DONE ->", out, "rc=", rc, flush=True)
    if rc != 0:
        raise RuntimeError(f"train failed rc={rc}")


@app.local_entrypoint()
def main(
    epochs: int = 2,
    batch_size: int = 512,
    model: str = "sbintuitions/modernbert-ja-70m",
    out: str = "/artifacts/trackA_v2_modernbert70m",
    train_path: str = "data/rerank_ctx/train_v2.jsonl",
    eval_path: str = "data/rerank_ctx/eval_unseen_v2.jsonl",
    init_from: str = "",
    max_len: int = 128,
    save_every: int = 200,
    auto_resume: bool = True,
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(train_path, eval_path, datasets=True)
    ensure_public_modal_paths(out, init_from)
    # spawn() keeps the GPU job alive after local client exits (remote()+detach
    # was cancelled when the WSL modal client received SIGTERM).
    call = train.spawn(
        train_path=train_path,
        eval_path=eval_path,
        model=model,
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        init_from=init_from,
        max_len=max_len,
        save_every=save_every,
        auto_resume=auto_resume,
    )
    print(f"SPAWNED function_call_id={call.object_id} out={out}", flush=True)
