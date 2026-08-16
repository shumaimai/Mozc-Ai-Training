#!/usr/bin/env python3
"""Modal: Track B shippable single-τ selection (no retraining).

Scores once per set × {with,blank} context, aggregates subset metrics at
fixed τ ∈ {1.5, 2.0, 2.5}, then selects:

  among τ where ALL 3 sets have overall regression < 2%,
  pick τ maximizing min(CS Δ on seen, unseen, fresh).

Writes under Volume /artifacts/eval/ (and local get path):
  shippable_trackB_tau{τ}.json
  shippable_trackB_selection.json

Usage (WSL, repo root):
  export PATH="$HOME/.local/bin:$PATH"
  modal run --detach scripts/modal_shippable_tau.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mozc-shippable-tau")

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
    .add_local_file(
        "scripts/_shippable_tau_worker.py",
        "/root/repo/scripts/_shippable_tau_worker.py",
    )
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SETS_DEFAULT = (
    "data/rerank_ctx/eval_seen_v2.jsonl,"
    "data/rerank_ctx/eval_unseen_v2.jsonl,"
    "data/rerank_ctx/eval_fresh_v2.jsonl"
)
SETS_CLEAN = (
    "data/rerank_ctx/eval_seen_v2_clean.jsonl,"
    "data/rerank_ctx/eval_unseen_v2_clean.jsonl,"
    "data/rerank_ctx/eval_fresh_v2_clean.jsonl"
)
TAUS_DEFAULT = "1.5,2.0,2.5"


@app.function(
    image=image,
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def run_shippable(
    ckpt: str = "/artifacts/trackB_v2_continue",
    sets: str = SETS_DEFAULT,
    taus: str = TAUS_DEFAULT,
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
    tag: str = "trackB_v2",
    out_prefix: str = "shippable_trackB",
    arch: str = "sbintuitions/modernbert-ja-70m",
    adopt_track: str = "B",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    os.chdir("/root/repo")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("scripts").mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "scripts/_shippable_tau_worker.py",
        "--ckpt",
        ckpt,
        "--sets",
        sets,
        "--taus",
        taus,
        "--out-dir",
        out_dir,
        "--batch-size",
        str(batch_size),
        "--max-len",
        str(max_len),
        "--device",
        "cuda",
        "--tag",
        tag,
        "--out-prefix",
        out_prefix,
        "--arch",
        arch,
        "--adopt-track",
        adopt_track,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    artifacts.commit()
    sel = Path(out_dir) / f"{out_prefix}_selection.json"
    payload = json.loads(sel.read_text(encoding="utf-8")) if sel.is_file() else {}
    print("SELECTION:", json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


@app.local_entrypoint()
def main(
    ckpt: str = "/artifacts/trackB_v2_continue",
    sets: str = SETS_DEFAULT,
    taus: str = TAUS_DEFAULT,
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
    tag: str = "trackB_v2",
    out_prefix: str = "shippable_trackB",
    arch: str = "sbintuitions/modernbert-ja-70m",
    adopt_track: str = "B",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    call = run_shippable.spawn(
        ckpt=ckpt,
        sets=sets,
        taus=taus,
        out_dir=out_dir,
        batch_size=batch_size,
        max_len=max_len,
        tag=tag,
        out_prefix=out_prefix,
        arch=arch,
        adopt_track=adopt_track,
    )
    print(f"SPAWNED function_call_id={call.object_id} ckpt={ckpt} prefix={out_prefix}", flush=True)
