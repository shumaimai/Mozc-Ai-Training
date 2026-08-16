#!/usr/bin/env python3
"""Modal T4: context-cap CS Δ sweep (50/30/20) at fixed τ."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mozc-ctx-cap-sweep")

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
        "scripts/_ctx_cap_quality_worker.py",
        "/root/repo/scripts/_ctx_cap_quality_worker.py",
    )
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SETS_CLEAN = (
    "data/rerank_ctx/eval_seen_v2_clean.jsonl,"
    "data/rerank_ctx/eval_unseen_v2_clean.jsonl,"
    "data/rerank_ctx/eval_fresh_v2_clean.jsonl"
)


@app.function(
    image=image,
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def run_sweep(
    ckpt: str = "/artifacts/track30m_ctx",
    tag: str = "30m",
    tau: float = 2.5,
    caps: str = "50,30,20",
    sets: str = SETS_CLEAN,
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    os.chdir("/root/repo")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("scripts").mkdir(parents=True, exist_ok=True)
    out = f"{out_dir}/ctx_cap_quality_{tag}.json"
    cmd = [
        "python",
        "scripts/_ctx_cap_quality_worker.py",
        "--ckpt",
        ckpt,
        "--sets",
        sets,
        "--caps",
        caps,
        "--tau",
        str(tau),
        "--out",
        out,
        "--batch-size",
        str(batch_size),
        "--max-len",
        str(max_len),
        "--device",
        "cuda",
        "--tag",
        tag,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    artifacts.commit()
    payload = json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).is_file() else {}
    print("SUMMARY:", json.dumps(payload.get("summary_rows"), ensure_ascii=False, indent=2), flush=True)
    return payload.get("summary_rows")


@app.local_entrypoint()
def main(
    ckpt: str = "/artifacts/track30m_ctx",
    tag: str = "30m",
    tau: float = 2.5,
    caps: str = "50,30,20",
    sets: str = SETS_CLEAN,
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt)
    call = run_sweep.spawn(ckpt=ckpt, tag=tag, tau=tau, caps=caps, sets=sets)
    print(f"SPAWNED function_call_id={call.object_id} ckpt={ckpt} tag={tag}", flush=True)
