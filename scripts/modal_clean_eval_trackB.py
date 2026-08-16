#!/usr/bin/env python3
"""Modal T4: Track B clean-eval at fixed τ=2.5 (does not touch shippable τ2.0).

Usage (WSL, repo root):
  export PATH="$HOME/.local/bin:$PATH"
  modal run --detach scripts/modal_clean_eval_trackB.py

Outputs on Volume /artifacts/eval/:
  clean_eval_trackB_tau2.5.json
  clean_eval_trackB_fixed_tau_sweep.json
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mozc-clean-eval-trackB")

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
        "scripts/_clean_eval_trackB_worker.py",
        "/root/repo/scripts/_clean_eval_trackB_worker.py",
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
def run_clean_eval(
    ckpt: str = "/artifacts/trackB_v2_continue",
    sets: str = SETS_CLEAN,
    taus: str = "2.0,2.5",
    report_tau: float = 2.5,
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
    tag: str = "trackB_v2_clean_eval",
    out_prefix: str = "clean_eval_trackB",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    os.chdir("/root/repo")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("scripts").mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "scripts/_clean_eval_trackB_worker.py",
        "--ckpt",
        ckpt,
        "--sets",
        sets,
        "--taus",
        taus,
        "--report-tau",
        str(report_tau),
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
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    artifacts.commit()
    tau_label = "2.5" if abs(float(report_tau) - 2.5) < 1e-9 else f"{report_tau:g}"
    out_path = Path(out_dir) / f"{out_prefix}_tau{tau_label}.json"
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    print("RESULT:", json.dumps(payload.get("summary_table"), ensure_ascii=False, indent=2), flush=True)
    return payload.get("summary_table")


@app.local_entrypoint()
def main(
    ckpt: str = "/artifacts/trackB_v2_continue",
    sets: str = SETS_CLEAN,
    taus: str = "2.0,2.5",
    report_tau: float = 2.5,
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
    tag: str = "trackB_v2_clean_eval",
    out_prefix: str = "clean_eval_trackB",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    call = run_clean_eval.spawn(
        ckpt=ckpt,
        sets=sets,
        taus=taus,
        report_tau=report_tau,
        out_dir=out_dir,
        batch_size=batch_size,
        max_len=max_len,
        tag=tag,
        out_prefix=out_prefix,
    )
    print(f"SPAWNED function_call_id={call.object_id} ckpt={ckpt} report_tau={report_tau}", flush=True)
