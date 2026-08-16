"""Export Track B to ONNX fp32 on Modal CPU, then parity + hook-offline eval.

Usage (repo root, WSL):
  modal run --detach scripts/modal_export_onnx.py

Outputs on Volume mozc-artifacts:
  /trackB_v2_continue/onnx/cross_encoder_fp32.onnx
  /trackB_v2_continue/onnx/tokenizer/
  /trackB_v2_continue/onnx/margin_policy.json
  /trackB_v2_continue/onnx/parity_scores.json
  /eval/hook_offline_tau2.5.json
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("mozc-reranker-onnx-export")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48,<5",
        "tokenizers>=0.21",
        "sentencepiece",
        "accelerate",
        "onnx",
        "onnxruntime",
        "numpy",
    )
    .add_local_dir("tools", "/root/repo/tools")
    .add_local_dir("data/public/rerank_ctx", "/root/repo/data/rerank_ctx")
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=8.0,
    timeout=3 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def export_and_check(
    ckpt: str = "/artifacts/trackB_v2_continue",
    out: str = "/artifacts/trackB_v2_continue/onnx",
    data: str = "data/rerank_ctx/eval_unseen_v2_clean.jsonl",
    tau: float = 2.5,
    hook_out: str = "/artifacts/eval/hook_offline_tau2.5.json",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(data, datasets=True)
    ensure_public_modal_paths(ckpt, out, hook_out)
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    Path(out).mkdir(parents=True, exist_ok=True)

    def run(cmd: list[str]) -> None:
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    run(
        [
            "python",
            "-m",
            "tools.rerank.export_onnx",
            "--ckpt",
            ckpt,
            "--out",
            out,
            "--fp32",
            "--max-len",
            "128",
            "--tau",
            str(tau),
            "--cand-cap",
            "30",
        ]
    )
    artifacts.commit()

    run(
        [
            "python",
            "-m",
            "tools.rerank.parity_check",
            "--ckpt",
            ckpt,
            "--onnx-dir",
            out,
            "--data",
            data,
            "--n-groups",
            "80",
            "--max-len",
            "128",
            "--fp32-only",
            "--out",
            f"{out}/parity_scores.json",
        ]
    )
    artifacts.commit()

    run(
        [
            "python",
            "-m",
            "tools.rerank.eval_hook_offline",
            "--onnx",
            f"{out}/cross_encoder_fp32.onnx",
            "--tokenizer",
            f"{out}/tokenizer",
            "--data-dir",
            "data/rerank_ctx",
            "--out",
            hook_out,
            "--tau",
            str(tau),
            "--max-len",
            "128",
            "--cand-cap",
            "30",
            "--intra-op",
            "1",
        ]
    )
    artifacts.commit()
    print("DONE export+parity+hook", flush=True)


@app.local_entrypoint()
def main(
    ckpt: str = "/artifacts/trackB_v2_continue",
    out: str = "/artifacts/trackB_v2_continue/onnx",
    data: str = "data/rerank_ctx/eval_unseen_v2_clean.jsonl",
    tau: float = 2.5,
    hook_out: str = "/artifacts/eval/hook_offline_tau2.5.json",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(data, datasets=True)
    ensure_public_modal_paths(ckpt, out, hook_out)
    # spawn() keeps the job alive after local client exits (see modal_train.py).
    call = export_and_check.spawn(
        ckpt=ckpt, out=out, data=data, tau=tau, hook_out=hook_out
    )
    print(f"SPAWNED function_call_id={call.object_id} out={out} tau={tau}", flush=True)
