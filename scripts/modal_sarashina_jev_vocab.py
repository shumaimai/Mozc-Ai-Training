"""Modal runner for Sarashina-JEV vocabulary-pruning ablation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("mozc-sarashina-jev-vocab")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "tokenizers==0.19.1",
        "sentencepiece>=0.2",
        "protobuf>=4.25",
        "safetensors>=0.4",
        "numpy<2",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
    )
    .add_local_dir("tools", "/root/repo/tools")
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
training_data = modal.Volume.from_name("mozc-training-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/repo"
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={
        "/artifacts": artifacts,
        "/data": training_data,
        "/root/.cache/huggingface": hf_cache,
    },
)
def run_vocab_ablation(
    artifact: str,
    train_path: str,
    eval_path: str,
    vocab_size: int,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    artifacts.reload()
    training_data.reload()

    out = f"/artifacts/sarashina_jev/vocab_ablation/{vocab_size}"
    onnx_out = f"{out}/onnx_mixed"

    _run(
        [
            sys.executable,
            "-m",
            "tools.sarashina_jev.prune_vocab",
            "--artifact",
            artifact,
            "--out",
            out,
            "--vocab-size",
            str(vocab_size),
            "--probe-data",
            eval_path,
            "--probe-limit",
            "204",
        ]
    )
    artifacts.commit()

    _run(
        [
            sys.executable,
            "-m",
            "tools.sarashina_jev.export_int8",
            "--artifact",
            out,
            "--eval",
            eval_path,
            "--calib",
            train_path,
            "--out",
            onnx_out,
            "--calib-pages",
            "384",
            "--page-size",
            "5",
            "--max-length",
            "128",
            "--threads",
            "8",
        ]
    )

    prune_report = json.loads(
        (Path(out) / "vocab_prune_report.json").read_text(encoding="utf-8")
    )
    int8_report = json.loads(
        (Path(onnx_out) / "int8_report.json").read_text(encoding="utf-8")
    )
    mixed = (int8_report.get("models") or {}).get(
        "int8_dynamic_fp16_embedding",
        {},
    )
    result = {
        "vocab_size": vocab_size,
        "artifact": out,
        "mixed_model": {
            "hit1": mixed.get("hit1"),
            "size_mib": mixed.get("size_mib"),
            "p50_ms": (mixed.get("latency_ms") or {}).get("p50"),
        },
        "tokenizer_probe": prune_report.get("tokenizer_probe"),
        "embedding": prune_report.get("embedding"),
        "report_path": f"{onnx_out}/int8_report.json",
    }
    (Path(out) / "vocab_ablation_report.json").write_text(
        json.dumps(
            {
                "summary": result,
                "prune": prune_report,
                "quantization": int8_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.commit()
    print("VOCAB_ABLATION_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main(
    artifact: str = "/artifacts/sarashina_jev/int8_recovery/qat_v2_repro",
    train_path: str = "/data/sarashina_jev_public_proxy/train.jsonl",
    eval_path: str = "/data/sarashina_jev_public_proxy/eval.jsonl",
    vocab_size: int = 64000,
):
    call = run_vocab_ablation.spawn(
        artifact=artifact,
        train_path=train_path,
        eval_path=eval_path,
        vocab_size=vocab_size,
    )
    print(
        f"SPAWNED vocab_ablation_call_id={call.object_id} "
        f"artifact={artifact} vocab={vocab_size}",
        flush=True,
    )
