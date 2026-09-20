"""End-to-end Sarashina-JEV INT8 recovery pipeline on Modal.

Stages:
1) Re-benchmark the original 8L artifact with all INT8 variants.
2) QAT + BF16-teacher distillation on L4 by default.
3) Export/evaluate INT8 on CPU.
4) If the accuracy gate is missed, run stronger refinement passes.
5) QAT v3 matches the best deployment path: weight-only Linear INT8 + INT8 embeddings.
6) Persist a single pipeline_report.json with the best compact model.

Run with:
    modal run --detach scripts/modal_sarashina_jev_qat_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("mozc-sarashina-jev-qat-pipeline")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
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

export_image = (
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

orchestrator_image = modal.Image.debian_slim(python_version="3.11")

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
    image=train_image,
    gpu="L4",
    timeout=8 * 60 * 60,
    volumes={
        "/artifacts": artifacts,
        "/data": training_data,
        "/root/.cache/huggingface": hf_cache,
    },
)
def qat_train(
    teacher_artifact: str,
    student_artifact: str,
    out: str,
    train_path: str,
    eval_path: str,
    epochs: int,
    limit: int,
    backbone_lr: float,
    embedding_lr: float,
    head_lr: float,
    kd_weight: float,
    mse_weight: float,
    margin_weight: float,
    quant_start: float,
    activation_quantization: bool,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    artifacts.reload()
    training_data.reload()

    cmd = [
        sys.executable,
        "-m",
        "tools.sarashina_jev.train_qat",
        "--teacher-artifact",
        teacher_artifact,
        "--student-artifact",
        student_artifact,
        "--train",
        train_path,
        "--eval",
        eval_path,
        "--out",
        out,
        "--epochs",
        str(epochs),
        "--limit",
        str(limit),
        "--batch-size",
        "2",
        "--grad-accum",
        "4",
        "--train-last-n-layers",
        "4",
        "--backbone-lr",
        str(backbone_lr),
        "--embedding-lr",
        str(embedding_lr),
        "--head-lr",
        str(head_lr),
        "--kd-weight",
        str(kd_weight),
        "--mse-weight",
        str(mse_weight),
        "--margin-weight",
        str(margin_weight),
        "--temperature",
        "2.0",
        "--quant-start",
        str(quant_start),
        "--quant-ramp-fraction",
        "0.35",
    ]
    if activation_quantization:
        cmd.append("--activation-quantization")
    _run(cmd)
    artifacts.commit()
    hf_cache.commit()
    return {"out": out}


@app.function(
    image=export_image,
    cpu=8.0,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={
        "/artifacts": artifacts,
        "/data": training_data,
        "/root/.cache/huggingface": hf_cache,
    },
)
def export_eval(
    artifact: str,
    train_path: str,
    eval_path: str,
    out: str,
    calib_pages: int = 256,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    artifacts.reload()
    training_data.reload()

    cmd = [
        sys.executable,
        "-m",
        "tools.sarashina_jev.export_int8",
        "--artifact",
        artifact,
        "--eval",
        eval_path,
        "--calib",
        train_path,
        "--out",
        out,
        "--calib-pages",
        str(calib_pages),
        "--page-size",
        "5",
        "--max-length",
        "128",
        "--threads",
        "8",
    ]
    _run(cmd)
    report_path = Path(out) / "int8_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts.commit()
    return report


def _best_compact(report: dict, max_size_mib: float = 400.0) -> dict:
    choices = []
    for name, metrics in (report.get("models") or {}).items():
        if not isinstance(metrics, dict) or "hit1" not in metrics:
            continue
        size = float(metrics.get("size_mib", 1e9))
        if name == "fp32":
            continue
        choices.append(
            {
                "name": name,
                "hit1": float(metrics["hit1"]),
                "size_mib": size,
                "p50_ms": float((metrics.get("latency_ms") or {}).get("p50", 0.0)),
            }
        )
    compact = [x for x in choices if x["size_mib"] <= max_size_mib]
    pool = compact or choices
    if not pool:
        return {"name": "none", "hit1": 0.0, "size_mib": 0.0, "p50_ms": 0.0}
    return max(pool, key=lambda x: (x["hit1"], -x["size_mib"]))


@app.function(
    image=orchestrator_image,
    cpu=1.0,
    memory=1024,
    timeout=12 * 60 * 60,
    volumes={"/artifacts": artifacts},
)
def pipeline(
    teacher_artifact: str = "/artifacts/sarashina_jev/8l_public_quick_v1",
    train_path: str = "/data/sarashina_jev_public_proxy/train.jsonl",
    eval_path: str = "/data/sarashina_jev_public_proxy/eval.jsonl",
    limit: int = 1500,
    accuracy_gate: float = 0.82,
):
    root = "/artifacts/sarashina_jev/int8_recovery"
    report: dict = {
        "teacher_artifact": teacher_artifact,
        "accuracy_gate": accuracy_gate,
        "stages": {},
    }

    # Stage 0: try less destructive PTQ choices first.
    baseline_out = f"{root}/baseline_ptq"
    print("PIPELINE stage=baseline_ptq", flush=True)
    baseline = export_eval.remote(
        teacher_artifact,
        train_path,
        eval_path,
        baseline_out,
        256,
    )
    baseline_best = _best_compact(baseline)
    report["stages"]["baseline_ptq"] = {
        "best": baseline_best,
        "report": baseline,
    }
    print(f"PIPELINE baseline_best={baseline_best}", flush=True)

    # Stage 1: balanced QAT/distillation.
    qat1 = f"{root}/qat_v1"
    print("PIPELINE stage=qat_v1", flush=True)
    qat_train.remote(
        teacher_artifact,
        teacher_artifact,
        qat1,
        train_path,
        eval_path,
        3,
        limit,
        5e-6,
        1e-5,
        2e-4,
        0.7,
        0.05,
        0.0,
        0.25,
        True,
    )
    qat1_report = export_eval.remote(
        qat1,
        train_path,
        eval_path,
        f"{qat1}/onnx_int8",
        256,
    )
    qat1_best = _best_compact(qat1_report)
    report["stages"]["qat_v1"] = {
        "best": qat1_best,
        "report": qat1_report,
    }
    print(f"PIPELINE qat_v1_best={qat1_best}", flush=True)

    candidates = [
        ("baseline_ptq", baseline_best),
        ("qat_v1", qat1_best),
    ]

    # Stage 2: stronger full-strength refinement only if compact INT8 is still weak.
    if qat1_best["hit1"] < accuracy_gate:
        qat2 = f"{root}/qat_v2"
        print(
            f"PIPELINE stage=qat_v2 reason=hit1<{accuracy_gate}",
            flush=True,
        )
        qat_train.remote(
            teacher_artifact,
            qat1,
            qat2,
            train_path,
            eval_path,
            2,
            limit,
            2e-6,
            5e-6,
            1e-4,
            1.2,
            0.10,
            0.0,
            1.0,
            True,
        )
        qat2_report = export_eval.remote(
            qat2,
            train_path,
            eval_path,
            f"{qat2}/onnx_int8",
            384,
        )
        qat2_best = _best_compact(qat2_report)
        report["stages"]["qat_v2"] = {
            "best": qat2_best,
            "report": qat2_report,
        }
        candidates.append(("qat_v2", qat2_best))
        print(f"PIPELINE qat_v2_best={qat2_best}", flush=True)

        # Stage 3: deployment-matched QAT. Dynamic Gather INT8 is weight-focused,
        # so do not fake-quantize Linear activations here. Distill teacher margins
        # explicitly and continue from the strongest QAT v2 checkpoint.
        if qat2_best["hit1"] < accuracy_gate:
            qat3 = f"{root}/qat_v3_weight_only"
            print(
                f"PIPELINE stage=qat_v3_weight_only reason=hit1<{accuracy_gate}",
                flush=True,
            )
            qat_train.remote(
                teacher_artifact,
                qat2,
                qat3,
                train_path,
                eval_path,
                3,
                limit,
                1e-6,
                5e-6,
                5e-5,
                1.5,
                0.10,
                0.25,
                1.0,
                False,
            )
            qat3_report = export_eval.remote(
                qat3,
                train_path,
                eval_path,
                f"{qat3}/onnx_int8",
                512,
            )
            qat3_best = _best_compact(qat3_report)
            report["stages"]["qat_v3_weight_only"] = {
                "best": qat3_best,
                "report": qat3_report,
            }
            candidates.append(("qat_v3_weight_only", qat3_best))
            print(f"PIPELINE qat_v3_best={qat3_best}", flush=True)

    best_stage, best = max(
        candidates,
        key=lambda item: (item[1]["hit1"], -item[1]["size_mib"]),
    )
    report["best"] = {
        "stage": best_stage,
        **best,
        "passed_accuracy_gate": best["hit1"] >= accuracy_gate,
    }
    report["next_step"] = (
        "vocab_32k_ablation"
        if best["hit1"] >= accuracy_gate
        else "QAT needs more recovery before vocabulary pruning"
    )

    artifacts.reload()
    report_path = Path(root) / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts.commit()
    print("PIPELINE_DONE", json.dumps(report["best"]), flush=True)
    return report["best"]


@app.local_entrypoint()
def main(
    teacher_artifact: str = "/artifacts/sarashina_jev/8l_public_quick_v1",
    train_path: str = "/data/sarashina_jev_public_proxy/train.jsonl",
    eval_path: str = "/data/sarashina_jev_public_proxy/eval.jsonl",
    limit: int = 1500,
    accuracy_gate: float = 0.82,
):
    call = pipeline.spawn(
        teacher_artifact=teacher_artifact,
        train_path=train_path,
        eval_path=eval_path,
        limit=limit,
        accuracy_gate=accuracy_gate,
    )
    print(
        f"SPAWNED pipeline_call_id={call.object_id} "
        f"teacher={teacher_artifact} gate={accuracy_gate}",
        flush=True,
    )
