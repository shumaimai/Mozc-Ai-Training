"""64k-vocabulary 8L -> 6L distillation/QAT experiment.

All outputs are isolated under /artifacts/sarashina_jev/6l_64k_v1.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("mozc-sarashina-jev-6l-64k")

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "sudachipy>=0.6.8",
        "sudachidict_core>=20240109",
    )
    .add_local_dir("tools", "/root/repo/tools")
)

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

ROOT = "/artifacts/sarashina_jev/6l_64k_v1"
BASE = "/artifacts/sarashina_jev/vocab_ablation/64000"
TEACHER = BASE
TRAIN = "/data/sarashina_jev_public_proxy/train.jsonl"
EVAL = "/data/sarashina_jev_public_proxy/eval.jsonl"
HOLDOUT = ROOT + "/holdout_public/holdout.jsonl"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/repo"
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=_env(), check=False)
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(cmd)}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=data_image,
    cpu=4.0,
    memory=16384,
    timeout=6 * 60 * 60,
    volumes={"/artifacts": artifacts, "/data": training_data},
)
def generate_holdout(
    out_dir: str = ROOT + "/holdout_public",
    scan_articles: int = 1500,
    example_start_article: int = 1500,
    example_articles: int = 3000,
    max_examples: int = 800,
):
    artifacts.reload()
    training_data.reload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "tools.sarashina_jev.bootstrap_public",
            "--out",
            str(out),
            "--scan-articles",
            str(scan_articles),
            "--example-articles",
            str(example_articles),
            "--max-examples",
            str(max_examples),
            "--eval-ratio",
            "1.0",
            "--scan-start-article",
            "0",
            "--example-start-article",
            str(example_start_article),
        ]
    )
    eval_path = out / "eval.jsonl"
    holdout_path = out / "holdout.jsonl"
    holdout_path.write_bytes(eval_path.read_bytes())
    base_ids = set()
    for source in (Path(TRAIN), Path(EVAL)):
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                base_ids.add(str(json.loads(line).get("source_id", "")))
    holdout_ids = {
        str(json.loads(line).get("source_id", ""))
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = sorted(base_ids & holdout_ids)
    if overlap:
        raise RuntimeError(f"holdout source overlap detected: {overlap[:5]}")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "holdout_file": "holdout.jsonl",
            "split_policy": "articles [1500,4500), base uses articles [0,1500)",
            "base_source_id_count": len(base_ids),
            "holdout_source_id_count": len(holdout_ids),
            "source_id_overlap_count": len(overlap),
            "sha256": {"holdout.jsonl": _sha256(holdout_path)},
        }
    )
    (out / "holdout_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.commit()
    result = {"out": str(out), "rows": len(holdout_ids), "source_id_overlap_count": len(overlap), "meta": str(out / "holdout_meta.json")}
    print("HOLDOUT_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.function(
    image=export_image,
    cpu=8.0,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/artifacts": artifacts},
)
def make_6l_student(
    source_artifact: str = TEACHER,
    out: str = ROOT + "/student_6l_init",
):
    artifacts.reload()
    _run([sys.executable, "-m", "tools.sarashina_jev.prune_layers", "--artifact", source_artifact, "--out", out, "--keep-layers", "6"])
    artifacts.commit()
    return {"out": out, "report": out + "/layer_prune_report.json"}


@app.function(
    image=train_image,
    gpu="L4",
    timeout=10 * 60 * 60,
    volumes={"/artifacts": artifacts, "/data": training_data, "/root/.cache/huggingface": hf_cache},
)
def distill_6l(
    teacher_artifact: str = TEACHER,
    student_artifact: str = ROOT + "/student_6l_init",
    out: str = ROOT + "/student_6l_distilled",
):
    artifacts.reload()
    training_data.reload()
    _run(
        [sys.executable, "-m", "tools.sarashina_jev.train_distill",
         "--teacher-artifact", teacher_artifact, "--student-artifact", student_artifact,
         "--train", TRAIN, "--eval", EVAL, "--eval-holdout", HOLDOUT, "--out", out,
         "--epochs", "3", "--limit", "0", "--batch-size", "2", "--grad-accum", "4",
         "--backbone-lr", "3e-6", "--head-lr", "1e-4", "--kd-weight", "1.0",
         "--mse-weight", "0.10", "--margin-weight", "0.10", "--temperature", "2.0"]
    )
    artifacts.commit()
    return {"out": out}


@app.function(
    image=train_image,
    gpu="L4",
    timeout=10 * 60 * 60,
    volumes={"/artifacts": artifacts, "/data": training_data, "/root/.cache/huggingface": hf_cache},
)
def qat_refine_6l(
    teacher_artifact: str = TEACHER,
    student_artifact: str = ROOT + "/student_6l_distilled",
    out: str = ROOT + "/student_6l_qat_v2",
):
    artifacts.reload()
    training_data.reload()
    _run(
        [sys.executable, "-m", "tools.sarashina_jev.train_qat",
         "--teacher-artifact", teacher_artifact, "--student-artifact", student_artifact,
         "--train", TRAIN, "--eval", EVAL, "--out", out, "--epochs", "2", "--limit", "0",
         "--batch-size", "2", "--grad-accum", "4", "--train-last-n-layers", "4",
         "--backbone-lr", "2e-6", "--embedding-lr", "5e-6", "--head-lr", "1e-4",
         "--kd-weight", "1.2", "--mse-weight", "0.10", "--margin-weight", "0.0",
         "--temperature", "2.0", "--quant-start", "0.25", "--quant-ramp-fraction", "0.35",
         "--activation-quantization"]
    )
    artifacts.commit()
    return {"out": out}


@app.function(
    image=export_image,
    cpu=8.0,
    memory=32768,
    timeout=10 * 60 * 60,
    volumes={"/artifacts": artifacts, "/data": training_data},
)
def export_and_compare(
    teacher_artifact: str = TEACHER,
    student_artifact: str = ROOT + "/student_6l_qat_v2",
    out: str = ROOT + "/comparison",
):
    artifacts.reload()
    Path(out).mkdir(parents=True, exist_ok=True)
    teacher_out = f"{out}/teacher_8l_export"
    student_out = f"{out}/student_6l_export"
    for artifact, export_out in ((teacher_artifact, teacher_out), (student_artifact, student_out)):
        _run([sys.executable, "-m", "tools.sarashina_jev.export_int8", "--artifact", artifact,
              "--eval", EVAL, "--calib", TRAIN, "--out", export_out, "--calib-pages", "384",
              "--page-size", "5", "--max-length", "128", "--threads", "8"])
    _run([sys.executable, "-m", "tools.sarashina_jev.evaluate_onnx_sets",
          "--model", f"teacher_8l={teacher_out}/sarashina_jev_int8_dynamic_fp16_embedding.onnx",
          "--model", f"student_6l={student_out}/sarashina_jev_int8_dynamic_fp16_embedding.onnx",
          "--tokenizer", f"teacher_8l={teacher_out}/tokenizer",
          "--tokenizer", f"student_6l={student_out}/tokenizer",
          "--dataset", f"existing_eval={EVAL}", "--dataset", f"fresh_holdout={HOLDOUT}",
          "--threads", "8", "--warmup", "10", "--out", f"{out}/same_cpu_comparison.json"])
    report = json.loads((Path(out) / "same_cpu_comparison.json").read_text(encoding="utf-8"))
    report["artifacts"] = {
        "teacher": teacher_artifact,
        "student": student_artifact,
        "teacher_export_report": teacher_out + "/int8_report.json",
        "student_export_report": student_out + "/int8_report.json",
    }
    for key, artifact in (("teacher", teacher_artifact), ("student", student_artifact)):
        cfg = Path(artifact) / "jev_config.json"
        if cfg.exists():
            report.setdefault("artifact_metadata", {})[key] = json.loads(cfg.read_text(encoding="utf-8"))
    (Path(out) / "same_cpu_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.commit()
    print("COMPARE_DONE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


@app.function(
    image=export_image,
    cpu=8.0,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/artifacts": artifacts, "/data": training_data},
)
def fresh_revalidate(
    out: str = ROOT + "/comparison/fresh_revalidation.json",
):
    """Re-run only the deployment model in a fresh CPU container.

    This complements same-container measurements because ORT INT8 kernels have
    previously shown CPU/container-dependent numerical failures.
    """
    artifacts.reload()
    training_data.reload()
    comparison = Path(ROOT) / "comparison"
    _run(
        [sys.executable, "-m", "tools.sarashina_jev.evaluate_onnx_sets",
         "--model", f"teacher_8l={comparison}/teacher_8l_export/sarashina_jev_int8_dynamic_fp16_embedding.onnx",
         "--model", f"student_6l={comparison}/student_6l_export/sarashina_jev_int8_dynamic_fp16_embedding.onnx",
         "--tokenizer", f"teacher_8l={comparison}/teacher_8l_export/tokenizer",
         "--tokenizer", f"student_6l={comparison}/student_6l_export/tokenizer",
         "--dataset", f"existing_eval={EVAL}", "--dataset", f"fresh_holdout={HOLDOUT}",
         "--threads", "8", "--warmup", "10", "--out", out]
    )
    artifacts.commit()
    return json.loads(Path(out).read_text(encoding="utf-8"))


@app.function(image=orchestrator_image, cpu=1.0, memory=2048, timeout=24 * 60 * 60, volumes={"/artifacts": artifacts})
def experiment():
    holdout = generate_holdout.remote()
    student = make_6l_student.remote()
    distill = distill_6l.remote()
    qat = qat_refine_6l.remote()
    result = export_and_compare.remote()
    artifacts.reload()
    report = {
        "experiment": "64k_vocab_8l_to_6l_distillation_qat_v2",
        "holdout": holdout,
        "student": student,
        "distill": distill,
        "qat": qat,
        "comparison": result,
    }
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    (Path(ROOT) / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.commit()
    print("EXPERIMENT_DONE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


@app.local_entrypoint()
def main():
    call = experiment.spawn()
    print(f"SPAWNED 6L experiment_call_id={call.object_id} root={ROOT}", flush=True)
