"""Run Sarashina-JEV ONNX INT8 export/evaluation on Modal CPU.

Example:
    modal run --detach scripts/modal_sarashina_jev_int8.py \
      --artifact /artifacts/sarashina_jev/8l_public_quick_v1 \
      --eval-path /data/sarashina_jev_public_proxy/eval.jsonl \
      --calib-path /data/sarashina_jev_public_proxy/train.jsonl
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

app = modal.App("mozc-sarashina-jev-int8")

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


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=3 * 60 * 60,
    volumes={
        "/artifacts": artifacts,
        "/data": training_data,
        "/root/.cache/huggingface": hf_cache,
    },
)
def quantize_and_eval(
    artifact: str = "/artifacts/sarashina_jev/8l_public_quick_v1",
    eval_path: str = "/data/sarashina_jev_public_proxy/eval.jsonl",
    calib_path: str = "/data/sarashina_jev_public_proxy/train.jsonl",
    out: str = "",
    calib_pages: int = 128,
    page_size: int = 5,
    max_length: int = 128,
    threads: int = 4,
):
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    artifacts.reload()
    training_data.reload()

    if not os.path.isdir(artifact):
        raise FileNotFoundError(f"artifact not found: {artifact}")
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"eval data not found: {eval_path}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"calibration data not found: {calib_path}")

    out = out or f"{artifact}/onnx_int8"
    cmd = [
        sys.executable,
        "-m",
        "tools.sarashina_jev.export_int8",
        "--artifact",
        artifact,
        "--eval",
        eval_path,
        "--calib",
        calib_path,
        "--out",
        out,
        "--calib-pages",
        str(calib_pages),
        "--page-size",
        str(page_size),
        "--max-length",
        str(max_length),
        "--threads",
        str(threads),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/repo"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)

    artifacts.commit()
    print(f"DONE rc={proc.returncode} out={out}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"INT8 export/eval failed rc={proc.returncode}")
    return {"out": out, "returncode": proc.returncode}


@app.local_entrypoint()
def main(
    artifact: str = "/artifacts/sarashina_jev/8l_public_quick_v1",
    eval_path: str = "/data/sarashina_jev_public_proxy/eval.jsonl",
    calib_path: str = "/data/sarashina_jev_public_proxy/train.jsonl",
    out: str = "",
    calib_pages: int = 128,
    page_size: int = 5,
    max_length: int = 128,
    threads: int = 4,
):
    call = quantize_and_eval.spawn(
        artifact=artifact,
        eval_path=eval_path,
        calib_path=calib_path,
        out=out,
        calib_pages=calib_pages,
        page_size=page_size,
        max_length=max_length,
        threads=threads,
    )
    print(
        f"SPAWNED function_call_id={call.object_id} "
        f"artifact={artifact} out={out or artifact + '/onnx_int8'}",
        flush=True,
    )
