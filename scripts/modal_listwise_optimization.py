"""Controlled public-data listwise experiments; excludes final_test and user data."""
from pathlib import Path
import modal

app = modal.App("mozc-listwise-optimization-20260930")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.0", "transformers==4.57.6", "sentencepiece==0.2.2",
        "numpy==2.4.6", "onnx==1.23.0", "onnxruntime==1.30.0")
    .add_local_dir("tools", "/root/repo/tools", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("data/public/contextual_ranking_v2_production_runtime_context/dataset/train.jsonl.gz",
        "/root/repo/data/contextual_ranking_v2_production_runtime_context/dataset/train.jsonl.gz")
    .add_local_file("data/public/contextual_ranking_v2_production_runtime_context/dataset/validation.jsonl.gz",
        "/root/repo/data/contextual_ranking_v2_production_runtime_context/dataset/validation.jsonl.gz"))
artifacts = modal.Volume.from_name("mozc-artifacts")
cache = modal.Volume.from_name("hf-cache")


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=10800,
    volumes={"/artifacts":artifacts,"/root/.cache/huggingface":cache})
def experiment(variant: str = "continue"):
    import os,sys,json
    os.chdir("/root/repo")
    sys.path.insert(0,"/root/repo")
    from tools.rerank.privacy import ensure_public_modal_paths
    from tools.rerank.listwise import run_experiment
    train = "data/contextual_ranking_v2_production_runtime_context/dataset/train.jsonl.gz"
    val = train.replace("train.jsonl.gz","validation.jsonl.gz")
    out = f"/artifacts/optimization_20260930/listwise_{variant}"
    ensure_public_modal_paths(train,val,datasets=True)
    ensure_public_modal_paths(out)
    if Path(train.replace("train.jsonl.gz","final_test.jsonl.gz")).exists():
        raise RuntimeError("final_test must be excluded")
    options = {"continue":dict(lr=5e-6), "fresh":dict(lr=2e-5),
        "fresh_canonical":dict(lr=2e-5),
        "refine":dict(lr=2e-6,distill_weight=2.),
        "student7":dict(lr=1e-5,distill_weight=.5,layer_indices=[0,1,2,3,7,8,9])}
    if variant not in options:
        raise ValueError(variant)
    try:
        baseline = ("/artifacts/optimization_20260930/listwise_fresh/epoch1" if variant=="refine"
            else "/artifacts/phase2_dataset_v2_runtime_context_modernbert_ja_30m_format_v2")
        report = run_experiment(train,val,baseline,
            out,"fresh" if variant=="fresh_canonical" else variant,
            epochs=1 if variant=="refine" else 2,**options[variant])
        print("DONE",json.dumps(report),flush=True)
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main(variant: str = "continue"):
    experiment.remote(variant)
