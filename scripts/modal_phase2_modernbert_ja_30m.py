"""Phase 2 baseline: ModernBERT-ja-30M on runtime-context-parity Dataset v2.

This job deliberately packages only the corrected train and validation splits.
``final_test`` is neither mounted nor accepted as an argument, so it cannot
participate in model selection by accident.

Run after the Phase 1 context-parity gate passes::

    source /home/hashiguchishuhei/Mozc-Ai-Training-jev/.venv/bin/activate
    modal run --detach scripts/modal_phase2_modernbert_ja_30m.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import modal


APP_NAME = "mozc-v2-phase2-modernbert-ja-30m"
DATA_ROOT = "data/contextual_ranking_v2_production_runtime_context/dataset"
TRAIN_PATH = f"{DATA_ROOT}/train.jsonl.gz"
VALIDATION_PATH = f"{DATA_ROOT}/validation.jsonl.gz"
MODEL = "sbintuitions/modernbert-ja-30m"
OUT = "/artifacts/phase2_dataset_v2_runtime_context_modernbert_ja_30m"

app = modal.App(APP_NAME)
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
    # This is the immutable corrected dataset directory, but final_test is
    # explicitly excluded from the image.  Only train + validation can reach
    # the training container.
    .add_local_dir(
        "data/public/contextual_ranking_v2_production_runtime_context/dataset",
        "/root/repo/data/contextual_ranking_v2_production_runtime_context/dataset",
        ignore=["final_test.jsonl.gz"],
    )
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _train_args(
    *, out: str, epochs: int, batch_size: int, max_len: int, save_every: int, auto_resume: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        train=TRAIN_PATH,
        # Training progress uses validation only. final_test is absent from
        # the image and cannot be supplied here.
        eval=VALIDATION_PATH,
        model=MODEL,
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
        eligibility_status="NEURAL_ELIGIBLE",
        init_ckpt="",
        resume="",
        auto_resume=auto_resume,
        on_checkpoint=None,
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def train_and_validate(
    out: str = OUT,
    epochs: int = 2,
    batch_size: int = 256,
    max_len: int = 128,
    save_every: int = 200,
    auto_resume: bool = True,
) -> None:
    os.chdir("/root/repo")
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")

    # Modal invokes the function outside the mounted repository directory.
    # Make the mounted ``tools`` package importable before using its privacy
    # guard or trainer modules.
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(TRAIN_PATH, VALIDATION_PATH, datasets=True)
    ensure_public_modal_paths(out)

    from tools.rerank.eval_cross_encoder import main as eval_main
    from tools.rerank.train_cross_encoder import command_train

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_commit(_path: Path | None = None) -> None:
        artifacts.commit()
        print(f"volume_committed path={_path}", flush=True)

    args = _train_args(
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        save_every=save_every,
        auto_resume=auto_resume,
    )
    args.on_checkpoint = checkpoint_commit
    run_manifest = {
        "phase": "2-modernbert-ja-30m-baseline",
        "dataset_contract": "phase1-runtime-context-parity-v1",
        "model": MODEL,
        "train_path": TRAIN_PATH,
        "validation_path": VALIDATION_PATH,
        "final_test_mounted": False,
        "training_eligibility_status": "NEURAL_ELIGIBLE",
        "require_gold_in_nbest": True,
        "candidate_cap": 30,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "gpu": "L4",
    }
    (out_dir / "phase2_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("RUN_MANIFEST", json.dumps(run_manifest, ensure_ascii=False), flush=True)
    rc = command_train(args)
    if rc != 0:
        raise RuntimeError(f"training failed rc={rc}")
    artifacts.commit()

    # Validation drives model selection. Write both the primary trainable
    # subset and ALL rows so protected/coverage-limited behavior remains
    # observable, while never loading final_test.
    for label, status in (("neural_eligible", "NEURAL_ELIGIBLE"), ("all", "")):
        eval_out = str(out_dir / f"validation_{label}_margin.json")
        argv = [
            "--data", TRAIN_PATH.replace("train.jsonl.gz", "validation.jsonl.gz"),
            "--ckpt", out,
            "--out", eval_out,
            "--device", "cuda",
            "--require-cuda",
            "--batch-size", "1024",
            "--max-len", str(max_len),
            "--cand-cap", "30",
            "--tau", "0",
            "--tau-sweep", "0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5",
        ]
        if status:
            argv.extend(["--eligibility-status", status])
        print("VALIDATE", label, " ".join(argv), flush=True)
        rc = eval_main(argv)
        if rc != 0:
            raise RuntimeError(f"validation failed label={label} rc={rc}")
        artifacts.commit()
    print(f"DONE -> {out}", flush=True)


@app.local_entrypoint()
def main(
    out: str = OUT,
    epochs: int = 2,
    batch_size: int = 256,
    max_len: int = 128,
    save_every: int = 200,
    auto_resume: bool = True,
) -> None:
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(TRAIN_PATH, VALIDATION_PATH, datasets=True)
    ensure_public_modal_paths(out)
    call = train_and_validate.spawn(
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        save_every=save_every,
        auto_resume=auto_resume,
    )
    print(f"SPAWNED function_call_id={call.object_id} out={out}", flush=True)
