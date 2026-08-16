"""Modal で contextual 評価を回す（文脈あり/なし × seen/unseen/fresh）。

使い方（リポジトリのルートで）:
    modal run --detach scripts/modal_eval.py \\
      --ckpt /artifacts/trackA_v2_modernbert70m \\
      --sets data/rerank_ctx/eval_seen_v2.jsonl,data/rerank_ctx/eval_unseen_v2.jsonl,data/rerank_ctx/eval_fresh_v2.jsonl \\
      --tag trackA_v2
    # NOTE: --ckpt must be the checkpoint DIRECTORY (contains cross_encoder.pt).

回収:
    modal volume get mozc-artifacts /eval ./artifacts/rerank_ctx/eval
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("mozc-reranker-eval")

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
)

artifacts = modal.Volume.from_name("mozc-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # 評価は学習の 1/3 以下 → T4 で十分
    timeout=2 * 60 * 60,
    volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache},
)
def evaluate(
    ckpt: str,
    sets: str,
    tag: str = "model",
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
    tau_sweep: str = "0,0.5,1,1.5,2,2.5,3,4,5",
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    os.chdir("/root/repo")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    set_paths = [s.strip() for s in sets.split(",") if s.strip()]
    summaries: list[dict] = []

    for data_path in set_paths:
        set_name = Path(data_path).stem  # e.g. eval_seen_v2
        out_json = str(Path(out_dir) / f"eval_{tag}_{set_name}.json")
        cmd = [
            "python",
            "-m",
            "tools.rerank.eval_contextual",
            "--data",
            data_path,
            "--ckpt",
            ckpt,
            "--out",
            out_json,
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
            "--max-len",
            str(max_len),
            "--tau-sweep",
            tau_sweep,
            "--model-name",
            tag,
        ]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        with open(out_json, encoding="utf-8") as f:
            summaries.append(json.load(f))

    # Compact rollup for decision (§5.1 primary = context_sensitive delta)
    rollup = {
        "tag": tag,
        "ckpt": ckpt,
        "primary_subset": "context_sensitive",
        "sets": [],
    }
    for s in summaries:
        rollup["sets"].append(
            {
                "data": s.get("data"),
                "n_groups": s.get("n_groups"),
                "n_context_sensitive": s.get("n_context_sensitive"),
                "tau_with": (s.get("with_context") or {}).get("tau"),
                "tau_blank": (s.get("blank_context") or {}).get("tau"),
                "hit1_ctx_on_cs": ((s.get("with_context") or {}).get("context_sensitive") or {}).get(
                    "final_hit1"
                ),
                "hit1_ctx_off_cs": (
                    (s.get("blank_context") or {}).get("context_sensitive") or {}
                ).get("final_hit1"),
                "context_delta_pt": s.get("context_delta_pt"),
                "hit1_all_ctx_on": ((s.get("with_context") or {}).get("all") or {}).get(
                    "final_hit1"
                ),
            }
        )

    rollup_path = Path(out_dir) / f"summary_{tag}.json"
    rollup_path.write_text(json.dumps(rollup, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROLLUP:", json.dumps(rollup, ensure_ascii=False, indent=2), flush=True)

    artifacts.commit()
    print("DONE ->", out_dir, flush=True)
    return rollup


@app.local_entrypoint()
def main(
    ckpt: str = "/artifacts/trackA_v2_modernbert70m",
    sets: str = "data/rerank_ctx/eval_seen_v2.jsonl,data/rerank_ctx/eval_unseen_v2.jsonl,data/rerank_ctx/eval_fresh_v2.jsonl",
    tag: str = "trackA_v2",
    out_dir: str = "/artifacts/eval",
    batch_size: int = 1024,
    max_len: int = 128,
):
    from tools.rerank.privacy import ensure_public_modal_paths

    ensure_public_modal_paths(sets, datasets=True)
    ensure_public_modal_paths(ckpt, out_dir)
    call = evaluate.spawn(
        ckpt=ckpt,
        sets=sets,
        tag=tag,
        out_dir=out_dir,
        batch_size=batch_size,
        max_len=max_len,
    )
    print(f"SPAWNED function_call_id={call.object_id} tag={tag} ckpt={ckpt}", flush=True)
