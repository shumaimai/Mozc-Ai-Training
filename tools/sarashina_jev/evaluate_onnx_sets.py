"""Evaluate exported ONNX artifacts on multiple datasets in one CPU process."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from transformers import AutoTokenizer

from tools.sarashina_jev.export_int8 import evaluate_onnx, file_size_mb, make_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="name=onnx_path")
    p.add_argument("--tokenizer", action="append", required=True, help="name=tokenizer_dir")
    p.add_argument("--dataset", action="append", required=True, help="name=jsonl")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--page-size", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    models = dict(item.split("=", 1) for item in args.model)
    tokenizers = dict(item.split("=", 1) for item in args.tokenizer)
    datasets = dict(item.split("=", 1) for item in args.dataset)
    cpu_model = platform.processor() or ""
    try:
        cpu_model = Path("/proc/cpuinfo").read_text(errors="replace").split("model name\t: ", 1)[1].splitlines()[0]
    except (OSError, IndexError):
        pass
    import onnxruntime as ort

    report = {
        "cpu_model": cpu_model,
        "platform": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "threads": args.threads,
        "inter_op_threads": 1,
        "pid": os.getpid(),
        "models": {},
    }
    for name, model_path in models.items():
        tokenizer = AutoTokenizer.from_pretrained(tokenizers[name])
        model_report = {
            "onnx_path": model_path,
            "size_mib": file_size_mb(Path(model_path)),
            "datasets": {},
        }
        for dataset_name, dataset_path in datasets.items():
            ds = make_dataset(dataset_path, tokenizer, page_size=args.page_size, max_length=args.max_length)
            model_report["datasets"][dataset_name] = evaluate_onnx(
                Path(model_path), ds, warmup=args.warmup, intra_threads=args.threads
            )
        report["models"][name] = model_report
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
