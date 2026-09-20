"""Export Sarashina-JEV to ONNX and compare FP32 vs INT8 PTQ.

The main INT8 path is static QDQ over MatMul/Gemm/Gather so the large token
embedding (Gather) is quantized too. Dynamic INT8 is emitted as a diagnostic
fallback; it normally leaves more weights in float.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.sarashina_jev.data import ListwisePageDataset, build_pages, read_jsonl
from tools.sarashina_jev.model import SarashinaJevScorer


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def file_size_mb(path: Path) -> float:
    total = 0
    candidates = [path]
    candidates.extend(path.parent.glob(path.name + ".data*"))
    candidates.extend(path.parent.glob(path.stem + ".data*"))
    seen: set[Path] = set()
    for item in candidates:
        if item in seen or not item.exists() or not item.is_file():
            continue
        seen.add(item)
        total += item.stat().st_size
    return total / (1024 * 1024)


def make_dataset(data_path: str, tokenizer, *, page_size: int, max_length: int):
    rows = read_jsonl(data_path)
    pages = build_pages(rows, page_size=page_size, max_pages=6, anchor_weight=0.25)
    return ListwisePageDataset(
        pages,
        tokenizer,
        page_size=page_size,
        max_length=max_length,
        shuffle_gold_candidates=False,
    )


class CalibrationReader:
    def __init__(self, dataset, limit: int):
        from onnxruntime.quantization import CalibrationDataReader

        if not issubclass(type(self), CalibrationDataReader):
            # Registration by inheritance is not required by ORT; this keeps the
            # implementation simple while matching the get_next contract.
            pass
        self.dataset = dataset
        self.limit = min(limit, len(dataset))
        self.index = 0

    def get_next(self):
        if self.index >= self.limit:
            return None
        item = self.dataset[self.index]
        self.index += 1
        return {
            "input_ids": item["input_ids"].numpy().astype(np.int64),
            "attention_mask": item["attention_mask"].numpy().astype(np.int64),
        }

    def rewind(self):
        self.index = 0


def evaluate_onnx(
    model_path: Path,
    dataset,
    *,
    warmup: int = 10,
    intra_threads: int = 4,
) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    total = correct = first_gold = predicted_first = 0
    latencies: list[float] = []
    for i in range(len(dataset)):
        item = dataset[i]
        feed = {
            "input_ids": item["input_ids"].numpy().astype(np.int64),
            "attention_mask": item["attention_mask"].numpy().astype(np.int64),
        }
        t0 = time.perf_counter()
        scores = session.run(["scores"], feed)[0]
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            latencies.append(elapsed)

        valid = item["candidate_mask"].numpy().astype(bool)
        scores = np.asarray(scores)
        scores = np.where(valid, scores, -1e9)
        pred = int(np.argmax(scores))
        target = int(item["target"].item())

        total += 1
        correct += int(pred == target)
        first_gold += int(target == 0)
        predicted_first += int(pred == 0)

    return {
        "hit1": correct / total if total else 0.0,
        "candidate0_baseline": first_gold / total if total else 0.0,
        "predicted_candidate0_rate": predicted_first / total if total else 0.0,
        "pages": total,
        "latency_ms": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--calib-pages", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    artifact = Path(args.artifact)
    out = Path(args.out) if args.out else artifact / "onnx_int8"
    out.mkdir(parents=True, exist_ok=True)

    print(f"load artifact={artifact} as fp32 for ONNX export", flush=True)
    model, tokenizer, meta = SarashinaJevScorer.load_artifact(
        artifact,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.cpu().eval()

    eval_ds = make_dataset(
        args.eval,
        tokenizer,
        page_size=args.page_size,
        max_length=args.max_length,
    )
    calib_ds = make_dataset(
        args.calib,
        tokenizer,
        page_size=args.page_size,
        max_length=args.max_length,
    )
    if not eval_ds or not calib_ds:
        raise SystemExit("empty eval/calibration dataset")

    dummy = eval_ds[0]
    dummy_ids = dummy["input_ids"]
    dummy_mask = dummy["attention_mask"]
    if tuple(dummy_ids.shape) != (args.page_size, args.max_length):
        raise RuntimeError(f"unexpected dummy shape {tuple(dummy_ids.shape)}")

    fp32_path = out / "sarashina_jev_fp32.onnx"
    dynamic_path = out / "sarashina_jev_int8_dynamic.onnx"
    static_path = out / "sarashina_jev_int8_qdq.onnx"

    print(f"export fp32 -> {fp32_path}", flush=True)
    with torch.inference_mode():
        try:
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["scores"],
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["scores"],
                opset_version=args.opset,
                do_constant_folding=True,
            )

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    print(f"dynamic INT8 MatMul/Gemm -> {dynamic_path}", flush=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dynamic_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    print(
        f"static QDQ INT8 MatMul/Gemm/Gather calib_pages={args.calib_pages} -> {static_path}",
        flush=True,
    )
    reader = CalibrationReader(calib_ds, args.calib_pages)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(static_path),
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )

    tokenizer.save_pretrained(out / "tokenizer")

    reports = {}
    for name, path in [
        ("fp32", fp32_path),
        ("int8_dynamic", dynamic_path),
        ("int8_qdq", static_path),
    ]:
        print(f"evaluate {name} size={file_size_mb(path):.1f} MiB", flush=True)
        try:
            metrics = evaluate_onnx(
                path,
                eval_ds,
                warmup=args.warmup,
                intra_threads=args.threads,
            )
            metrics["size_mib"] = file_size_mb(path)
            reports[name] = metrics
            print(f"{name} {json.dumps(metrics)}", flush=True)
        except Exception as exc:
            reports[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "size_mib": file_size_mb(path),
            }
            print(f"{name} evaluation failed: {exc}", flush=True)

    fp_hit = reports.get("fp32", {}).get("hit1")
    for name in ("int8_dynamic", "int8_qdq"):
        hit = reports.get(name, {}).get("hit1")
        if fp_hit is not None and hit is not None:
            reports[name]["delta_hit1_vs_fp32"] = hit - fp_hit

    final = {
        "artifact": str(artifact),
        "source_meta": meta,
        "page_size": args.page_size,
        "max_length": args.max_length,
        "calib_pages": min(args.calib_pages, len(calib_ds)),
        "eval_pages": len(eval_ds),
        "models": reports,
        "preferred_if_accuracy_holds": "int8_qdq",
        "note": (
            "int8_qdq includes Gather so the token embedding can be quantized. "
            "If Hit@1 regresses materially, move to QAT/quantization-aware distillation."
        ),
    }
    (out / "int8_report.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
