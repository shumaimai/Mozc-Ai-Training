"""Export cross-encoder to ONNX fp32 + improved INT8 (static calib or dynamic).

Root cause found by parity_check: dynamic int8 collapsed score std (1.44→0.51)
and broke ranking. Prefer static QInt8 with calibration; keep dynamic as fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/rerank/modernbert70m_ce")
    p.add_argument("--out", default="")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--calib-data", default="data/rerank_v2/train.jsonl")
    p.add_argument("--calib-groups", type=int, default=64)
    p.add_argument(
        "--quant",
        choices=["static", "dynamic", "both", "none"],
        default="dynamic",
        help="dynamic MatMul/Gemm (preferred after experiments); static QDQ often collapses ModernBERT scores; none=fp32 only",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="export fp32 only (no int8). Alias for --quant none. Required for contextual ship.",
    )
    p.add_argument("--tau", type=float, default=2.5, help="margin_policy.json tau")
    p.add_argument("--cand-cap", type=int, default=30)
    args = p.parse_args(argv)
    if args.fp32:
        args.quant = "none"

    import numpy as np
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    from tools.dataset.jsonl import read_jsonl
    from tools.rerank.eval_cross_encoder import prepare_groups
    from tools.rerank.train_cross_encoder import build_pair_text

    ckpt = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt / "onnx"
    out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(ckpt / "cross_encoder.pt", map_location="cpu", weights_only=False)
    base = blob.get("base_model")
    if not base:
        base = json.loads((ckpt / "train_meta.json").read_text(encoding="utf-8"))["base_model"]

    tok_dir = ckpt / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_dir if tok_dir.exists() else base), trust_remote_code=True
    )

    class CrossEncoder(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                name, trust_remote_code=True, torch_dtype=torch.float32
            )
            hidden = int(self.encoder.config.hidden_size)
            self.score = nn.Linear(hidden, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            return self.score(cls).squeeze(-1)

    model = CrossEncoder(base)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()

    texts = ["読み: テスト [SEP] 候補: 試験", "読み: あ [SEP] 候補: 亜"]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_tensors="pt",
    )
    fp32_path = out / "cross_encoder_fp32.onnx"
    int8_dyn_path = out / "cross_encoder_int8_dynamic.onnx"
    int8_path = out / "cross_encoder_int8.onnx"  # preferred (static if available)

    print(f"exporting fp32 -> {fp32_path}", flush=True)
    with torch.inference_mode():
        try:
            torch.onnx.export(
                model,
                (enc["input_ids"], enc["attention_mask"]),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                (enc["input_ids"], enc["attention_mask"]),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                opset_version=args.opset,
                do_constant_folding=True,
            )

    if args.quant != "none":
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process

    quant_meta: dict[str, Any] = {"quant": args.quant}
    if args.quant != "none":
        pre_path = out / "cross_encoder_fp32_pre.onnx"
        print(f"quant_pre_process -> {pre_path}", flush=True)
        try:
            quant_pre_process(
                input_model_path=str(fp32_path),
                output_model_path=str(pre_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
            )
            quant_input = str(pre_path)
        except Exception as exc:
            print(f"quant_pre_process failed ({exc}); use raw fp32", flush=True)
            quant_input = str(fp32_path)
        quant_meta["quant_input"] = quant_input

    if args.quant in ("dynamic", "both"):
        print(f"quantize dynamic (MatMul/Gemm only) -> {int8_dyn_path}", flush=True)
        quantize_dynamic(
            model_input=quant_input,
            model_output=str(int8_dyn_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        quant_meta["dynamic"] = str(int8_dyn_path)
        # Prefer dynamic over naive static for this encoder (static collapsed worse).
        int8_path.write_bytes(int8_dyn_path.read_bytes())
        quant_meta["preferred_int8"] = "dynamic_matmul_gemm"
    if args.quant in ("static", "both"):
        # Build calibration batches from train groups.
        calib_rows = list(read_jsonl(Path(args.calib_data)))
        calib_groups = prepare_groups(calib_rows)[: max(8, args.calib_groups)]
        calib_batches: list[dict[str, np.ndarray]] = []
        for g in calib_groups:
            cands = g["candidates"][:8]
            ts = [build_pair_text(g["reading"], g["context_prev"], c) for c in cands]
            if not ts:
                continue
            e = tokenizer(
                ts,
                truncation=True,
                max_length=args.max_len,
                padding="max_length",
                return_tensors="np",
            )
            calib_batches.append(
                {
                    "input_ids": e["input_ids"].astype(np.int64),
                    "attention_mask": e["attention_mask"].astype(np.int64),
                }
            )
        print(f"static calib batches={len(calib_batches)}", flush=True)

        class Reader(CalibrationDataReader):
            def __init__(self, batches: list[dict[str, np.ndarray]]):
                self.batches = batches
                self.i = 0

            def get_next(self):
                if self.i >= len(self.batches):
                    return None
                b = self.batches[self.i]
                self.i += 1
                return b

        print(f"quantize static QDQ -> {out / 'cross_encoder_int8_static.onnx'}", flush=True)
        static_path = out / "cross_encoder_int8_static.onnx"
        try:
            quantize_static(
                model_input=quant_input,
                model_output=str(static_path),
                calibration_data_reader=Reader(calib_batches),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            quant_meta["static"] = str(static_path)
            # Do NOT overwrite preferred int8 with static unless explicitly static-only.
            if args.quant == "static":
                int8_path.write_bytes(static_path.read_bytes())
                quant_meta["preferred_int8"] = "static"
        except Exception as exc:
            print(f"static quant failed: {exc}", flush=True)
            quant_meta["static_error"] = str(exc)
            if args.quant == "static":
                if int8_dyn_path.exists():
                    int8_path.write_bytes(int8_dyn_path.read_bytes())
                    quant_meta["preferred_int8"] = "dynamic_fallback"
                else:
                    raise
    if args.quant == "dynamic" and int8_dyn_path.exists():
        int8_path.write_bytes(int8_dyn_path.read_bytes())
        quant_meta["preferred_int8"] = "dynamic_matmul_gemm"

    tokenizer.save_pretrained(out / "tokenizer")
    policy = {
        "tau": float(args.tau),
        "cand_cap": int(args.cand_cap),
        "max_len": int(args.max_len),
        "backend": "onnx_fp32",
        "int8": False,
        "top_k": None,
        "timeout_ms": 200,
        "context_clip_max_chars": 50,
        "base_model": base,
        "note": "Contextual Track B ship policy (NEXT_TASK_PHASE3_CTX). Do not use int8.",
    }
    (out / "margin_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "base_model": base,
        "max_len": args.max_len,
        "opset": args.opset,
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path) if args.quant != "none" else None,
        "margin_policy": str(out / "margin_policy.json"),
        "quantization": quant_meta,
        "calib_data": args.calib_data,
        "calib_groups": args.calib_groups,
        "note": (
            "Export fp32 matches PT. Naive dynamic/static int8 collapsed score std. "
            "Preferred path: dynamic MatMul/Gemm-only after quant_pre_process; "
            "if parity still fails, use ONNX fp32 for accuracy."
        ),
    }
    (out / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
