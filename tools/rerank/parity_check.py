"""Compare PyTorch fp32 vs ONNX fp32 vs ONNX int8 raw scores.

Example:
  python -m tools.rerank.parity_check \\
    --data data/rerank_v2/holdout.jsonl \\
    --ckpt artifacts/rerank/modernbert70m_ce \\
    --onnx-dir artifacts/rerank/modernbert70m_ce/onnx \\
    --n-groups 200 \\
    --out artifacts/rerank/parity_scores.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.train_cross_encoder import build_pair_text


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    deny = math.sqrt(sum((b - my) ** 2 for b in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def _stats(xs: list[float], ys: list[float], name: str) -> dict[str, Any]:
    diffs = [a - b for a, b in zip(xs, ys)]
    abs_diffs = [abs(d) for d in diffs]
    return {
        "pair": name,
        "n": len(xs),
        "spearman": round(_spearman(xs, ys), 6),
        "pearson": round(_pearson(xs, ys), 6),
        "mae": round(sum(abs_diffs) / max(1, len(abs_diffs)), 6),
        "rmse": round(math.sqrt(sum(d * d for d in diffs) / max(1, len(diffs))), 6),
        "mean_diff": round(sum(diffs) / max(1, len(diffs)), 6),
        "max_abs_diff": round(max(abs_diffs) if abs_diffs else 0.0, 6),
        "x_mean": round(sum(xs) / max(1, len(xs)), 6),
        "y_mean": round(sum(ys) / max(1, len(ys)), 6),
        "x_std": round(math.sqrt(sum((x - sum(xs)/len(xs))**2 for x in xs)/len(xs)), 6) if xs else 0.0,
        "y_std": round(math.sqrt(sum((y - sum(ys)/len(ys))**2 for y in ys)/len(ys)), 6) if ys else 0.0,
    }


def load_pt(ckpt: Path, max_len: int, device: str = "cpu"):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

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
    model.to(device)
    model.eval()
    return tokenizer, model, base, max_len, torch, device


def score_pt(
    tokenizer,
    model,
    torch,
    texts: list[str],
    max_len: int,
    chunk: int = 256,
    device: str = "cpu",
) -> list[float]:
    out: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), chunk):
            batch = texts[i : i + chunk]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            logits = model(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            )
            out.extend(float(x) for x in logits.detach().float().cpu().tolist())
    return out


def score_onnx(session, tokenizer, texts: list[str], max_len: int, chunk: int = 256) -> list[float]:
    import numpy as np

    out: list[float] = []
    for i in range(0, len(texts), chunk):
        batch = texts[i : i + chunk]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="np",
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        arr = session.run(None, feeds)[0]
        out.extend(float(x) for x in np.array(arr).reshape(-1).tolist())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/rerank_v2/holdout.jsonl")
    p.add_argument("--ckpt", default="artifacts/rerank/modernbert70m_ce")
    p.add_argument("--onnx-dir", default="artifacts/rerank/modernbert70m_ce/onnx")
    p.add_argument("--n-groups", type=int, default=200)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--fp32-only", action="store_true", help="skip int8 (contextual ship)")
    p.add_argument(
        "--pt-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch parity device; auto uses CUDA/ROCm when available",
    )
    p.add_argument(
        "--out",
        default="artifacts/rerank_ctx/trackB_v2_continue/onnx/parity_scores.json",
    )
    args = p.parse_args(argv)

    import onnxruntime as ort

    rows = list(read_jsonl(Path(args.data)))
    groups = prepare_groups(rows)[: max(1, args.n_groups)]
    print(f"groups={len(groups)}", flush=True)

    import torch

    pt_device = (
        "cuda" if args.pt_device == "auto" and torch.cuda.is_available() else args.pt_device
    )
    if pt_device == "auto":
        pt_device = "cpu"
    tokenizer, model, base, max_len, torch, pt_device = load_pt(
        Path(args.ckpt), args.max_len, pt_device
    )
    onnx_dir = Path(args.onnx_dir)
    fp32_path = onnx_dir / "cross_encoder_fp32.onnx"
    int8_path = onnx_dir / "cross_encoder_int8.onnx"
    if not fp32_path.exists():
        raise SystemExit(f"missing onnx fp32 under {onnx_dir}")
    skip_int8 = bool(args.fp32_only) or not int8_path.exists()

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = None
    if not skip_int8:
        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    tok_onnx = onnx_dir / "tokenizer"
    if tok_onnx.exists():
        from transformers import AutoTokenizer

        # Prefer ckpt tokenizer for parity with training; warn if onnx tok differs.
        pass

    flat_texts: list[str] = []
    flat_meta: list[tuple[dict[str, Any], str]] = []
    for g in groups:
        for cand in g["candidates"]:
            flat_texts.append(build_pair_text(g["reading"], g["context_prev"], cand))
            flat_meta.append((g, cand))
    print(f"scores={len(flat_texts)} pt_device={pt_device} batched=true", flush=True)
    xs_pt = score_pt(
        tokenizer, model, torch, flat_texts, max_len, device=pt_device
    )
    xs_of = score_onnx(sess_fp32, tokenizer, flat_texts, max_len)
    xs_oi = (
        score_onnx(sess_int8, tokenizer, flat_texts, max_len)
        if sess_int8 is not None
        else []
    )
    records: list[dict[str, Any]] = []
    for i, ((g, cand), a, b) in enumerate(zip(flat_meta, xs_pt, xs_of)):
        rec = {
            "reading": g["reading"],
            "candidate": cand,
            "gold": g["gold"],
            "mozc_top1": g["mozc_top1"],
            "score_pt_fp32": a,
            "score_onnx_fp32": b,
        }
        if sess_int8 is not None:
            rec["score_onnx_int8"] = xs_oi[i]
        records.append(rec)

    comparisons = [_stats(xs_pt, xs_of, "pt_fp32_vs_onnx_fp32")]
    if sess_int8 is not None:
        comparisons.extend(
            [
                _stats(xs_pt, xs_oi, "pt_fp32_vs_onnx_int8"),
                _stats(xs_of, xs_oi, "onnx_fp32_vs_onnx_int8"),
            ]
        )
    summary = {
        "base_model": base,
        "ckpt": str(args.ckpt),
        "onnx_dir": str(args.onnx_dir),
        "data": str(args.data),
        "n_groups": len(groups),
        "n_scores": len(records),
        "comparisons": comparisons,
        # Margin-relevant: score(rerank_best)-score(mozc) distribution proxies via std
        "diagnosis_hint": None,
        "samples": records[:50],
        "all_scores": records,
    }

    c0 = summary["comparisons"][0]
    if c0["mae"] > 0.05 or (c0["spearman"] == c0["spearman"] and c0["spearman"] < 0.99):
        cause = "export"
        hint = "ONNX fp32 diverges from PyTorch fp32 → export / pooling / tokenizer bug"
    elif sess_int8 is None:
        cause = "fp32_ok"
        hint = "ONNX fp32 matches PT (int8 skipped; ship path)"
    else:
        c1 = summary["comparisons"][1]
        if c1["mae"] > 0.2 or (c1["spearman"] == c1["spearman"] and c1["spearman"] < 0.95):
            cause = "int8_quantization"
            hint = "ONNX fp32 matches PT but int8 diverges → quantization issue"
        elif abs(c1["y_std"] / max(1e-8, c1["x_std"]) - 1.0) > 0.25:
            cause = "tau_scale"
            hint = "scores correlated but scale/std shifted → retune tau per backend"
        else:
            cause = "tau_scale_or_minor_quant"
            hint = "mostly aligned; check tau / small quant drift"
    summary["root_cause_candidate"] = cause
    summary["diagnosis_hint"] = hint

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n_groups", "n_scores", "comparisons", "root_cause_candidate", "diagnosis_hint"
    )}, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
