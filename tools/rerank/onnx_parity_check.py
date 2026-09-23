"""Verify PyTorch↔ONNX score parity and token-ID parity for the Phase 2 model.

Covers three tokenizer paths that must agree on the same input string:
  1. HF AutoTokenizer (what train/eval used, saved next to the checkpoint)
  2. SentencePiece direct + manual BOS/EOS (what the shipped runtime daemon does)
  3. Score-level: PyTorch fp32 vs ONNX Runtime fp32 on identical token IDs

The C++ IME engine does not tokenize; it ships raw strings to the loopback
daemon over TCP, so path 2 *is* the live runtime tokenization.  The C++
``hf_tokenizer.cc`` WordPiece loader belongs to the retired WordPiece track
and is deliberately not part of this parity set.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import cap_groups, prepare_groups
from tools.rerank.train_cross_encoder import build_pair_text, parse_eligibility_statuses


def sp_tokenize(sp_model_path: Path, text: str, max_len: int) -> list[int]:
    """Shipped daemon tokenization (Mozc-Ai/runtime/rerank_daemon.py)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    return [1] + sp.encode(text, out_type=int)[: max_len - 2] + [2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True, help="dir with cross_encoder.pt + tokenizer/")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--sp-model", required=True, help="tokenizer.model for daemon path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--eligibility-status", default="")
    parser.add_argument("--cand-cap", type=int, default=30)
    args = parser.parse_args()

    import numpy as np
    import torch
    import onnxruntime as ort
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    rows = list(read_jsonl(Path(args.data)))
    groups = cap_groups(
        prepare_groups(rows, eligibility_statuses=parse_eligibility_statuses(args.eligibility_status)),
        args.cand_cap,
    )[: args.groups]
    texts: list[str] = []
    for g in groups:
        for cand in g["candidates"]:
            texts.append(build_pair_text(g["reading"], g["context_prev"], cand))
    print(f"groups={len(groups)} texts={len(texts)}", flush=True)

    # --- Token parity: HF (train/eval) vs SentencePiece direct (runtime daemon)
    hf_tok = AutoTokenizer.from_pretrained(str(Path(args.ckpt) / "tokenizer"), trust_remote_code=True)
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(args.sp_model))
    token_mismatch = 0
    mismatch_non_truncated = 0
    over_truncation = 0
    max_len_seen = 0
    for text in texts:
        hf_ids = hf_tok(text, truncation=True, max_length=args.max_len, add_special_tokens=True)["input_ids"]
        sp_ids = [1] + sp.encode(text, out_type=int)[: args.max_len - 2] + [2]
        sp_len = len(sp.encode(text, out_type=int))
        max_len_seen = max(max_len_seen, len(sp_ids))
        truncated = sp_len > args.max_len - 2
        if truncated:
            over_truncation += 1
        if list(hf_ids) != sp_ids:
            token_mismatch += 1
            if not truncated:
                mismatch_non_truncated += 1
    token_report = {
        "texts": len(texts),
        "hf_vs_sp_exact_match": len(texts) - token_mismatch,
        "hf_vs_sp_mismatch": token_mismatch,
        "mismatch_excluding_truncated_texts": mismatch_non_truncated,
        "exact_match_rate": (len(texts) - token_mismatch) / len(texts),
        "texts_exceeding_sp_truncation_window": over_truncation,
        "max_sp_sequence_len": max_len_seen,
        "max_len": args.max_len,
    }
    print("TOKEN_PARITY", json.dumps(token_report), flush=True)

    # --- PyTorch fp32 reference (the training forward pass)
    blob = torch.load(Path(args.ckpt) / "cross_encoder.pt", map_location="cpu", weights_only=False)

    class CrossEncoder(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32)
            self.score = nn.Linear(int(self.encoder.config.hidden_size), 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.score(out.last_hidden_state[:, 0]).squeeze(-1)

    model = CrossEncoder(blob["base_model"])
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    torch_scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i : i + args.batch_size]
            enc = hf_tok(chunk, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            logits = model(enc["input_ids"], enc["attention_mask"])
            torch_scores.extend(float(x) for x in logits.float().cpu().tolist())

    # --- ORT fp32 on identical HF token IDs (isolates graph conversion error)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, (torch.get_num_threads() or 1))
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=["CPUExecutionProvider"])
    ort_scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i : i + args.batch_size]
            enc = hf_tok(chunk, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            result = session.run(
                None,
                {
                    "input_ids": enc["input_ids"].numpy().astype(np.int64),
                    "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
                },
            )[0]
            ort_scores.extend(float(x) for x in np.asarray(result).reshape(-1).tolist())

    diffs = [abs(a - b) for a, b in zip(torch_scores, ort_scores)]
    per_group = {"n": 0, "argmax_agree": 0, "tau25_final_agree": 0}
    offset = 0
    for g in groups:
        k = len(g["candidates"])
        t = torch_scores[offset : offset + k]
        o = ort_scores[offset : offset + k]
        offset += k
        per_group["n"] += 1
        if max(range(k), key=lambda j: t[j]) == max(range(k), key=lambda j: o[j]):
            per_group["argmax_agree"] += 1
        tau = 2.5
        mozc_top1 = g["mozc_top1"]

        def final(scores: list[float]) -> str:
            best = max(range(k), key=lambda j: scores[j])
            margin = scores[best] - scores[0]
            return g["candidates"][best] if (best != 0 and margin >= tau) else mozc_top1

        if final(t) == final(o):
            per_group["tau25_final_agree"] += 1
    score_report = {
        "n_texts": len(texts),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": statistics.fmean(diffs),
        "p99_abs_diff": sorted(diffs)[int(len(diffs) * 0.99) - 1],
        "groups": per_group["n"],
        "group_argmax_agree": per_group["argmax_agree"],
        "group_tau25_final_agree": per_group["tau25_final_agree"],
        "torch_version": torch.__version__,
    }
    print("SCORE_PARITY", json.dumps(score_report), flush=True)

    report = {"token_parity": token_report, "score_parity": score_report}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
