"""Evaluate trained cross-encoder reranker on Mozc N-best groups.

Margin gate (default policy):
  final = rerank_top1  if score(rerank)-score(mozc_top1) >= tau
  else   mozc_top1

GPU: full quality metrics + tau sweep on holdout.
CPU: one latency pass (response speed).

Example:
  python -m tools.rerank.eval_cross_encoder \\
    --data data/rerank_v2/holdout.jsonl \\
    --ckpt artifacts/rerank/modernbert70m_ce \\
    --device cuda --batch-size 1024 --fp16 \\
    --tau 0 --tau-sweep 0,0.5,1,1.5,2,3 \\
    --out artifacts/rerank/eval_holdout_margin.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.margin import metrics_at_tau
from tools.rerank.train_cross_encoder import build_pair_text


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _vram_mb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def load_model(ckpt_dir: Path, device: str, fp16: bool):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    meta_path = ckpt_dir / "train_meta.json"
    pt_path = ckpt_dir / "cross_encoder.pt"
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    base = blob.get("base_model")
    if not base and meta_path.exists():
        base = json.loads(meta_path.read_text(encoding="utf-8")).get("base_model")
    if not base:
        raise SystemExit(f"base_model missing in {pt_path}")

    tok_dir = ckpt_dir / "tokenizer"
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
    use_amp = bool(fp16 and device == "cuda")
    return tokenizer, model, base, use_amp


def score_texts(
    texts: list[str],
    *,
    tokenizer,
    model,
    device: str,
    max_len: int,
    batch_size: int,
    use_amp: bool,
) -> list[float]:
    import torch

    scores: list[float] = []
    if not texts:
        return scores
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
        scores.extend(float(x) for x in logits.detach().float().cpu().tolist())
    return scores


def prepare_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        reading = row.get("reading") or ""
        gold = row.get("gold") or ""
        nbest = [c for c in (row.get("mozc_nbest") or []) if c]
        if not reading or not nbest:
            continue
        seen: set[str] = set()
        cands: list[str] = []
        for c in nbest:
            if c in seen:
                continue
            seen.add(c)
            cands.append(c)
        mozc_top1 = row.get("mozc_top1") or cands[0]
        # Ensure Mozc top-1 is scoreable even if extractor omitted it.
        if mozc_top1 not in seen:
            cands = [mozc_top1, *cands]
            seen.add(mozc_top1)
        groups.append(
            {
                "idx": i,
                "reading": reading,
                "gold": gold,
                "context_prev": row.get("context_prev") or "",
                "candidates": cands,
                "mozc_top1": mozc_top1,
                "mozc_hit1": bool(mozc_top1 == gold),
                "gold_in_nbest": bool(gold in seen) if gold else False,
                "source": row.get("source") or "",
                "category": row.get("category") or "",
            }
        )
    return groups


def cap_groups(groups: list[dict[str, Any]], cand_cap: int) -> list[dict[str, Any]]:
    """Apply the serving candidate cap while always retaining Mozc top-1."""
    if cand_cap <= 0:
        return groups
    out: list[dict[str, Any]] = []
    for group in groups:
        candidates = list(group["candidates"][:cand_cap])
        mozc = group["mozc_top1"]
        if mozc not in candidates:
            candidates = [mozc, *[c for c in candidates if c != mozc]][:cand_cap]
        out.append({**group, "candidates": candidates})
    return out


def score_groups(
    groups: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    device: str,
    max_len: int,
    batch_size: int,
    use_amp: bool,
    measure_latency: bool,
) -> tuple[list[list[float]], dict[str, Any]]:
    import torch

    n = len(groups)
    per_group_ms: list[float] = []
    group_scores: list[list[float]] = []
    total_cands = sum(len(g["candidates"]) for g in groups)

    if measure_latency and device == "cpu":
        t0 = time.perf_counter()
        for g in groups:
            g_texts = [
                build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]
            ]
            t1 = time.perf_counter()
            scores = score_texts(
                g_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_len=max_len,
                batch_size=batch_size,
                use_amp=False,
            )
            per_group_ms.append((time.perf_counter() - t1) * 1000.0)
            group_scores.append(scores)
        elapsed = time.perf_counter() - t0
    else:
        offsets: list[tuple[int, int]] = []
        texts: list[str] = []
        for g in groups:
            start = len(texts)
            for cand in g["candidates"]:
                texts.append(build_pair_text(g["reading"], g["context_prev"], cand))
            offsets.append((start, len(texts)))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_scores = score_texts(
            texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=max_len,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        for a, b in offsets:
            group_scores.append(all_scores[a:b])

    latency: dict[str, Any] = {
        "wall_s": round(elapsed, 3),
        "ms_per_group": round((elapsed / n) * 1000, 3) if n else 0.0,
        "ms_per_candidate": round((elapsed / total_cands) * 1000, 3) if total_cands else 0.0,
        "groups_per_s": round(n / elapsed, 3) if elapsed > 0 else 0.0,
        "measured": "per_group_sequential" if per_group_ms else "batched_wall",
    }
    if per_group_ms:
        latency.update(
            {
                "p50_ms_per_group": round(_percentile(per_group_ms, 50), 3),
                "p95_ms_per_group": round(_percentile(per_group_ms, 95), 3),
                "mean_ms_per_group": round(statistics.mean(per_group_ms), 3),
                "max_ms_per_group": round(max(per_group_ms), 3),
            }
        )
    stats: dict[str, Any] = {
        "n_groups": n,
        "n_candidates": total_cands,
        "latency": latency,
    }
    if device == "cuda":
        stats["vram"] = _vram_mb()
    return group_scores, stats


def _recommend_tau(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for m in sweep:
        if m["regression_rate_on_mozc_hit"] <= 0.02:
            if best is None or m["final_hit1"] > best["final_hit1"]:
                best = m
    rule = "max final_hit1 among tau with regression<=2%"
    if best is None:
        rule = "max final_hit1 among tau with regression<=5%"
        for m in sweep:
            if m["regression_rate_on_mozc_hit"] <= 0.05:
                if best is None or m["final_hit1"] > best["final_hit1"]:
                    best = m
    if best is None:
        rule = "max final_hit1 (no regression constraint met)"
        best = max(sweep, key=lambda m: (m["final_hit1"], -m["regression_rate_on_mozc_hit"]))
    return {
        "tau": best["tau"],
        "final_hit1": best["final_hit1"],
        "delta_vs_mozc_pt": best["delta_vs_mozc_pt"],
        "regression_rate_on_mozc_hit": best["regression_rate_on_mozc_hit"],
        "recovery_rate_on_mozc_miss": best["recovery_rate_on_mozc_miss"],
        "overwrite_rate": best["overwrite_rate"],
        "rule": rule,
    }


def evaluate_groups(
    groups: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    device: str,
    max_len: int,
    batch_size: int,
    use_amp: bool,
    measure_latency: bool,
    tau: float = 0.0,
    tau_sweep: list[float] | None = None,
) -> dict[str, Any]:
    group_scores, stats = score_groups(
        groups,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        use_amp=use_amp,
        measure_latency=measure_latency,
    )

    primary = metrics_at_tau(groups, group_scores, tau)
    report: dict[str, Any] = {
        "device": device,
        "policy": "margin_gate",
        "tau": float(tau),
        "n_groups": stats["n_groups"],
        "n_candidates": stats["n_candidates"],
        "mozc_hit1": primary["mozc_hit1"],
        "rerank_raw_hit1": primary["rerank_raw_hit1"],
        "rerank_hit1": primary["final_hit1"],
        "final_hit1": primary["final_hit1"],
        "delta_hit1_pt": primary["delta_vs_mozc_pt"],
        "recovery_rate_on_mozc_miss": primary["recovery_rate_on_mozc_miss"],
        "regression_rate_on_mozc_hit": primary["regression_rate_on_mozc_hit"],
        "overwrite_rate": primary["overwrite_rate"],
        "counts": primary["counts"],
        "latency": stats["latency"],
        "flip_samples": primary["flip_samples"],
        "primary": primary,
    }
    if "vram" in stats:
        report["vram"] = stats["vram"]

    sweep_vals = list(tau_sweep or [])
    if float(tau) not in {float(x) for x in sweep_vals}:
        sweep_vals = [float(tau), *sweep_vals]
    seen: set[float] = set()
    uniq: list[float] = []
    for t in sweep_vals:
        ft = float(t)
        if ft in seen:
            continue
        seen.add(ft)
        uniq.append(ft)
    report["tau_sweep"] = [metrics_at_tau(groups, group_scores, t) for t in uniq]
    report["recommended_tau"] = _recommend_tau(report["tau_sweep"])
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval IME cross-encoder reranker")
    p.add_argument("--data", default="data/rerank_v2/holdout.jsonl")
    p.add_argument("--ckpt", default="artifacts/rerank/modernbert70m_ce")
    p.add_argument("--out", default="artifacts/rerank/eval_holdout.json")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument(
        "--cand-cap",
        type=int,
        default=0,
        help="serving candidate cap; 0 keeps the full evaluation N-best",
    )
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--latency-only", action="store_true")
    p.add_argument("--latency-groups", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="optional group cap")
    p.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="overwrite Mozc top-1 only if score(rerank)-score(mozc)>=tau",
    )
    p.add_argument(
        "--tau-sweep",
        default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5",
        help="comma-separated tau values evaluated after one scoring pass",
    )
    args = p.parse_args(argv)
    if args.no_fp16:
        args.fp16 = False
    tau_sweep = _parse_float_list(args.tau_sweep)

    import torch

    if args.device == "cuda":
        if args.require_cuda and not torch.cuda.is_available():
            raise SystemExit("CUDA required but unavailable")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
        _ = torch.zeros(8, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(
            f"device=cuda name={torch.cuda.get_device_name(0)} "
            f"vram_total_gb={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}",
            flush=True,
        )
    else:
        print("device=cpu (latency path)", flush=True)

    rows = list(read_jsonl(Path(args.data)))
    groups = prepare_groups(rows)
    groups = cap_groups(groups, int(args.cand_cap))
    if args.limit and args.limit > 0:
        groups = groups[: args.limit]
    if args.latency_only:
        groups = groups[: max(1, args.latency_groups)]

    print(f"groups={len(groups)} loading {args.ckpt}", flush=True)
    tokenizer, model, base, use_amp = load_model(
        Path(args.ckpt), args.device, args.fp16
    )
    print(
        f"base_model={base} amp={use_amp} batch_size={args.batch_size} tau={args.tau}",
        flush=True,
    )

    warm_n = min(8, len(groups[0]["candidates"]) if groups else 0)
    if warm_n:
        warm_texts = [
            build_pair_text(groups[0]["reading"], groups[0]["context_prev"], c)
            for c in groups[0]["candidates"][:warm_n]
        ]
        _ = score_texts(
            warm_texts,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            max_len=args.max_len,
            batch_size=args.batch_size,
            use_amp=use_amp,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()

    report = evaluate_groups(
        groups,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_amp=use_amp,
        measure_latency=bool(args.latency_only or args.device == "cpu"),
        tau=float(args.tau),
        tau_sweep=tau_sweep,
    )
    report.update(
        {
            "data": str(args.data),
            "ckpt": str(args.ckpt),
            "base_model": base,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "cand_cap": int(args.cand_cap),
            "fp16": use_amp,
            "latency_only": bool(args.latency_only),
            "policy_desc": "final=rerank iff score_r-score_m>=tau else mozc_top1",
        }
    )

    print("tau_sweep_table:", flush=True)
    for m in report.get("tau_sweep", []):
        print(
            f"  tau={m['tau']:>5} final={m['final_hit1']:.4f} "
            f"dMozc={m['delta_vs_mozc_pt']:+.2f}pt "
            f"rec={m['recovery_rate_on_mozc_miss']:.4f} "
            f"reg={m['regression_rate_on_mozc_hit']:.4f} "
            f"ow={m['overwrite_rate']:.4f}",
            flush=True,
        )
    print("recommended_tau:", report.get("recommended_tau"), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Avoid dumping huge JSON to stdout; summary already printed.
    summary = {
        k: report[k]
        for k in (
            "device",
            "tau",
            "n_groups",
            "mozc_hit1",
            "rerank_raw_hit1",
            "final_hit1",
            "delta_hit1_pt",
            "recovery_rate_on_mozc_miss",
            "regression_rate_on_mozc_hit",
            "overwrite_rate",
            "recommended_tau",
            "latency",
            "vram",
        )
        if k in report
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
