"""Offline B-inject trial (Phase 2 candidate sourcing).

Append reading->surface dict entries to Mozc N-best (list **tail** only),
score with existing CE ckpt, and apply dual margin:
  - Mozc-native cands: tau_native (e.g. 2.5)
  - injected cands:    tau_inject (stricter; sweep 3.0/3.5/4.0)

Reports:
  - % of B where gold becomes in list after inject (coverage lift)
  - hit@1 / regress on full holdout and hit-subset
  - recovery on A and on newly-covered B
  - ship/no-ship recommendation

Does NOT call Mozc. Dict sources (prefer production Mozc-diff bundle):
  - --dict-entries artifacts/dict/.../mozc_diff_entries.jsonl
  - --dict-map     artifacts/dict/.../reading_map.json
  - legacy: interim classify_in JSONL / TSV reading\\tsurface

Example (honest holdout-safe bundle + score):
  python -m tools.rerank.b_inject_eval \\
    --holdout data/rerank_v3/holdout.jsonl \\
    --dict-entries artifacts/dict/holdout_safe/mozc_diff_entries.jsonl \\
    --protocol source_holdout_mozc_diff \\
    --ckpt artifacts/rerank/modernbert70m_ce_v3 \\
    --device cuda --fp16 --require-cuda \\
    --tau-native 2.5 --tau-inject-sweep 2.5,3.0,3.5,4.0,4.5 \\
    --out artifacts/rerank/phase2_b_inject_honest_holdout_safe.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import re
import unicodedata

from tools.dataset.jsonl import read_jsonl
from tools.rerank.margin import apply_margin, metrics_at_tau, pick_rerank_top1, score_of
from tools.rerank.phase2_expand import analyze_ab, slice_tag


def normalize_surface(text: str) -> str:
    """NFKC + strip + collapse spaces/middle-dots for gold↔candidate compare."""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.strip()
    s = s.replace("\u3000", "").replace(" ", "")
    s = s.replace("・", "").replace("･", "")
    s = re.sub(r"\s+", "", s)
    return s


def _classify_rows_to_reading_gold(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Minimal classify_in -> (reading, gold) without importing prepare/mozc_batch."""
    out: list[tuple[str, str]] = []
    for row in rows:
        rec = row.get("record") or row
        reading = (rec.get("reading") or "").strip()
        gold = (rec.get("surface") or "").strip()
        if reading and gold:
            out.append((reading, gold))
    return out


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def load_reading_dict_from_classify(paths: list[Path]) -> dict[str, list[str]]:
    """reading -> unique surfaces (order preserved, Mozc-diff not applied yet)."""
    by_reading: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for p in paths:
        raw = list(read_jsonl(p))
        n_before = len(by_reading)
        for reading, gold in _classify_rows_to_reading_gold(raw):
            ng = normalize_surface(gold)
            if not reading or not ng:
                continue
            if ng in seen[reading]:
                continue
            seen[reading].add(ng)
            by_reading[reading].append(gold)
        print(
            f"dict loaded {p}: rows={len(raw)} readings_total={len(by_reading)} "
            f"(+{len(by_reading) - n_before})",
            flush=True,
        )
    return dict(by_reading)


def load_reading_dict_from_tsv(path: Path) -> dict[str, list[str]]:
    by_reading: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        reading, surface = parts[0].strip(), normalize_surface(parts[1])
        if not reading or not surface or surface in seen[reading]:
            continue
        seen[reading].add(surface)
        by_reading[reading].append(surface)
    return dict(by_reading)


def load_reading_dict_from_entries(path: Path) -> dict[str, list[str]]:
    """Load Mozc-diff bundle entries JSONL (tools.dict.bundle output)."""
    from tools.dict.bundle import load_reading_map_from_entries

    m = load_reading_map_from_entries(path)
    print(
        f"dict loaded entries {path}: readings={len(m)} "
        f"surfaces={sum(len(v) for v in m.values())}",
        flush=True,
    )
    return m


def load_reading_dict_from_map(path: Path) -> dict[str, list[str]]:
    from tools.dict.bundle import load_reading_map_json

    m = load_reading_map_json(path)
    print(
        f"dict loaded map {path}: readings={len(m)} "
        f"surfaces={sum(len(v) for v in m.values())}",
        flush=True,
    )
    return m


def inject_tail(
    nbest: list[str],
    extras: list[str],
    *,
    max_inject: int,
) -> tuple[list[str], list[str]]:
    """Append extras not already in nbest (normalized compare). Cap count.

    Returns (full_list, injected_only). Inject always at list tail.
    """
    have = {normalize_surface(c) for c in nbest}
    out = list(nbest)
    added: list[str] = []
    for s in extras:
        ns = normalize_surface(s)
        if not ns or ns in have:
            continue
        out.append(s)
        have.add(ns)
        added.append(s)
        if len(added) >= max_inject:
            break
    return out, added


def gold_in_list(gold: str, cands: list[str]) -> bool:
    g = normalize_surface(gold)
    if not g:
        return False
    return any(normalize_surface(c) == g for c in cands)


def apply_margin_dual(
    candidates: list[str],
    scores: list[float],
    mozc_top1: str,
    *,
    n_native: int,
    tau_native: float,
    tau_inject: float,
) -> dict[str, Any]:
    """Margin gate with stricter tau when argmax lands on an injected cand."""
    rerank_top1, best_i, score_r = pick_rerank_top1(candidates, scores)
    score_m = score_of(candidates, scores, mozc_top1)
    is_inject_pick = best_i >= int(n_native)
    tau = float(tau_inject if is_inject_pick else tau_native)
    if score_m is None:
        return {
            "final_top1": mozc_top1,
            "rerank_top1": rerank_top1,
            "mozc_top1": mozc_top1,
            "score_rerank": score_r,
            "score_mozc": None,
            "margin": None,
            "tau": tau,
            "tau_native": float(tau_native),
            "tau_inject": float(tau_inject),
            "pick_is_inject": is_inject_pick,
            "overwritten": False,
            "reason": "mozc_top1_not_in_candidates",
            "rerank_index": best_i,
        }
    margin = score_r - score_m
    overwrite = (rerank_top1 != mozc_top1) and (margin >= tau)
    return {
        "final_top1": rerank_top1 if overwrite else mozc_top1,
        "rerank_top1": rerank_top1,
        "mozc_top1": mozc_top1,
        "score_rerank": score_r,
        "score_mozc": score_m,
        "margin": margin,
        "tau": tau,
        "tau_native": float(tau_native),
        "tau_inject": float(tau_inject),
        "pick_is_inject": is_inject_pick,
        "overwritten": overwrite,
        "reason": (
            "margin_ok"
            if overwrite
            else ("same_top1" if rerank_top1 == mozc_top1 else "margin_below_tau")
        ),
        "rerank_index": best_i,
    }


def metrics_at_dual_tau(
    groups: list[dict[str, Any]],
    group_scores: list[list[float]],
    *,
    tau_native: float,
    tau_inject: float,
) -> dict[str, Any]:
    n = len(groups)
    mozc_hit = rerank_raw_hit = final_hit = 0
    recover = regress = overwrite_n = overwrite_inject = 0
    miss_n = hit_n = 0
    a_n = a_recover = 0
    b_n = b_covered = b_recover = 0
    hit_subset_final = 0

    for g, scores in zip(groups, group_scores):
        cands = g["candidates"]
        gold = g["gold"]
        mozc_top1 = g["mozc_top1"]
        n_native = int(g["n_native"])
        mh = bool(mozc_top1 == gold)
        decision = apply_margin_dual(
            cands,
            scores,
            mozc_top1,
            n_native=n_native,
            tau_native=tau_native,
            tau_inject=tau_inject,
        )
        final = decision["final_top1"]
        rh_raw = decision["rerank_top1"] == gold
        fh = final == gold
        tag = g.get("slice") or ""

        if mh:
            mozc_hit += 1
            hit_n += 1
            if fh:
                hit_subset_final += 1
            else:
                regress += 1
        else:
            miss_n += 1
            if fh:
                recover += 1
        if rh_raw:
            rerank_raw_hit += 1
        if fh:
            final_hit += 1
        if decision["overwritten"]:
            overwrite_n += 1
            if decision["pick_is_inject"]:
                overwrite_inject += 1

        if tag == "A":
            a_n += 1
            if fh:
                a_recover += 1
        elif tag == "B":
            b_n += 1
            if g.get("gold_in_after"):
                b_covered += 1
                if fh:
                    b_recover += 1

    def pct(a: int, b: int) -> float:
        return round(a / b, 6) if b else 0.0

    return {
        "tau_native": float(tau_native),
        "tau_inject": float(tau_inject),
        "n_groups": n,
        "mozc_hit1": pct(mozc_hit, n),
        "rerank_raw_hit1": pct(rerank_raw_hit, n),
        "final_hit1": pct(final_hit, n),
        "delta_vs_mozc_pt": round((pct(final_hit, n) - pct(mozc_hit, n)) * 100, 3),
        "recovery_rate_on_mozc_miss": pct(recover, miss_n),
        "regression_rate_on_mozc_hit": pct(regress, hit_n),
        "hit_subset_final_hit1": pct(hit_subset_final, hit_n),
        "overwrite_rate": pct(overwrite_n, n),
        "overwrite_inject_rate": pct(overwrite_inject, n),
        "A": {
            "n": a_n,
            "recover": a_recover,
            "recovery_rate": pct(a_recover, a_n),
        },
        "B_covered": {
            "n": b_covered,
            "recover": b_recover,
            "recovery_rate": pct(b_recover, b_covered),
            "note": "newly-covered B = gold not in native N-best but gold in list after inject",
        },
        "counts": {
            "mozc_hit": mozc_hit,
            "rerank_raw_hit": rerank_raw_hit,
            "final_hit": final_hit,
            "recover": recover,
            "regress": regress,
            "overwrite": overwrite_n,
            "overwrite_inject": overwrite_inject,
            "mozc_miss": miss_n,
            "mozc_hit_subset": hit_n,
            "A": a_n,
            "B": b_n,
            "B_covered": b_covered,
        },
    }


def coverage_eval(
    rows: list[dict[str, Any]],
    reading_dict: dict[str, list[str]],
    *,
    max_inject: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build injected groups + coverage stats (no model)."""
    ab = analyze_ab(rows)
    before_in = after_in = 0
    b_n = b_rescued = 0
    inject_events = 0
    inject_added_sum = 0
    hit_n = hit_still_in = 0
    groups: list[dict[str, Any]] = []

    for i, r in enumerate(rows):
        reading = (r.get("reading") or "").strip()
        gold = r.get("gold") or ""
        nbest_raw = [c for c in (r.get("mozc_nbest") or []) if c]
        if not reading or not nbest_raw:
            continue
        seen: set[str] = set()
        nbest: list[str] = []
        for c in nbest_raw:
            if c in seen:
                continue
            seen.add(c)
            nbest.append(c)
        mozc_top1 = r.get("mozc_top1") or nbest[0]
        if mozc_top1 not in seen:
            nbest = [mozc_top1, *nbest]
            seen.add(mozc_top1)

        extras = reading_dict.get(reading, [])
        injected, added = inject_tail(nbest, extras, max_inject=max_inject)
        if added:
            inject_events += 1
            inject_added_sum += len(added)

        bi = gold_in_list(gold, nbest)
        ai = gold_in_list(gold, injected)
        if bi:
            before_in += 1
        if ai:
            after_in += 1

        tag = slice_tag(r)
        if tag == "B":
            b_n += 1
            if (not bi) and ai:
                b_rescued += 1
        if tag == "hit":
            hit_n += 1
            if ai:
                hit_still_in += 1

        groups.append(
            {
                "idx": i,
                "reading": reading,
                "gold": gold,
                "context_prev": r.get("context_prev") or "",
                "candidates": injected,
                "n_native": len(nbest),
                "injected": added,
                "mozc_top1": mozc_top1,
                "mozc_hit1": bool(mozc_top1 == gold),
                "gold_in_before": bi,
                "gold_in_after": ai,
                "slice": tag,
                "source": r.get("source") or "",
                "category": r.get("category") or "",
            }
        )

    n = len(groups)
    cov = {
        "n": n,
        "ab_holdout": ab,
        "max_inject": max_inject,
        "dict_readings": len(reading_dict),
        "dict_surfaces": sum(len(v) for v in reading_dict.values()),
        "inject_events": inject_events,
        "inject_event_rate": round(inject_events / n, 6) if n else 0.0,
        "inject_added_mean": round(inject_added_sum / inject_events, 3) if inject_events else 0.0,
        "gold_in_list_before": round(before_in / n, 6) if n else 0.0,
        "gold_in_list_after": round(after_in / n, 6) if n else 0.0,
        "gold_in_list_lift": round((after_in - before_in) / n, 6) if n else 0.0,
        "B": {
            "n": b_n,
            "rescued": b_rescued,
            "rescue_rate": round(b_rescued / b_n, 6) if b_n else 0.0,
        },
        "hit": {
            "n": hit_n,
            "gold_still_in_after": hit_still_in,
            "retention": round(hit_still_in / hit_n, 6) if hit_n else 0.0,
            "note": "append-only inject should keep retention=1.0",
        },
    }
    return cov, groups


def _native_only_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in groups:
        n_native = int(g["n_native"])
        cands = list(g["candidates"][:n_native])
        out.append(
            {
                **g,
                "candidates": cands,
                "n_native": len(cands),
                "injected": [],
                "gold_in_after": g.get("gold_in_before"),
            }
        )
    return out


def _recommend_inject(
    baseline: dict[str, Any],
    sweep: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Pick best inject-tau: max hit@1 with regress <= baseline+0.5pp and <=2.5%."""
    base_hit = float(baseline["final_hit1"])
    base_reg = float(baseline["regression_rate_on_mozc_hit"])
    rescue = float(coverage["B"]["rescue_rate"])
    candidates = []
    for m in sweep:
        reg = float(m["regression_rate_on_mozc_hit"])
        if reg <= max(0.025, base_reg + 0.005):
            candidates.append(m)
    rule = "max final_hit1 among inject-tau with regress<=max(2.5%, baseline+0.5pp)"
    if not candidates:
        rule = "max final_hit1 (no regress constraint met)"
        best = max(sweep, key=lambda m: (m["final_hit1"], -m["regression_rate_on_mozc_hit"]))
    else:
        best = max(candidates, key=lambda m: (m["final_hit1"], -m["regression_rate_on_mozc_hit"]))

    delta_hit = round((float(best["final_hit1"]) - base_hit) * 100, 3)
    delta_reg = round(
        (float(best["regression_rate_on_mozc_hit"]) - base_reg) * 100, 3
    )
    b_rec = float(best["B_covered"]["recovery_rate"])
    ship = (
        rescue >= 0.15
        and float(best["regression_rate_on_mozc_hit"]) <= 0.025
        and (delta_hit >= 0.0 or b_rec >= 0.01)
        and float(best["hit_subset_final_hit1"]) >= 0.97
    )
    return {
        "best_tau_inject": best["tau_inject"],
        "tau_native": best["tau_native"],
        "final_hit1": best["final_hit1"],
        "regression_rate_on_mozc_hit": best["regression_rate_on_mozc_hit"],
        "hit_subset_final_hit1": best["hit_subset_final_hit1"],
        "A_recovery": best["A"]["recovery_rate"],
        "B_covered_recovery": best["B_covered"]["recovery_rate"],
        "delta_hit1_pt_vs_no_inject": delta_hit,
        "delta_regress_pt_vs_no_inject": delta_reg,
        "B_rescue_rate": rescue,
        "recommendation": "SHIP_inject_offline_ok" if ship else "NO_SHIP_keep_no_inject",
        "rule": rule,
        "rationale": (
            f"B rescue={rescue:.1%}; best inject-tau={best['tau_inject']} "
            f"hit@1={best['final_hit1']:.2%} (d{delta_hit:+.2f}pt vs no-inject) "
            f"regress={best['regression_rate_on_mozc_hit']:.2%} (d{delta_reg:+.2f}pt); "
            f"B_covered recovery={b_rec:.2%}"
        ),
    }


def run_scoring(
    groups: list[dict[str, Any]],
    *,
    ckpt: Path,
    device: str,
    batch_size: int,
    max_len: int,
    fp16: bool,
    require_cuda: bool,
    tau_native: float,
    tau_inject_sweep: list[float],
    cand_cap: int,
) -> dict[str, Any]:
    import torch

    from tools.rerank.eval_cross_encoder import load_model, score_groups

    if device == "cuda":
        if require_cuda and not torch.cuda.is_available():
            raise SystemExit("CUDA required but unavailable")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
        _ = torch.zeros(8, device="cuda")
        torch.cuda.synchronize()
        print(
            f"device=cuda name={torch.cuda.get_device_name(0)}",
            flush=True,
        )
    else:
        print("device=cpu", flush=True)

    capped: list[dict[str, Any]] = []
    for g in groups:
        cands = list(g["candidates"])
        n_native = int(g["n_native"])
        if cand_cap > 0 and len(cands) > cand_cap:
            native = cands[:n_native]
            inj = cands[n_native:]
            keep_inj = max(0, cand_cap - len(native))
            if keep_inj < len(inj):
                inj = inj[:keep_inj]
            if len(native) + len(inj) > cand_cap:
                native = native[:cand_cap]
                inj = []
            cands = native + inj
            n_native = len(native)
        capped.append({**g, "candidates": cands, "n_native": n_native})

    tokenizer, model, base, use_amp = load_model(ckpt, device, fp16 and device == "cuda")
    print(f"loaded ckpt={ckpt} base={base} groups={len(capped)}", flush=True)

    inj_scores, inj_stats = score_groups(
        capped,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        use_amp=use_amp,
        measure_latency=False,
    )

    native_groups = _native_only_groups(capped)
    native_scores = [s[: g["n_native"]] for g, s in zip(capped, inj_scores)]

    baseline = metrics_at_tau(native_groups, native_scores, tau_native)
    a_n = a_rec = 0
    for g, scores in zip(native_groups, native_scores):
        if g.get("slice") != "A":
            continue
        a_n += 1
        d = apply_margin(g["candidates"], scores, g["mozc_top1"], tau_native)
        if d["final_top1"] == g["gold"]:
            a_rec += 1
    baseline_ext = {
        **{k: baseline[k] for k in baseline if k != "flip_samples"},
        "A": {
            "n": a_n,
            "recover": a_rec,
            "recovery_rate": round(a_rec / a_n, 6) if a_n else 0.0,
        },
        "B_covered": {
            "n": 0,
            "recover": 0,
            "recovery_rate": 0.0,
            "note": "no-inject baseline: B remains uncovered",
        },
        "hit_subset_final_hit1": round(
            (baseline["counts"]["mozc_hit"] - baseline["counts"]["regress"])
            / baseline["counts"]["mozc_hit_subset"],
            6,
        )
        if baseline["counts"]["mozc_hit_subset"]
        else 0.0,
        "mode": "no_inject",
    }

    sweep = [
        metrics_at_dual_tau(
            capped, inj_scores, tau_native=tau_native, tau_inject=t
        )
        for t in tau_inject_sweep
    ]
    return {
        "ckpt": str(ckpt),
        "base_model": base,
        "device": device,
        "max_len": max_len,
        "cand_cap": cand_cap,
        "batch_size": batch_size,
        "fp16": bool(fp16 and device == "cuda"),
        "tau_native": float(tau_native),
        "tau_inject_sweep": [float(t) for t in tau_inject_sweep],
        "score_stats_inject": inj_stats,
        "baseline_no_inject": baseline_ext,
        "inject_tau_sweep": sweep,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline B-inject coverage + dual-tau scoring")
    p.add_argument("--holdout", default="data/rerank_v3/holdout.jsonl")
    p.add_argument("--dict-classify", nargs="*", default=[])
    p.add_argument("--dict-tsv", default="")
    p.add_argument(
        "--dict-entries",
        default="",
        help="Mozc-diff bundle mozc_diff_entries.jsonl (preferred)",
    )
    p.add_argument(
        "--dict-map",
        default="",
        help="Mozc-diff bundle reading_map.json",
    )
    p.add_argument(
        "--protocol",
        default="",
        help="eval protocol label (e.g. leaky_same_family / source_holdout / cross_source)",
    )
    p.add_argument("--trial-name", default="phase2_b_inject_offline")
    p.add_argument("--dict-source-policy", default="")
    p.add_argument("--max-inject", type=int, default=5)
    p.add_argument("--b-sample", type=int, default=0, help="optional cap on B rows only (0=all)")
    p.add_argument("--out", default="artifacts/rerank/phase2_b_inject_trial.json")
    p.add_argument("--ckpt", default="")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--cand-cap", type=int, default=30, help="0=no cap; distribution default 30")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--tau-native", type=float, default=2.5)
    p.add_argument("--tau-inject-sweep", default="2.5,3.0,3.5,4.0,4.5")
    args = p.parse_args(argv)
    if args.no_fp16:
        args.fp16 = False

    reading_dict: dict[str, list[str]] = {}
    dict_paths: list[str] = []
    policy = args.dict_source_policy
    if args.dict_entries:
        reading_dict = load_reading_dict_from_entries(Path(args.dict_entries))
        dict_paths = [args.dict_entries]
        policy = policy or "mozc_diff_bundle_entries"
    elif args.dict_map:
        reading_dict = load_reading_dict_from_map(Path(args.dict_map))
        dict_paths = [args.dict_map]
        policy = policy or "mozc_diff_bundle_map"
    elif args.dict_tsv:
        reading_dict = load_reading_dict_from_tsv(Path(args.dict_tsv))
        dict_paths = [args.dict_tsv]
        policy = policy or "tsv"
    elif args.dict_classify:
        paths = [Path(x) for x in args.dict_classify]
        reading_dict = load_reading_dict_from_classify(paths)
        dict_paths = [str(x) for x in paths]
        policy = policy or "classify_in_legacy"
    else:
        defaults = [
            Path("data/interim/mozc_batch/wikidata/classify_in.jsonl"),
        ]
        present = [x for x in defaults if x.exists()]
        if not present:
            raise SystemExit(
                "No dict sources. Pass --dict-entries / --dict-map / "
                "--dict-classify / --dict-tsv"
            )
        reading_dict = load_reading_dict_from_classify(present)
        dict_paths = [str(x) for x in present]
        policy = policy or "one_source_prefer_wikidata_places"

    rows = list(read_jsonl(Path(args.holdout)))
    if args.b_sample and args.b_sample > 0:
        b_idx = [i for i, r in enumerate(rows) if slice_tag(r) == "B"]
        keep_b = set(b_idx[: args.b_sample])
        rows = [r for i, r in enumerate(rows) if slice_tag(r) != "B" or i in keep_b]
        print(f"b_sample={args.b_sample}: holdout rows now {len(rows)}", flush=True)

    coverage, groups = coverage_eval(rows, reading_dict, max_inject=args.max_inject)
    report: dict[str, Any] = {
        "trial": args.trial_name,
        "protocol": args.protocol or "unspecified",
        "holdout": args.holdout,
        "dict_paths": dict_paths,
        "dict_source_policy": policy,
        "constraints": {
            "inject_position": "tail",
            "tau_native": float(args.tau_native),
            "int8": False,
            "cand_cap": int(args.cand_cap),
            "max_len_score": int(args.max_len),
        },
        "coverage": coverage,
        "anti_leak_note": (
            "If protocol is leaky_same_family, B rescue/hit@1 are upper bounds "
            "(dict shares family with holdout golds). Prefer source_holdout / "
            "cross_source for honest numbers."
        ),
    }

    if args.ckpt:
        scored = run_scoring(
            groups,
            ckpt=Path(args.ckpt),
            device=args.device,
            batch_size=args.batch_size,
            max_len=args.max_len,
            fp16=args.fp16,
            require_cuda=args.require_cuda,
            tau_native=float(args.tau_native),
            tau_inject_sweep=_parse_float_list(args.tau_inject_sweep),
            cand_cap=int(args.cand_cap),
        )
        report["scoring"] = scored
        report["recommendation"] = _recommend_inject(
            scored["baseline_no_inject"],
            scored["inject_tau_sweep"],
            coverage,
        )
    else:
        report["scoring"] = None
        report["recommendation"] = {
            "recommendation": "COVERAGE_ONLY_run_with_ckpt_for_ship_gate",
            "B_rescue_rate": coverage["B"]["rescue_rate"],
            "rationale": "Coverage sketch only; pass --ckpt for dual-tau hit/regress.",
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
