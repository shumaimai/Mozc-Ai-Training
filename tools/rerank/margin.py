"""Margin-gated overwrite of Mozc top-1 by reranker scores.

Policy:
  final = rerank_top1  if  score(rerank_top1) - score(mozc_top1) >= tau
  else   mozc_top1

tau=0  → overwrite whenever reranker prefers another cand (argmax always >=)
tau→∞ → never overwrite (Mozc-only)
"""

from __future__ import annotations

from typing import Any


def pick_rerank_top1(candidates: list[str], scores: list[float]) -> tuple[str, int, float]:
    if not candidates:
        raise ValueError("empty candidates")
    if len(candidates) != len(scores):
        raise ValueError("candidates/scores length mismatch")
    best_i = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_i], best_i, float(scores[best_i])


def score_of(candidates: list[str], scores: list[float], text: str) -> float | None:
    try:
        return float(scores[candidates.index(text)])
    except ValueError:
        return None


def apply_margin(
    candidates: list[str],
    scores: list[float],
    mozc_top1: str,
    tau: float,
) -> dict[str, Any]:
    """Return final top-1 under margin gate.

    If mozc_top1 is missing from candidates, keep mozc_top1 (no overwrite).
    """
    rerank_top1, best_i, score_r = pick_rerank_top1(candidates, scores)
    score_m = score_of(candidates, scores, mozc_top1)
    if score_m is None:
        return {
            "final_top1": mozc_top1,
            "rerank_top1": rerank_top1,
            "mozc_top1": mozc_top1,
            "score_rerank": score_r,
            "score_mozc": None,
            "margin": None,
            "tau": float(tau),
            "overwritten": False,
            "reason": "mozc_top1_not_in_candidates",
        }
    margin = score_r - score_m
    overwrite = (rerank_top1 != mozc_top1) and (margin >= float(tau))
    return {
        "final_top1": rerank_top1 if overwrite else mozc_top1,
        "rerank_top1": rerank_top1,
        "mozc_top1": mozc_top1,
        "score_rerank": score_r,
        "score_mozc": score_m,
        "margin": margin,
        "tau": float(tau),
        "overwritten": overwrite,
        "reason": "margin_ok" if overwrite else ("same_top1" if rerank_top1 == mozc_top1 else "margin_below_tau"),
        "rerank_index": best_i,
    }


def metrics_at_tau(
    groups: list[dict[str, Any]],
    group_scores: list[list[float]],
    tau: float,
) -> dict[str, Any]:
    n = len(groups)
    mozc_hit = rerank_raw_hit = final_hit = 0
    recover = regress = overwrite_n = 0
    miss_n = hit_n = 0
    margins_overwrite: list[float] = []
    flip_samples: list[dict[str, Any]] = []

    for g, scores in zip(groups, group_scores):
        cands = g["candidates"]
        gold = g["gold"]
        mozc_top1 = g["mozc_top1"]
        mh = bool(mozc_top1 == gold)
        decision = apply_margin(cands, scores, mozc_top1, tau)
        final = decision["final_top1"]
        rh_raw = decision["rerank_top1"] == gold
        fh = final == gold

        if mh:
            mozc_hit += 1
            hit_n += 1
            if not fh:
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
            if decision["margin"] is not None:
                margins_overwrite.append(float(decision["margin"]))

        if len(flip_samples) < 15 and (fh != mh):
            flip_samples.append(
                {
                    "reading": g["reading"],
                    "gold": gold,
                    "mozc_top1": mozc_top1,
                    "rerank_top1": decision["rerank_top1"],
                    "final_top1": final,
                    "margin": decision["margin"],
                    "overwritten": decision["overwritten"],
                    "mozc_hit1": mh,
                    "final_hit1": fh,
                }
            )

    def pct(a: int, b: int) -> float:
        return round(a / b, 6) if b else 0.0

    return {
        "tau": float(tau),
        "n_groups": n,
        "mozc_hit1": pct(mozc_hit, n),
        "rerank_raw_hit1": pct(rerank_raw_hit, n),
        "final_hit1": pct(final_hit, n),
        "delta_vs_mozc_pt": round((pct(final_hit, n) - pct(mozc_hit, n)) * 100, 3),
        "delta_vs_raw_rerank_pt": round((pct(final_hit, n) - pct(rerank_raw_hit, n)) * 100, 3),
        "recovery_rate_on_mozc_miss": pct(recover, miss_n),
        "regression_rate_on_mozc_hit": pct(regress, hit_n),
        "overwrite_rate": pct(overwrite_n, n),
        "counts": {
            "mozc_hit": mozc_hit,
            "rerank_raw_hit": rerank_raw_hit,
            "final_hit": final_hit,
            "recover": recover,
            "regress": regress,
            "overwrite": overwrite_n,
            "mozc_miss": miss_n,
            "mozc_hit_subset": hit_n,
        },
        "overwrite_margin_p50": round(sorted(margins_overwrite)[len(margins_overwrite) // 2], 4)
        if margins_overwrite
        else None,
        "flip_samples": flip_samples,
    }
