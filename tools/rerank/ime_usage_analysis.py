#!/usr/bin/env python3
"""Build IME usage analysis: intended (committed) vs shown conversion."""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path.home() / "AppData/LocalLow/Mozc/rerank_smoke.jsonl"
DAEMON = ROOT / "artifacts/rerank_ctx/eval/daemon_requests.jsonl"
OUT_DIR = ROOT / "artifacts/rerank_ctx/eval"
USER_PAIRS = ROOT / "artifacts/rerank_ctx/eval/ime_usage_pairs.jsonl"
NOISE_READING = re.compile(r"^[\s\-~\\[\](){}.,0-9a-zA-Z]+$")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_noise(row: dict) -> bool:
    reading = str(row.get("reading") or "")
    chosen = str(row.get("chosen") or "")
    if not reading or not chosen:
        return True
    if len(reading) <= 1 and not ("\u3040" <= reading <= "\u30ff"):
        return True
    if NOISE_READING.match(reading) and len(reading) < 4:
        return True
    if reading in {"]", "~", " ", "　"}:
        return True
    if sum(ch.isdigit() or ("０" <= ch <= "９") for ch in reading) >= 3:
        return True
    return False


def is_log_artifact(row: dict) -> bool:
    """Finish() logs conversion_segment(0), not the last reranked segment."""
    chosen = str(row.get("chosen") or "")
    ctx = str(row.get("context_prev") or "")
    shown = str(row.get("final_top1") or "")
    if ctx and chosen == ctx:
        return True
    if ctx and chosen.startswith(ctx) and shown and chosen != shown:
        return True
    return False


def daemon_index(path: Path) -> dict[tuple[str, str, str], dict]:
    """Last daemon row per (reading, context, final_top1)."""
    idx: dict[tuple[str, str, str], dict] = {}
    if not path.is_file():
        return idx
    for r in load_jsonl(path):
        key = (
            str(r.get("reading") or ""),
            str(r.get("context_prev") or ""),
            str(r.get("final_top1") or ""),
        )
        idx[key] = r
    return idx


def classify(row: dict) -> str:
    nbest = row.get("nbest") or []
    mozc = nbest[0] if nbest else ""
    shown = str(row.get("final_top1") or mozc)
    chosen = str(row.get("chosen") or "")
    overwritten = bool(row.get("overwritten"))
    if chosen == shown:
        if overwritten and shown != mozc:
            return "rerank_helped"
        return "accepted_shown"
    if overwritten and chosen == mozc and shown != mozc:
        return "rerank_rejected"
    return "user_picked_other"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SMOKE.is_file():
        shutil.copy2(SMOKE, OUT_DIR / "rerank_smoke.jsonl")
    rows = [r for r in load_jsonl(SMOKE) if r.get("source") == "ime_online"]
    commits = [r for r in rows if not is_noise(r)]
    artifacts = [r for r in commits if is_log_artifact(r)]
    usable = [r for r in commits if not is_log_artifact(r)]
    didx = daemon_index(DAEMON)
    events = []
    for r in usable:
        nbest = r.get("nbest") or []
        mozc = nbest[0] if nbest else ""
        shown = str(r.get("final_top1") or mozc)
        chosen = str(r.get("chosen") or "")
        kind = classify(r)
        tau = float(r.get("tau") if r.get("tau") is not None else 2.5)
        drow = didx.get(
            (str(r.get("reading") or ""), str(r.get("context_prev") or ""), shown)
        ) or {}
        margin = drow.get("margin")
        reason = drow.get("reason")
        events.append(
            {
                "ts": r.get("ts"),
                "reading": r.get("reading"),
                "context": r.get("context_prev") or "",
                "wanted": chosen,
                "shown": shown,
                "mozc_top1": mozc,
                "rerank_top1": r.get("rerank_top1"),
                "overwritten": bool(r.get("overwritten")),
                "tau": tau,
                "margin": margin,
                "reason": reason,
                "margin_ge_tau": (
                    None if margin is None else bool(float(margin) >= tau)
                ),
                "kind": kind,
                "match_shown": chosen == shown,
                "match_mozc": chosen == mozc,
            }
        )

    counts = Counter(e["kind"] for e in events)
    n = len(events)
    helped = [e for e in events if e["kind"] == "rerank_helped"]
    rejected = [e for e in events if e["kind"] == "rerank_rejected"]
    picked = [e for e in events if e["kind"] == "user_picked_other"]
    overwrites = [e for e in events if e["overwritten"]]

    def compact(e: dict) -> dict:
        return {
            "reading": e["reading"],
            "context": e["context"],
            "wanted": e["wanted"],
            "shown": e["shown"],
            "mozc": e["mozc_top1"],
            "rerank": e["rerank_top1"],
        }

    report = {
        "source": str(SMOKE),
        "n_raw": len(rows),
        "n_commits": n,
        "n_log_artifact": len(artifacts),
        "shown_hit": round(100.0 * sum(1 for e in events if e["match_shown"]) / n, 1) if n else 0,
        "mozc_hit": round(100.0 * sum(1 for e in events if e["match_mozc"]) / n, 1) if n else 0,
        "counts": dict(counts),
        "n_overwrite": len(overwrites),
        "helped": [compact(e) for e in helped],
        "rejected": [compact(e) for e in rejected],
        "user_picked_other": [compact(e) for e in picked],
        "all": events,
    }
    out = OUT_DIR / "ime_usage_analysis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    slim = OUT_DIR / "ime_usage_pairs.jsonl"
    USER_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
    slim.write_text(text, encoding="utf-8")
    USER_PAIRS.write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_raw", "n_commits", "n_log_artifact", "shown_hit", "mozc_hit", "counts", "n_overwrite")}, ensure_ascii=False, indent=2))
    print("helped", len(helped), "rejected", len(rejected), "picked_other", len(picked))
    print("WROTE", out)
    print("WROTE", slim)
    print("WROTE", USER_PAIRS)
    n_margin = sum(1 for e in events if e.get("margin") is not None)
    print("margin_joined", n_margin, "/", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
