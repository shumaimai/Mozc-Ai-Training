#!/usr/bin/env python3
"""Compare previous vs quiet-rerun latency numbers."""
from __future__ import annotations

import json
from pathlib import Path


def _ship_rec(path: Path) -> dict:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rec = blob.get("recommended") or {}
    lat = rec.get("latency") or {}
    return {"p50": lat.get("p50"), "p95": lat.get("p95"), "mean": lat.get("mean")}


def _cap_map(path: Path) -> dict[tuple[str, int], dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in blob.get("rows") or []:
        out[(str(row["tag"]), int(row["cap"]))] = row
    return out


def main() -> int:
    ev = Path("artifacts/rerank_ctx/eval")
    old70 = _ship_rec(ev / "latency_ship_profile.json")
    old30 = _ship_rec(ev / "latency_ship_profile_30m.json")
    new70 = _ship_rec(ev / "latency_ship_profile_70m_quiet.json")
    new30 = _ship_rec(ev / "latency_ship_profile_30m_quiet.json")
    pareto = json.loads((ev / "ctx_cap_pareto.json").read_text(encoding="utf-8"))
    old_pts = {(p["tag"], p["cap"]): p for p in pareto["points"]}
    new_cap = _cap_map(ev / "ctx_cap_latency_quiet.json")

    ship_rows = []
    for tag, old, new in (("70m", old70, new70), ("30m", old30, new30)):
        ship_rows.append(
            {
                "tag": tag,
                "cap": 50,
                "kind": "ship_longest_intra12",
                "old_p50": old.get("p50"),
                "old_p95": old.get("p95"),
                "new_p50": new.get("p50"),
                "new_p95": new.get("p95"),
                "p50_ratio": round(float(new["p50"]) / float(old["p50"]), 3)
                if old.get("p50")
                else None,
                "p95_ratio": round(float(new["p95"]) / float(old["p95"]), 3)
                if old.get("p95")
                else None,
            }
        )

    cap_rows = []
    for tag in ("70m", "30m"):
        for cap in (50, 30, 20):
            o = old_pts.get((tag, cap), {})
            n = new_cap.get((tag, cap), {})
            cap_rows.append(
                {
                    "tag": tag,
                    "cap": cap,
                    "old_p50_run1": o.get("p50_run1"),
                    "old_p95_run1": o.get("p95_run1"),
                    "old_p50_run2": o.get("p50_run2"),
                    "old_p95_run2": o.get("p95_run2"),
                    "new_p50": n.get("p50"),
                    "new_p95": n.get("p95"),
                    "new_p95_lt_200": n.get("p95_lt_200"),
                    "seq_mean": (n.get("seq") or {}).get("mean") or o.get("seq_mean"),
                }
            )

    # Pareto on quiet cap sweep, CS Δ from quality (unchanged)
    cs = {(p["tag"], p["cap"]): p["min_cs_delta_pt"] for p in pareto["points"]}
    eligible = [r for r in cap_rows if r.get("new_p95_lt_200")]
    if eligible:
        pick = max(eligible, key=lambda r: (cs.get((r["tag"], r["cap"]), 0), -float(r["new_p95"] or 0)))
        pareto_choice = f"{pick['tag']}@{pick['cap']}"
    else:
        pick = min(cap_rows, key=lambda r: float(r.get("new_p95") or 1e9))
        pareto_choice = f"{pick['tag']}@{pick['cap']} (no p95<200; lowest p95)"

    report = {
        "note": "Quiet sequential rerun. Old = previous busy-machine numbers. New = this run.",
        "ship_profile_cap50": ship_rows,
        "ctx_cap": cap_rows,
        "pareto_quiet": {
            "choice": pareto_choice,
            "rule": "among new p95<200, maximize min CS Δ (quality sweep, unchanged)",
        },
    }
    out = ev / "latency_quiet_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
