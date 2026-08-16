"""Python vs C++ context_clip parity (NEXT_TASK_PHASE3_CTX §2)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.rerank.context_clip import clean_context, clip_context_prev, normalize_reading

CASES = [
    "",
    "こんにちは",
    "昨日は雨だった。今日は汽車が",
    "概説\n== 歴史 ==\n* 項目\n本文。続きは記者の",
    "foo [[編集]] bar",
    "see [edit] now",
    "リンクは[[東京|とうきょう]]です。次は",
    "注釈[1]と<ref>x</ref>残る。進行中",
    "　　全角　空白\tタブ\n改行",
    "A" * 80,
    "文。！？混在! yes? 進行",
]


def run_cli(cli: Path, op: str, text: str, extra: list[str] | None = None) -> str:
    cmd = [str(cli), "--op", op]
    if extra:
        cmd.extend(extra)
    p = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"cli rc={p.returncode} err={p.stderr.decode('utf-8', 'replace')}")
    return p.stdout.decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    args = ap.parse_args()
    cli = Path(args.cli)
    n_ok = 0
    n = 0
    for raw in CASES:
        n += 1
        py = clean_context(raw)
        cpp = run_cli(cli, "clean", raw)
        if py != cpp:
            print("FAIL clean", repr(raw[:80]))
            print("  py ", repr(py))
            print("  cpp", repr(cpp))
            continue
        n_ok += 1
        n += 1
        py_r = normalize_reading("キシャ１２Ａ")
        cpp_r = run_cli(cli, "reading", "キシャ１２Ａ")
        if py_r != cpp_r:
            print("FAIL reading", repr(py_r), repr(cpp_r))
        else:
            n_ok += 1
    # clip
    full = "前文。今日は汽車が駅に"
    start = len(full) - 2
    n += 1
    py = clip_context_prev(full, start)
    cpp = run_cli(cli, "clip", full, ["--start", str(start)])
    if py != cpp:
        print("FAIL clip", repr(py), repr(cpp))
    else:
        n_ok += 1
    print(f"parity {n_ok}/{n}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
