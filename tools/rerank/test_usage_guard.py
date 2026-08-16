"""Unit tests for usage_guard (NEXT_TASK_USAGE_GUARD)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.rerank.phase3_hook import rerank_one
from tools.rerank.usage_guard import (
    REASON_CONTEXT_EMPTY_OR_SYMBOL,
    REASON_JUNK_CANDIDATE,
    REASON_READING_NOT_ELIGIBLE,
    REASON_READING_TOO_SHORT,
    context_empty_or_symbol,
    is_junk_surface,
    skip_reason,
)

ROOT = Path(__file__).resolve().parents[2]
CLI_SRC = ROOT / "mozc_compat" / "rerank_guard.cc"
CLI_MAIN = ROOT / "mozc_compat" / "rerank_guard_cli.cc"


class BoomScorer:
    def score(self, texts: list[str]) -> list[float]:
        raise AssertionError(f"scorer should not run: {texts}")


class PreferSecondScorer:
    def score(self, texts: list[str]) -> list[float]:
        return [0.0] + [5.0] * (len(texts) - 1)


class UsageGuardTest(unittest.TestCase):
    def test_short_reading(self) -> None:
        self.assertEqual(skip_reason("い", "文化"), REASON_READING_TOO_SHORT)
        self.assertEqual(skip_reason("ねん", "5年"), REASON_READING_TOO_SHORT)
        self.assertEqual(skip_reason("がつ", "文化"), REASON_READING_TOO_SHORT)
        self.assertEqual(skip_reason("ぶん", "PR"), REASON_READING_TOO_SHORT)

    def test_empty_or_symbol_context(self) -> None:
        self.assertTrue(context_empty_or_symbol(""))
        self.assertTrue(context_empty_or_symbol("1"))
        self.assertTrue(context_empty_or_symbol("２"))
        self.assertTrue(context_empty_or_symbol("、"))
        self.assertTrue(context_empty_or_symbol("  "))
        self.assertFalse(context_empty_or_symbol("駅に"))
        self.assertFalse(context_empty_or_symbol("新聞の"))
        self.assertEqual(skip_reason("きしゃ", ""), REASON_CONTEXT_EMPTY_OR_SYMBOL)
        self.assertEqual(skip_reason("きしゃ", "1"), REASON_CONTEXT_EMPTY_OR_SYMBOL)

    def test_whitelist(self) -> None:
        self.assertEqual(
            skip_reason("いいんちょう", "文化", mode="strict"),
            REASON_READING_NOT_ELIGIBLE,
        )
        self.assertEqual(
            skip_reason("きょうかい", "全国商業高等学校", mode="strict"),
            REASON_READING_NOT_ELIGIBLE,
        )
        self.assertIsNone(skip_reason("きしゃ", "駅に", mode="strict"))
        self.assertIsNone(skip_reason("きょうぎ", "全国高等学校", mode="strict"))

    def test_safety_mode_relaxes_only_whitelist(self) -> None:
        self.assertIsNone(skip_reason("いいんちょう", "文化", mode="safety"))
        self.assertEqual(
            skip_reason("い", "文化", mode="safety"), REASON_READING_TOO_SHORT
        )
        self.assertEqual(
            skip_reason("いいんちょう", "1", mode="safety"),
            REASON_CONTEXT_EMPTY_OR_SYMBOL,
        )

    def test_junk_surface(self) -> None:
        self.assertTrue(is_junk_surface("ヨセン"))
        self.assertTrue(is_junk_surface("實際に"))
        self.assertTrue(is_junk_surface("讃仰"))
        self.assertFalse(is_junk_surface("予選"))
        self.assertFalse(is_junk_surface("競技"))
        self.assertFalse(is_junk_surface("汽車"))

    def test_rerank_one_skips_without_scoring(self) -> None:
        out = rerank_one(
            {"reading": "い", "context_prev": "3", "nbest": ["位", "李"]},
            BoomScorer(),
            tau=2.5,
            cand_cap=30,
        )
        self.assertFalse(out["overwritten"])
        self.assertEqual(out["final_top1"], "位")
        self.assertEqual(out["reason"], REASON_READING_TOO_SHORT)
        self.assertTrue(out["guard_skip"])

    def test_kishya_with_station_context_still_scores(self) -> None:
        out = rerank_one(
            {"reading": "きしゃ", "context_prev": "駅に", "nbest": ["記者", "汽車"]},
            PreferSecondScorer(),
            tau=2.5,
            cand_cap=30,
        )
        self.assertTrue(out["overwritten"])
        self.assertEqual(out["final_top1"], "汽車")
        self.assertFalse(out["guard_skip"])

    def test_junk_overwrite_reverts(self) -> None:
        out = rerank_one(
            {
                "reading": "きょうぎ",
                "context_prev": "全国大会",
                "nbest": ["競技", "ヨセン"],
            },
            PreferSecondScorer(),
            tau=0.1,
            cand_cap=30,
        )
        self.assertFalse(out["overwritten"])
        self.assertEqual(out["final_top1"], "競技")
        self.assertEqual(out["reason"], REASON_JUNK_CANDIDATE)

    def test_guard_env_off(self) -> None:
        old = os.environ.get("MOZC_RERANK_GUARD")
        os.environ["MOZC_RERANK_GUARD"] = "0"
        try:
            self.assertIsNone(skip_reason("い", "3"))
            out = rerank_one(
                {"reading": "い", "context_prev": "3", "nbest": ["位", "李"]},
                PreferSecondScorer(),
                tau=0.1,
                cand_cap=30,
            )
            self.assertTrue(out["overwritten"])
            self.assertEqual(out["final_top1"], "李")
        finally:
            if old is None:
                os.environ.pop("MOZC_RERANK_GUARD", None)
            else:
                os.environ["MOZC_RERANK_GUARD"] = old


class UsageGuardCppParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = None
        gxx = "g++"
        try:
            subprocess.run([gxx, "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return
        tmp = Path(tempfile.mkdtemp(prefix="rerank_guard_"))
        exe = tmp / ("rerank_guard_cli.exe" if os.name == "nt" else "rerank_guard_cli")
        cmd = [
            gxx,
            "-std=c++17",
            "-DMOZC_RERANK_STANDALONE",
            "-O2",
            str(CLI_SRC),
            str(CLI_MAIN),
            "-o",
            str(exe),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return
        cls.cli = exe

    def _cli(self, op: str, text: str, reading: str = "") -> str:
        if self.cli is None:
            self.skipTest("g++ not available")
        cmd = [str(self.cli), "--op", op]
        if reading:
            cmd.extend(["--reading", reading])
        p = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "MOZC_RERANK_GUARD_MODE": "strict"},
            check=False,
        )
        self.assertEqual(p.returncode, 0, p.stderr.decode("utf-8", "replace"))
        return p.stdout.decode("utf-8")

    def test_skip_parity(self) -> None:
        cases = [
            ("い", "文化", REASON_READING_TOO_SHORT),
            ("きしゃ", "", REASON_CONTEXT_EMPTY_OR_SYMBOL),
            ("きしゃ", "1", REASON_CONTEXT_EMPTY_OR_SYMBOL),
            ("いいんちょう", "文化", REASON_READING_NOT_ELIGIBLE),
            ("きしゃ", "駅に", ""),
        ]
        for reading, ctx, expect in cases:
            with self.subTest(reading=reading, ctx=ctx):
                py = skip_reason(reading, ctx, mode="strict") or ""
                self.assertEqual(py, expect)
                self.assertEqual(self._cli("skip", ctx, reading=reading), expect)

    def test_junk_parity(self) -> None:
        for surface, expect in [("ヨセン", True), ("實際に", True), ("予選", False)]:
            with self.subTest(surface=surface):
                self.assertEqual(is_junk_surface(surface), expect)
                self.assertEqual(self._cli("junk", surface), "1" if expect else "0")


if __name__ == "__main__":
    unittest.main()
