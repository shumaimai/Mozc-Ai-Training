"""Shared train/eval/runtime context fixture contract."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.rerank.context_clip import runtime_context_prev
from tools.rerank.train_cross_encoder import build_pair_text


FIXTURE = Path(__file__).parents[2] / "tests/fixtures/contextual_ranking_v2/runtime_context_parity.jsonl"


class RuntimeContextParityTest(unittest.TestCase):
    def test_fixture_matches_training_context_builder(self) -> None:
        for line in FIXTURE.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            actual = runtime_context_prev(case["previous_committed_text"], case["conversion_prefix_top1"])
            self.assertEqual(case["expected_context_prev"], actual, case["case_id"])
            # The training pair formatter must consume the same canonical
            # context value without applying a second, divergent transform.
            prompt = build_pair_text("きしゃ", actual, "記者")
            self.assertIn(f"文脈: {actual}", prompt)

