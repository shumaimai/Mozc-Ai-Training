from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.rerank.context_clip import clean_context, normalize_reading
from tools.rerank.contextual_ranking_v2_contract import (
    FORMAT_VERSION,
    format_v1_runtime,
    format_v1_train,
    format_v2,
)


FIXTURE = Path(__file__).parents[2] / "tests/fixtures/contextual_ranking_v2/cases.jsonl"


class Phase0ContractTest(unittest.TestCase):
    def test_fixture_has_required_cases_and_fields(self):
        rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:
            self.assertTrue({"reading", "raw_preceding_text", "cleaned_context", "candidates", "target_segment", "expected_format", "format_version"} <= row.keys())
            self.assertEqual(row["format_version"], FORMAT_VERSION)
            self.assertGreaterEqual(len(row["candidates"]), 3)

    def test_v1_formatters_are_explicitly_different(self):
        train = format_v1_train("きしゃ", "駅に", "汽車")
        runtime = format_v1_runtime("きしゃ", "駅に", "汽車")
        self.assertNotEqual(train, runtime)
        self.assertIn(" [SEP] ", train)
        self.assertIn("\n", runtime)

    def test_context_contract_examples(self):
        self.assertEqual(clean_context("前の文。駅に停まった列車"), "駅に停まった列車")
        self.assertEqual(normalize_reading("キシャ"), "きしゃ")

    def test_v2_formatter_is_explicitly_versioned(self):
        self.assertEqual(FORMAT_VERSION, "contextual-ranking-v2-format-v1")
        self.assertEqual(format_v2("きしゃ", "駅に", "汽車"), format_v1_runtime("きしゃ", "駅に", "汽車"))


if __name__ == "__main__":
    unittest.main()
