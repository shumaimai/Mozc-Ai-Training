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
from tools.rerank.contextual_ranking_v2_schema import (
    FORMAT_VERSION as RECORD_FORMAT_VERSION,
    SCHEMA_VERSION,
    validate_record,
)


FIXTURE = Path(__file__).parents[2] / "tests/fixtures/contextual_ranking_v2/cases.jsonl"
REPLAY_FIXTURE = FIXTURE.parent / "actual_mozc_multisegment_replay.json"


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

    def test_actual_mozc_multisegment_replay_fixture(self):
        row = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(row["result"], "actual_converter_multisegment_replay_pass")
        self.assertEqual(row["scored_segment_index"], row["target_segment_index"])
        self.assertEqual(row["commit_operation"]["committed_candidate"], "汽車")

    def test_phase1_record_schema_rejects_surface_only_records(self):
        record = {
            "schema_version": SCHEMA_VERSION,
            "format_version": RECORD_FORMAT_VERSION,
            "source_id": "fixture:1",
            "reading": "きしゃ",
            "context_prev": "駅に",
            "gold": "汽車",
            "target_segment_index": 1,
            "candidates": [{"surface": "汽車"}],
        }
        self.assertIn("candidate_missing:attributes", validate_record(record))

    def test_phase1_record_schema_accepts_metadata_complete_record(self):
        record = {
            "schema_version": SCHEMA_VERSION,
            "format_version": RECORD_FORMAT_VERSION,
            "source_id": "fixture:1",
            "reading": "きしゃ",
            "context_prev": "駅に",
            "gold": "汽車",
            "target_segment_index": 1,
            "candidates": [{
                "surface": "汽車", "rank": 0, "cost": 100, "cost_delta": 0,
                "lid": 1, "rid": 2, "attributes": 0, "category": "DEFAULT",
                "converted_segment_count": 1, "protection": "NORMAL",
            }],
        }
        self.assertEqual(validate_record(record), [])

    def test_phase1_record_schema_keeps_coverage_failure_as_eval_record(self):
        record = {
            "schema_version": SCHEMA_VERSION,
            "format_version": RECORD_FORMAT_VERSION,
            "source_id": "fixture:coverage",
            "reading": "きしゃ",
            "context_prev": "駅に",
            "gold": "未知表記",
            "target_segment_index": 0,
            "example_status": "COVERAGE_FAILURE",
            "example_reason": "gold_not_in_top_k",
            "candidates": [{
                "surface": "汽車", "rank": 0, "cost": 100, "cost_delta": 0,
                "lid": 1, "rid": 2, "attributes": 0, "category": "DEFAULT",
                "converted_segment_count": 1, "protection": "NORMAL", "wcost": 80,
            }],
        }
        self.assertEqual(validate_record(record), [])

    def test_runtime_parity_metadata_is_validated_when_present(self):
        record = {
            "schema_version": SCHEMA_VERSION,
            "format_version": RECORD_FORMAT_VERSION,
            "source_id": "fixture:runtime",
            "reading": "きしゃ",
            "context_prev": "駅に電車で",
            "gold": "汽車",
            "target_segment_index": 1,
            "conversion_segments_size": 2,
            "sampling_identity": "a" * 64,
            "context_builder": "runtime_preceding_text_plus_conversion_top1-v1",
            "candidates": [{
                "surface": "汽車", "rank": 0, "cost": 100, "cost_delta": 0,
                "lid": 1, "rid": 2, "attributes": 0, "category": "DEFAULT",
                "converted_segment_count": 1, "protection": "NORMAL",
            }],
        }
        self.assertEqual(validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
