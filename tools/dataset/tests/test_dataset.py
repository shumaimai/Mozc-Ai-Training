from __future__ import annotations

import unittest

from tools.dataset.aozora import ruby_records
from tools.dataset.classify import classify
from tools.dataset.deepseek_review import ReviewBudget, build_request
from tools.dataset.normalize import is_valid_reading, normalize_reading
from tools.dataset.records import Provenance, TermRecord


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = TermRecord(
            surface="人工呼吸器",
            reading="じんこうこきゅうき",
            category="technical_term",
            provenance=Provenance("test", "https://example.invalid", "test", "2026-08-06T00:00:00Z"),
            reading_source="fixture",
            reading_confidence="official",
        )

    def test_normalize_reading(self) -> None:
        self.assertEqual(normalize_reading("ジンコウ・コキュウキ"), "じんこうこきゅうき")
        self.assertEqual(normalize_reading("ケヵヶ"), "けかけ")
        self.assertTrue(is_valid_reading("じんこうこきゅうき"))
        self.assertFalse(is_valid_reading("abc"))

    def test_extracts_aozora_ruby(self) -> None:
        records = ruby_records("｜今日《きょう》は晴れ。", "fixture", "https://example.invalid")
        self.assertEqual([(record.surface, record.reading) for record in records], [("今日", "きょう")])

    def test_classifies_existing_top_five_as_abstain(self) -> None:
        result = classify(self.record, ["人工呼吸器", "人工呼吸機"])
        self.assertEqual(result.action, "abstain")
        self.assertEqual(result.gold_rank, 1)

    def test_classifies_low_rank_as_rerank(self) -> None:
        result = classify(self.record, ["候補" + str(index) for index in range(6)] + ["人工呼吸器"])
        self.assertEqual(result.action, "rerank")
        self.assertEqual(result.gold_rank, 7)

    def test_classifies_missing_fixed_term_as_dictionary_gap(self) -> None:
        result = classify(self.record, ["人工呼吸機"])
        self.assertEqual(result.action, "dictionary_gap")

    def test_review_request_is_structured(self) -> None:
        comparison = classify(self.record, ["人工呼吸機"])
        request = build_request("deepseek-chat", comparison.to_dict())
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_budget_caps_requests(self) -> None:
        budget = ReviewBudget(1.0, 1.0, 1.0)
        self.assertTrue(budget.can_afford(100_000, 100_000))
        exhausted = budget.charge(500_000, 500_000)
        self.assertFalse(exhausted.can_afford(1, 1))


if __name__ == "__main__":
    unittest.main()
