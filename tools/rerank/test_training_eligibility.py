from __future__ import annotations

import unittest

from tools.rerank.contextual_ranking_v2_contract import format_v2
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.train_cross_encoder import (
    build_pair_text,
    expand_groups,
    parse_eligibility_statuses,
)


def _row(status: str, gold: str) -> dict[str, object]:
    return {
        "source_id": "fixture:1",
        "reading": "きしゃ",
        "context_prev": "駅に",
        "gold": gold,
        "gold_in_nbest": True,
        "eligibility_status": status,
        "mozc_nbest": [gold, "記者"],
        "mozc_top1": gold,
    }


class TrainingEligibilityTest(unittest.TestCase):
    def test_empty_allow_list_preserves_legacy_selection(self):
        rows = [_row("NEURAL_ELIGIBLE", "汽車"), _row("PROTECTED_EVAL_ONLY", "記者")]
        pairs = expand_groups(rows, require_gold_in_nbest=True)
        self.assertEqual(sum(pair.label for pair in pairs), 2)

    def test_neural_allow_list_excludes_protected_and_coverage_limited(self):
        rows = [
            _row("NEURAL_ELIGIBLE", "汽車"),
            _row("PROTECTED_EVAL_ONLY", "記者"),
            _row("COVERAGE_LIMITED", "CPU"),
        ]
        pairs = expand_groups(
            rows,
            require_gold_in_nbest=True,
            eligibility_statuses=parse_eligibility_statuses("NEURAL_ELIGIBLE"),
        )
        self.assertEqual(sum(pair.label for pair in pairs), 1)
        self.assertIn("候補: 汽車", pairs[0].text)

    def test_eval_filter_retains_only_requested_status(self):
        rows = [_row("NEURAL_ELIGIBLE", "汽車"), _row("COVERAGE_LIMITED", "CPU")]
        groups = prepare_groups(
            rows,
            eligibility_statuses=parse_eligibility_statuses("NEURAL_ELIGIBLE"),
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["eligibility_status"], "NEURAL_ELIGIBLE")

    def test_eval_reads_dataset_v2_candidate_objects(self):
        row = _row("NEURAL_ELIGIBLE", "汽車")
        row.pop("mozc_nbest")
        row.pop("mozc_top1")
        row["candidates"] = [{"surface": "記者"}, {"surface": "汽車"}]
        groups = prepare_groups([row])
        self.assertEqual(groups[0]["candidates"], ["記者", "汽車"])
        self.assertEqual(groups[0]["mozc_top1"], "記者")
        self.assertFalse(groups[0]["mozc_hit1"])

    def test_train_eval_prompt_is_canonical_newline_format(self):
        text = build_pair_text("きしゃ", "駅に", "汽車")
        self.assertEqual(text, format_v2("きしゃ", "駅に", "汽車"))
        self.assertEqual(text, "読み: きしゃ\n文脈: 駅に\n候補: 汽車")
        # Empty context remains an explicit field so train/eval/serving match.
        empty = build_pair_text("きしゃ", "", "汽車")
        self.assertEqual(empty, "読み: きしゃ\n文脈: \n候補: 汽車")
        self.assertNotIn(" [SEP] ", text)

    def test_empty_resume_path_is_not_a_checkpoint(self):
        from pathlib import Path

        raw = ""
        resume = Path(raw) if raw.strip() else None
        self.assertIsNone(resume)
        self.assertFalse(Path("").is_file())

    def test_dataset_v2_candidate_objects_derive_gold_coverage(self):
        row = _row("NEURAL_ELIGIBLE", "汽車")
        row.pop("gold_in_nbest")
        row.pop("mozc_nbest")
        row["candidates"] = [{"surface": "汽車"}, {"surface": "記者"}]
        pairs = expand_groups(
            [row],
            require_gold_in_nbest=True,
            eligibility_statuses={"NEURAL_ELIGIBLE"},
        )
        self.assertEqual(sum(pair.label for pair in pairs), 1)


if __name__ == "__main__":
    unittest.main()
