from __future__ import annotations

import unittest

from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.train_cross_encoder import expand_groups, parse_eligibility_statuses


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


if __name__ == "__main__":
    unittest.main()
