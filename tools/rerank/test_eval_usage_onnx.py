from __future__ import annotations

import unittest

from tools.rerank.eval_usage_onnx import evaluate_scored


class UsageOnnxEvalTest(unittest.TestCase):
    def test_margin_and_safety_skip(self) -> None:
        groups = [
            {
                "reading": "きしゃ",
                "context_prev": "駅に",
                "candidates": ["記者", "汽車"],
                "mozc_top1": "記者",
                "gold": "汽車",
            },
            {
                "reading": "い",
                "context_prev": "3",
                "candidates": ["位", "李"],
                "mozc_top1": "位",
                "gold": "位",
            },
        ]
        scores = [[0.0, 3.0], [0.0, 5.0]]
        result = evaluate_scored(groups, scores, tau=2.5, policy="safety")
        self.assertEqual(result["n_helped"], 1)
        self.assertEqual(result["n_hurt"], 0)
        self.assertEqual(result["n_skip"], 1)
        self.assertEqual(result["net_pt_vs_mozc"], 50.0)


if __name__ == "__main__":
    unittest.main()
