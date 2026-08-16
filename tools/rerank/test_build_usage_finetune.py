from __future__ import annotations

import unittest

from tools.rerank.build_usage_finetune import build_mix, split_groups, usage_to_group


class UsageFineTuneTest(unittest.TestCase):
    def test_conversion_keeps_mozc_first_and_wanted_scoreable(self) -> None:
        row = {
            "reading": "キシャ",
            "context": "駅に",
            "wanted": "汽車",
            "mozc_top1": "記者",
            "rerank_top1": "汽車",
            "shown": "汽車",
        }
        group = usage_to_group(row, 3)
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group["reading"], "きしゃ")
        self.assertEqual(group["mozc_nbest"][0], "記者")
        self.assertIn("汽車", group["mozc_nbest"])
        self.assertEqual(group["category"], "usage_correction")

    def test_split_preserves_train_for_repeated_reading(self) -> None:
        rows = [
            {"reading": "きしゃ", "usage_index": i} for i in range(4)
        ] + [{"reading": "こうしょう", "usage_index": 10}]
        train, holdout = split_groups(rows, 0.25, 7)
        self.assertTrue(any(r["reading"] == "きしゃ" for r in train))
        self.assertTrue(any(r["reading"] == "きしゃ" for r in holdout))
        self.assertEqual(len(train) + len(holdout), len(rows))

    def test_mix_is_bounded_and_contains_both_sources(self) -> None:
        usage = [
            {
                "reading": "きしゃ",
                "gold": "汽車",
                "source": "ime_usage_local",
                "usage_index": 1,
            }
        ]
        public = [
            {
                "reading": f"r{i}",
                "gold": f"g{i}",
                "mozc_top1": f"g{i}",
                "gold_in_nbest": True,
                "source": "public",
            }
            for i in range(20)
        ]
        mixed = build_mix(usage, public, 10, 0.5, 2)
        self.assertEqual(len(mixed), 10)
        self.assertEqual(sum(r.get("source") == "ime_usage_local" for r in mixed), 5)


if __name__ == "__main__":
    unittest.main()
