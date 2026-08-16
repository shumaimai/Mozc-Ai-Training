from __future__ import annotations

import unittest

from tools.rerank.privacy import ensure_public_modal_paths


class ModalPrivacyTest(unittest.TestCase):
    def test_public_dataset_is_allowed(self) -> None:
        ensure_public_modal_paths(
            "data/rerank_ctx/train_v2.jsonl,data/rerank_ctx/eval_v2.jsonl",
            datasets=True,
        )

    def test_nonstaged_dataset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensure_public_modal_paths("data/private/train.jsonl", datasets=True)

    def test_usage_and_log_paths_are_rejected(self) -> None:
        for path in (
            "/artifacts/private/model",
            "/artifacts/personal-v1",
            "/artifacts/usage30m_v1",
            "data/rerank_ctx/ime_usage_pairs.jsonl",
            "/artifacts/eval/conversion.log",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ensure_public_modal_paths(path)


if __name__ == "__main__":
    unittest.main()
