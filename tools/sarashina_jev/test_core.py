from __future__ import annotations

import unittest

from tools.sarashina_jev.data import PageExample, encode_candidate_sequence, row_to_pages, shuffle_gold_page
from tools.sarashina_jev.util import select_even_layers
from tools.sarashina_jev.qat_utils import (
    FakeQuantEmbedding,
    FakeQuantLinear,
    set_qat_strength,
)


class LayerSelectionTest(unittest.TestCase):
    def test_24_to_12(self):
        idx = select_even_layers(24, 12)
        self.assertEqual(len(idx), 12)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 23)
        self.assertEqual(len(set(idx)), 12)


class PageBuilderTest(unittest.TestCase):
    def test_gold_second_page_and_first_anchor(self):
        row = {
            "reading": "きしゃ",
            "context_prev": "新聞の",
            "gold": "貴社",
            "mozc_nbest": ["記者", "汽車", "帰社", "喜捨", "記社", "貴社", "記写"],
        }
        pages = row_to_pages(row, page_size=5, anchor_weight=0.25)
        self.assertEqual(len(pages), 2)
        self.assertFalse(pages[0].is_gold_page)
        self.assertEqual(pages[0].target, 0)
        self.assertTrue(pages[1].is_gold_page)
        self.assertEqual(pages[1].target, 0)

    def test_gold_first_page_has_no_extra_anchor(self):
        row = {
            "reading": "きしゃ",
            "context_prev": "新聞の",
            "gold": "記者",
            "mozc_nbest": ["記者", "汽車", "貴社", "帰社", "喜捨"],
        }
        pages = row_to_pages(row, page_size=5)
        self.assertEqual(len(pages), 1)
        self.assertTrue(pages[0].is_gold_page)
        self.assertEqual(pages[0].target, 0)

    def test_shuffle_gold_page_remaps_target_and_preserves_gold(self):
        item = PageExample(
            reading="きしゃ",
            context="新聞の",
            candidates=("記者", "汽車", "貴社", "帰社", "喜捨"),
            target=0,
            weight=1.0,
            is_gold_page=True,
        )
        shuffled = shuffle_gold_page(item, seed=42, epoch=1, index=7)
        self.assertEqual(shuffled.candidates[shuffled.target], "記者")
        self.assertCountEqual(shuffled.candidates, item.candidates)

    def test_anchor_page_is_not_shuffled(self):
        item = PageExample(
            reading="きしゃ",
            context="新聞の",
            candidates=("記者", "汽車", "貴社", "帰社", "喜捨"),
            target=0,
            weight=0.25,
            is_gold_page=False,
        )
        shuffled = shuffle_gold_page(item, seed=42, epoch=9, index=7)
        self.assertEqual(shuffled, item)


    def test_truncation_preserves_candidate_reading_and_decision(self):
        class FakeTokenizer:
            bos_token_id = 101
            pad_token_id = 0
            eos_token_id = 102

            def encode(self, text, add_special_tokens=False):
                return [1000 + ord(ch) for ch in text]

        tok = FakeTokenizer()
        context_ids = [2000 + i for i in range(100)]
        ids, mask = encode_candidate_sequence(
            tok,
            reading="きしゃ",
            context_ids=context_ids,
            candidate="記者",
            max_length=40,
        )
        used = [token for token, keep in zip(ids, mask) if keep]
        prefix = tok.encode("候補: 記者\n読み: きしゃ\n文脈: ", add_special_tokens=False)
        suffix = tok.encode("\n判定:", add_special_tokens=False)

        self.assertEqual(used[0], tok.bos_token_id)
        self.assertEqual(used[1 : 1 + len(prefix)], prefix)
        self.assertEqual(used[-len(suffix) :], suffix)
        self.assertNotIn(context_ids[0], used)
        self.assertIn(context_ids[-1], used)

    def test_different_candidates_remain_different_after_truncation(self):
        class FakeTokenizer:
            bos_token_id = 101
            pad_token_id = 0
            eos_token_id = 102

            def encode(self, text, add_special_tokens=False):
                return [1000 + ord(ch) for ch in text]

        tok = FakeTokenizer()
        context_ids = [2000 + i for i in range(100)]
        a, _ = encode_candidate_sequence(
            tok,
            reading="きしゃ",
            context_ids=context_ids,
            candidate="記者",
            max_length=40,
        )
        b, _ = encode_candidate_sequence(
            tok,
            reading="きしゃ",
            context_ids=context_ids,
            candidate="汽車",
            max_length=40,
        )
        self.assertNotEqual(a, b)



class QATUtilsTest(unittest.TestCase):
    def test_fake_quant_linear_keeps_state_dict_names(self):
        import torch
        from torch import nn

        base = nn.Linear(4, 3)
        qat = FakeQuantLinear.from_linear(base)
        self.assertEqual(set(qat.state_dict().keys()), {"weight", "bias"})
        x = torch.randn(2, 4)
        y = qat(x)
        self.assertEqual(tuple(y.shape), (2, 3))

    def test_fake_quant_embedding_keeps_state_dict_names(self):
        import torch
        from torch import nn

        base = nn.Embedding(16, 4)
        qat = FakeQuantEmbedding.from_embedding(base)
        self.assertEqual(set(qat.state_dict().keys()), {"weight"})
        set_qat_strength(qat, 1.0)
        y = qat(torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(y.shape), (1, 3, 4))


if __name__ == "__main__":
    unittest.main()
