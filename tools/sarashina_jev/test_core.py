from __future__ import annotations

import unittest

from tools.sarashina_jev.data import PageExample, row_to_pages, shuffle_gold_page
from tools.sarashina_jev.util import select_even_layers


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


if __name__ == "__main__":
    unittest.main()
