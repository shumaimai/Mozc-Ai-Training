from __future__ import annotations

import unittest

from tools.dataset.aozora import public_domain_works, ruby_records, strip_boilerplate
from tools.dataset.classify import classify
from tools.dataset.mozc_batch import load_env_file, merge, parse_candidates_tsv, readings_from_records
from tools.dataset.deepseek_review import ReviewBudget, build_request
from tools.dataset.normalize import is_valid_reading, normalize_reading
from tools.dataset.records import Provenance, TermRecord
from tools.dataset.wikidata import records_from_bindings


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

    def test_strip_boilerplate_removes_notes_and_colophon(self) -> None:
        text = "\n".join(
            [
                "羅生門",
                "芥川龍之介",
                "",
                "-------------------------------------------------------",
                "【テキスト中に現れる記号について】",
                "《》：ルビ",
                "-------------------------------------------------------",
                "",
                "｜下人《げにん》が羅生門《らしょうもん》の下にいた。",
                "",
                "底本：「芥川龍之介全集1」ちくま文庫",
                "入力：ボランティア",
            ]
        )
        body = strip_boilerplate(text)
        self.assertIn("下人", body)
        self.assertNotIn("《》：ルビ", body)
        self.assertNotIn("底本", body)
        records = ruby_records(body, "fixture", "https://example.invalid")
        pairs = {(record.surface, record.reading) for record in records}
        self.assertIn(("下人", "げにん"), pairs)
        self.assertIn(("羅生門", "らしょうもん"), pairs)

    def test_ruby_context_capture(self) -> None:
        records = ruby_records("｜今日《きょう》は晴れ。", "fixture", "https://example.invalid", capture_context=True)
        self.assertEqual(records[0].metadata["context"], "｜今日《きょう》は晴れ。")

    def test_public_domain_filter_uses_both_flags(self) -> None:
        rows = [
            {
                "作品ID": "1",
                "作品名": "公有",
                "作品名読み": "こうゆう",
                "人物ID": "10",
                "姓": "青空",
                "名": "太郎",
                "文字遣い種別": "新字新仮名",
                "作品著作権フラグ": "なし",
                "人物著作権フラグ": "なし",
                "テキストファイルURL": "https://example.invalid/a.zip",
                "テキストファイル符号化方式": "ShiftJIS",
                "図書カードURL": "https://example.invalid/card1",
                "公開日": "2000-01-01",
                "底本名1": "底本A",
            },
            {"作品ID": "2", "作品著作権フラグ": "あり", "人物著作権フラグ": "なし", "テキストファイルURL": "https://example.invalid/b.zip"},
            {"作品ID": "3", "作品著作権フラグ": "なし", "人物著作権フラグ": "なし", "テキストファイルURL": ""},
        ]
        works = public_domain_works(rows)
        self.assertEqual([work["work_id"] for work in works], ["1"])
        self.assertEqual(works[0]["author"], "青空太郎")

    def test_wikidata_bindings_parse_and_filter(self) -> None:
        bindings = [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q1000516"},
                "itemLabel": {"value": "厄神駅"},
                "kana": {"value": "ヤクジンエキ"},
                "typeLabel": {"value": "鉄道駅"},
            },
            {  # duplicate surface/reading (katakana vs already-normalized) is de-duplicated
                "item": {"value": "http://www.wikidata.org/entity/Q1000516"},
                "itemLabel": {"value": "厄神駅"},
                "kana": {"value": "やくじんえき"},
                "typeLabel": {"value": "地上駅"},
            },
            {  # invalid reading (contains kanji) is dropped
                "item": {"value": "http://www.wikidata.org/entity/Q999"},
                "itemLabel": {"value": "変な駅"},
                "kana": {"value": "変な"},
            },
        ]
        records = records_from_bindings(bindings, retrieved_at="2026-08-06T00:00:00Z")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record.surface, record.reading), ("厄神駅", "やくじんえき"))
        self.assertEqual(record.reading_confidence, "unverified")
        self.assertEqual(record.provenance.license_id, "CC0-1.0")
        self.assertEqual(record.metadata["qid"], "Q1000516")

    def test_mozc_batch_keys_and_merge_feed_classify(self) -> None:
        records = [
            {
                "surface": "人工呼吸器",
                "reading": "じんこうこきゅうき",
                "category": "technical_term",
                "provenance": {"source_id": "s", "source_url": "u", "license_id": "l", "retrieved_at": "t"},
                "reading_source": "fixture",
                "reading_confidence": "official",
                "metadata": {"context": "病室の人工呼吸器"},
            },
            {  # duplicate reading is collapsed for the batch key list
                "surface": "人工呼吸器",
                "reading": "じんこうこきゅうき",
                "category": "technical_term",
                "provenance": {"source_id": "s", "source_url": "u", "license_id": "l", "retrieved_at": "t"},
                "reading_source": "fixture",
                "reading_confidence": "official",
                "metadata": {},
            },
        ]
        keys = readings_from_records(records)
        self.assertEqual(keys, ["じんこうこきゅうき"])

        tsv = ["じんこうこきゅうき\t人工呼吸機\t人工呼吸器\n"]
        candidates = parse_candidates_tsv(tsv)
        merged = merge(records, candidates)
        self.assertEqual(merged[0]["candidates"], ["人工呼吸機", "人工呼吸器"])
        self.assertEqual(merged[0]["context"], ["病室の人工呼吸器"])

        # The merged row is directly consumable by classify.
        record = TermRecord.from_dict(merged[0]["record"])
        result = classify(record, merged[0]["candidates"], merged[0]["context"])
        self.assertEqual(result.gold_rank, 2)
        self.assertEqual(result.action, "abstain")  # gold within top_k=5

    def test_load_env_file_ignores_comments(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mozc_batch.env"
            path.write_text(
                "# comment\nMOZC_BATCH_EXE=C:\\bin\\mozc_batch.exe\nMOZC_MAX_CANDIDATES=50\n",
                encoding="utf-8",
            )
            values = load_env_file(path)
            self.assertEqual(values["MOZC_BATCH_EXE"], r"C:\bin\mozc_batch.exe")
            self.assertEqual(values["MOZC_MAX_CANDIDATES"], "50")

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

    def test_export_train_prompt_matches_ime_shape(self) -> None:
        from tools.dataset.export_train import build_ime_prompt, review_to_example

        prompt = build_ime_prompt("きょう", ["きょう", "九曜"], ["昨日", "は"])
        self.assertIn("直前の入力: 昨日, は", prompt)
        self.assertIn("現在の入力: きょう", prompt)
        self.assertIn("既存候補（これら以外を提案）: きょう, 九曜", prompt)

        row = {
            "review": {"decision": "accept", "confidence": 0.9, "reason_code": "ok"},
            "comparison": {
                "candidates": ["きょう", "九曜"],
                "context": ["昨日"],
                "record": {
                    "reading": "きょう",
                    "surface": "今日",
                    "category": "test",
                    "provenance": {"source_id": "test"},
                },
            },
        }
        example = review_to_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["output"], "今日")
        self.assertIn("今日", example["text"])


if __name__ == "__main__":
    unittest.main()
