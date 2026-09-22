from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.dataset.jsonl import read_jsonl, write_jsonl


class JsonlTest(unittest.TestCase):
    def test_gzip_round_trip(self):
        rows = [{"reading": "きしゃ", "gold": "汽車"}, {"reading": "きしゃ", "gold": "記者"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl.gz"
            self.assertEqual(write_jsonl(path, rows), 2)
            self.assertEqual(list(read_jsonl(path)), rows)


if __name__ == "__main__":
    unittest.main()
