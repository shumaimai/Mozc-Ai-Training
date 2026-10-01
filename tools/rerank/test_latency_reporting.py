"""Exercise deadline and native fallback accounting in the TCP benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.rerank import daemon_latency_bench as bench


class LatencyReportingTest(unittest.TestCase):
    def test_timeout_counts_native_hit_and_slow_latency(self):
        rows=[{"reading":"きしゃ","context_prev":"新聞の","gold":"記者",
            "candidates":[{"surface":"記者","protection":"NORMAL"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"report.json"
            argv=["bench","--data","unused","--warmup","0","--out",str(out)]
            with patch("sys.argv",argv),patch.object(bench,"read_jsonl",return_value=iter(rows)), \
                patch.object(bench,"one_request",return_value=(False,200.5,None)):
                self.assertEqual(bench.main(),0)
            report=json.loads(out.read_text())
            self.assertEqual(report["mozc_hit1"],1.)
            self.assertEqual(report["daemon_final_hit1"],1.)
            self.assertEqual(report["timeouts"],1)
            self.assertEqual(report["roundtrip_ms"]["p50"],200.5)

    def test_cpp_protection_reverts_a_daemon_overwrite(self):
        rows=[{"reading":"きしゃ","context_prev":"新聞の","gold":"記者",
            "candidates":[{"surface":"記者","protection":"HARD_PROTECT"},
                {"surface":"汽車","protection":"NORMAL"}]}]
        response={"ok":True,"final_top1":"汽車","overwritten":True,"daemon_ms":25}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"report.json"
            argv=["bench","--data","unused","--warmup","0","--out",str(out)]
            with patch("sys.argv",argv),patch.object(bench,"read_jsonl",return_value=iter(rows)), \
                patch.object(bench,"one_request",return_value=(True,26,response)):
                bench.main()
            report=json.loads(out.read_text())
            self.assertEqual(report["hurt"],0)
            self.assertEqual(report["daemon_final_hit1"],1.)


if __name__=="__main__":unittest.main()
