"""Protocol tests for the resident rerank daemon (no ONNX required)."""

from __future__ import annotations

import socket
import threading
import time
import unittest

from tools.rerank.rerank_daemon import handle_payload, send_request, serve


class FakeScorer:
    def score(self, texts: list[str]) -> list[float]:
        out = []
        for t in texts:
            if "記者" in t:
                out.append(5.0)
            elif "汽車" in t:
                out.append(1.0)
            else:
                out.append(0.0)
        return out


class RerankDaemonTest(unittest.TestCase):
    def test_handle_ping_and_rerank(self) -> None:
        scorer = FakeScorer()
        ping = handle_payload({"op": "ping"}, scorer, tau=0.1, cand_cap=30)
        self.assertTrue(ping.get("ok"))
        resp = handle_payload(
            {
                "reading": "きしゃ",
                "context_prev": "新聞の",
                "nbest": ["汽車", "記者"],
            },
            scorer,
            tau=0.1,
            cand_cap=30,
        )
        self.assertEqual(resp["rerank_top1"], "記者")
        self.assertTrue(resp["overwritten"])
        self.assertEqual(resp["ranked_surfaces"][0], "記者")

    def test_handle_skips_short_reading(self) -> None:
        class Boom:
            def score(self, texts: list[str]) -> list[float]:
                raise AssertionError("should not score")

        resp = handle_payload(
            {"reading": "い", "context_prev": "2", "nbest": ["位", "李"]},
            Boom(),
            tau=0.1,
            cand_cap=30,
        )
        self.assertFalse(resp["overwritten"])
        self.assertEqual(resp["final_top1"], "位")
        self.assertEqual(resp["reason"], "reading_too_short")

    def test_tcp_ndjson_roundtrip(self) -> None:
        scorer = FakeScorer()
        sock_tmp = socket.socket()
        sock_tmp.bind(("127.0.0.1", 0))
        host, port = sock_tmp.getsockname()[:2]
        sock_tmp.close()
        th = threading.Thread(
            target=serve,
            args=(scorer,),
            kwargs={"host": host, "port": port, "tau": 0.1, "cand_cap": 30},
            daemon=True,
        )
        th.start()
        deadline = time.time() + 5.0
        last_err = None
        conn = None
        while time.time() < deadline:
            try:
                ping, conn = send_request(host, port, {"op": "ping"}, timeout=0.5)
                self.assertTrue(ping.get("ok"))
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.05)
        else:
            self.fail(f"daemon did not accept: {last_err}")
        assert conn is not None
        resp, conn = send_request(
            host,
            port,
            {
                "reading": "きしゃ",
                "context_prev": "新聞の",
                "candidates": ["汽車", "記者"],
            },
            timeout=1.0,
            sock=conn,
        )
        conn.close()
        self.assertEqual(resp["final_top1"], "記者")
        self.assertIn("daemon_ms", resp)


if __name__ == "__main__":
    unittest.main()
