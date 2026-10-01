"""Compile shipping diagnostics and policy methods with real Abseil parsing.

Only the Mozc class shell and LOG macro are replaced. The file reader, JSON
helpers, LoadPolicyFile, diagnostics and guard implementation are shipping code.
Requires g++, pkg-config and libabsl-dev; no ONNX model or GPU is needed.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

RUNTIME_ROOT = Path(os.environ.get("MOZCAI_RUNTIME_ROOT",
    str(Path(__file__).resolve().parents[3] / "Mozc-Ai")))


def section(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


class CppReviewContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        compat = RUNTIME_ROOT / "mozc_compat"
        includes = root / "rewriter"
        includes.mkdir()
        for name in ("rerank_diag.h", "rerank_guard.h"):
            (includes / name).write_bytes((compat / name).read_bytes())
        text = (compat / "rerank_rewriter.cc").read_text()
        methods = "\n".join((
            section(text, "bool ReadFileToString(", "void DeleteFileQuiet("),
            section(text, "void RerankRewriter::LoadPolicyFile(",
                    "void RerankRewriter::NoteTimeout("),
            section(text, "bool RerankRewriter::ExtractJsonString(",
                    "bool RerankRewriter::ExtractJsonBool("),
            section(text, "bool RerankRewriter::ExtractJsonDouble(",
                    "bool RerankRewriter::ExtractJsonUint64("),
        ))
        code = r'''
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "rewriter/rerank_diag.h"
#include "rewriter/rerank_guard.h"
#define LOG(level) std::cerr
namespace mozc {
class RerankRewriter {
 public:
  float tau_ = 2.5f;
  int cand_cap_ = 30, timeout_ms_ = 200, max_len_ = 128, context_chars_ = 50;
  void LoadPolicyFile(const std::string& path);
  static bool ExtractJsonString(const std::string&, const char*, std::string*);
  static bool ExtractJsonDouble(const std::string&, const char*, double*);
};
''' + methods + r'''
}  // namespace mozc
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  if (std::string(argv[1]) == "policy") {
    mozc::RerankRewriter rewriter;
    rewriter.LoadPolicyFile(argv[2]);
    std::cout << "{\"tau\":" << rewriter.tau_
      << ",\"cand_cap\":" << rewriter.cand_cap_
      << ",\"timeout_ms\":" << rewriter.timeout_ms_
      << ",\"max_len\":" << rewriter.max_len_
      << ",\"context_clip_max_chars\":" << rewriter.context_chars_ << "}";
    return 0;
  }
  std::string payload, reason, stage = "rewrite", result = "ok";
  std::getline(std::cin, payload);
  mozc::RerankRewriter::ExtractJsonString(payload, "reason", &reason);
  mozc::RerankRewriter::ExtractJsonString(payload, "stage", &stage);
  mozc::RerankRewriter::ExtractJsonString(payload, "daemon_result", &result);
  setenv("MOZC_RERANK_DIAG_LOG", argv[2], 1);
  setenv("MOZC_RERANK_DIAG_SUMMARY_EVERY", "200", 1);
  mozc::rerank::DiagEvent event;
  event.stage = stage.c_str();
  event.daemon_result = result.c_str();
  event.reason = reason.c_str();
  if (std::string(argv[1]) == "diag-null") {
    event.stage = nullptr;
    event.daemon_result = nullptr;
    event.reason = nullptr;
  }
  mozc::rerank::AppendDiagEvent(event);
  mozc::rerank::FlushDiagSummary();
  std::ifstream log(argv[2]);
  std::cout << log.rdbuf();
}
'''
        source = root / "contract.cc"
        source.write_text(code)
        cls.binary = root / "contract"
        flags = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "absl_strings"], text=True))
        built = subprocess.run(["g++", "-std=c++17", "-O2", "-pthread",
            "-DMOZC_RERANK_STANDALONE", "-I", str(root), str(source),
            str(compat / "rerank_diag.cc"), str(compat / "rerank_guard.cc"),
            "-o", str(cls.binary), *flags], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)

    def policy(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(text)
            return json.loads(subprocess.check_output(
                [str(self.binary), "policy", str(path)], text=True))

    def diagnostics(self, payload, mode="diag"):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([str(self.binary), mode, str(Path(tmp)/"diag.jsonl")],
                input=json.dumps(payload, ensure_ascii=False) + "\n", text=True,
                capture_output=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2, "one event and one summary; JSONL must stay intact")
        return result.stdout, [json.loads(line) for line in lines]

    def test_installed_policy_reads_real_tau_and_all_numeric_fields(self):
        text = (RUNTIME_ROOT / "runtime/model/margin_policy.json").read_text()
        expected = json.loads(text)
        actual = self.policy(text)
        for key in actual:
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual["tau"], 1.5)

    def test_policy_numbers_with_following_keys_update_every_setting(self):
        expected = dict(tau=1.25, cand_cap=17, timeout_ms=175,
                        max_len=96, context_clip_max_chars=41)
        self.assertEqual(self.policy(json.dumps(expected)), expected)

    def test_policy_supports_crlf_and_scientific_numeric_tokens(self):
        actual = self.policy('{\r\n"tau":\t1.5e0, "cand_cap": 12, "timeout_ms": 180}')
        self.assertEqual(actual["tau"], 1.5)
        self.assertEqual(actual["cand_cap"], 12)
        self.assertEqual(actual["timeout_ms"], 180)

    def test_missing_and_invalid_policy_fields_keep_defaults(self):
        for tau in ("1.5", None, True, "not-a-number"):
            with self.subTest(tau=tau):
                actual = self.policy(json.dumps(dict(tau=tau, cand_cap=0)))
                self.assertEqual(actual, dict(tau=2.5, cand_cap=30,
                    timeout_ms=200, max_len=128, context_clip_max_chars=50))

    def test_known_response_reasons_are_retained(self):
        for reason in ("", "reading_too_short", "context_empty_or_symbol",
                       "reading_not_eligible", "junk_candidate"):
            with self.subTest(reason=reason):
                _, lines = self.diagnostics(dict(reason=reason))
                self.assertEqual(lines[0]["reason"], reason)
                self.assertEqual(lines[1]["daemon_ok"], 1)

    def test_unknown_response_reason_never_logs_reading_or_context(self):
        marker = "きしゃ 文脈は新聞社の取材"
        raw, lines = self.diagnostics(dict(reason=marker))
        self.assertNotIn(marker, raw)
        self.assertEqual(lines[0]["reason"], "unknown")

    def test_response_control_characters_do_not_break_jsonl(self):
        raw, lines = self.diagnostics(dict(reason='きしゃ\n駅に\r\t"\\'))
        self.assertNotIn("きしゃ", raw)
        self.assertNotIn("駅に", raw)
        self.assertEqual(lines[0]["reason"], "unknown")

    def test_reason_prefix_and_other_string_fields_are_allowlisted(self):
        raw, lines = self.diagnostics(dict(reason="reading_too_short\n本文",
            stage="私の文脈\n", daemon_result="私の候補\t"))
        self.assertNotIn("本文", raw)
        self.assertEqual(lines[0]["reason"], "unknown")
        self.assertEqual(lines[0]["stage"], "unknown")
        self.assertEqual(lines[0]["daemon_result"], "unknown")
        self.assertEqual(lines[1]["rewrite_calls"], 0)

    def test_null_diagnostic_fields_are_safe(self):
        _, lines = self.diagnostics({}, "diag-null")
        self.assertEqual(lines[0]["reason"], "")
        self.assertEqual(lines[0]["stage"], "")
        self.assertEqual(lines[0]["daemon_result"], "")


if __name__ == "__main__":
    unittest.main()
