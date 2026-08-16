// Copyright 2026 AI Mozc IME Project
// RerankRewriter — reorder Mozc N-best via margin-gated cross-encoder.
//
// Design (docs/NEXT_TASK_PHASE3_CTX.md):
//   conversion_segment(0) + history context_prev (C++ context_clip)
//     → score (ONNX fp32 preferred; interim: Python phase3_hook bridge)
//     → margin gate τ from margin_policy.json (default 2.5)
//     → rewrite candidate order (NOT append; parallel to AIRewriter)
//
// Runtime (enabled by default in the all-in-one v1.0 package):
//   MOZC_RERANK_ENABLED=0          # optional administrator kill switch
//   MOZC_RERANK_DAEMON_ADDR=127.0.0.1:17890  # resident Python daemon (default)
//   MOZC_RERANK_HOOK_CMD=<bridge>            # legacy one-shot; used if set
//   MOZC_RERANK_POLICY=<margin_policy.json>
// Optional:
//   MOZC_RERANK_TAU=2.5
//   MOZC_RERANK_CAND_CAP=30
//   MOZC_RERANK_TIMEOUT_MS=200
//   MOZC_RERANK_LOG=<jsonl path>
//   MOZC_RERANK_GUARD=1          # usage whitelist+heuristics (default ON)
//
// Usage guard (NEXT_TASK_USAGE_GUARD): skip daemon/hook unless the reading is
// context-sensitive, length>2, and context has linguistic content.
//
// Fail-safe: daemon down / timeout / exception → Mozc order, never block IME.

#ifndef MOZC_REWRITER_RERANK_REWRITER_H_
#define MOZC_REWRITER_RERANK_REWRITER_H_

#include <mutex>
#include <string>
#include <vector>

#include "rewriter/rewriter_interface.h"

namespace mozc {

class RerankRewriter : public RewriterInterface {
 public:
  RerankRewriter();
  ~RerankRewriter() override;

  int capability(const ConversionRequest& request) const override;

  // Reorder conversion_segment(0). Prototype bridge may block.
  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

  // Opt-in conversion log append when MOZC_RERANK_LOG is set.
  void Finish(const ConversionRequest& request,
              const Segments& segments) override;

  void Clear() override;

  // Ship defaults (do not enable int8).
  static constexpr float kDefaultTau = 2.5f;
  static constexpr int kDefaultCandCap = 30;
  static constexpr int kDefaultMaxLen = 128;
  static constexpr int kDefaultTimeoutMs = 200;
  static constexpr int kDefaultContextChars = 50;

  bool IsEnabled() const { return enabled_; }

 private:
  struct HookResult {
    std::vector<std::string> ranked_surfaces;
    std::string rerank_top1;
    std::string final_top1;
    bool overwritten = false;
  };

  struct PendingLog {
    std::string reading;
    std::vector<std::string> nbest;
    std::string context_prev;
    std::string rerank_top1;
    std::string final_top1;
    bool overwritten = false;
    float tau = kDefaultTau;
  };

  void LoadConfigFromEnv();
  void LoadPolicyFile(const std::string& path);
  void NoteTimeout() const;
  void NoteSuccess() const;
  int EffectiveCandCap() const;
  int EffectiveContextChars() const;
  bool DegradeDisabled() const;
  bool CallHook(const std::string& reading,
                const std::vector<std::string>& nbest,
                const std::string& context_prev,
                HookResult* out) const;
  bool CallDaemon(const std::string& reading,
                  const std::vector<std::string>& nbest,
                  const std::string& context_prev,
                  HookResult* out) const;
  bool CallHookWithTimeout(const std::string& reading,
                           const std::vector<std::string>& nbest,
                           const std::string& context_prev,
                           HookResult* out) const;
  static bool ParseDaemonAddr(const std::string& addr, std::string* host,
                              int* port);
  static bool ReorderSegment(Segment* segment,
                             const std::vector<std::string>& ranked);
  static std::string EscapeJson(const std::string& s);
  static bool WriteRequestJson(const std::string& path,
                               const std::string& reading,
                               const std::vector<std::string>& nbest,
                               const std::string& context_prev);
  static bool ParseHookResponse(const std::string& json, HookResult* out);
  static bool ExtractJsonString(const std::string& json, const char* key,
                                std::string* value);
  static bool ExtractJsonBool(const std::string& json, const char* key,
                              bool* value);
  static bool ExtractJsonStringArray(const std::string& json, const char* key,
                                     std::vector<std::string>* values);
  void AppendConversionLog(const PendingLog& pending,
                           const std::string& chosen) const;

  bool enabled_ = false;
  std::string hook_cmd_;
  std::string daemon_addr_;
  std::string policy_path_;
  float tau_ = kDefaultTau;
  int cand_cap_ = kDefaultCandCap;
  int max_len_ = kDefaultMaxLen;
  int timeout_ms_ = kDefaultTimeoutMs;
  int context_chars_ = kDefaultContextChars;
  std::string log_path_;

  mutable std::mutex pending_mutex_;
  mutable PendingLog pending_log_;
  mutable bool has_pending_log_ = false;
  mutable int consecutive_timeouts_ = 0;
  mutable int consecutive_ok_ = 0;
  mutable int degrade_tier_ = 0;  // 0=full .. 4=Mozc-only
};

}  // namespace mozc

#endif  // MOZC_REWRITER_RERANK_REWRITER_H_
