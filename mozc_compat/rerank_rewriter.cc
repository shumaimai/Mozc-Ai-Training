// Copyright 2026 AI Mozc IME Project
// RerankRewriter — see docs/NEXT_TASK_PHASE3.md / docs/RERANK_HOOK.md

#include "rewriter/rerank_rewriter.h"
#include "rewriter/context_clip.h"
#include "rewriter/rerank_guard.h"
#include "rewriter/rerank_margin.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "converter/segments.h"
#include "request/conversion_request.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#pragma comment(lib, "ws2_32.lib")
#define MOZC_RERANK_GETPID() _getpid()
using RerankSocket = SOCKET;
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define MOZC_RERANK_GETPID() getpid()
using RerankSocket = int;
#endif

namespace mozc {
namespace {

std::string GetEnvOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return v == nullptr ? std::string() : std::string(v);
}

bool EnvTruthy(const std::string& v) {
  if (v.empty()) {
    return false;
  }
  std::string lower = std::string(absl::AsciiStrToLower(v));
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

// Optional privacy-safe runtime diagnostics.  This deliberately records only
// byte/count metadata and fixed stage names: never surrounding text, readings,
// candidates, or any other user-provided string.
void AppendPrivacySafeDiag(const char* stage, size_t request_context_bytes,
                           size_t history_bytes, size_t clean_context_bytes,
                           size_t reading_bytes, size_t candidate_count,
                           size_t history_segment_count,
                           size_t conversion_segment_count) {
  const std::string path = GetEnvOrEmpty("MOZC_RERANK_DIAG_LOG");
  if (path.empty()) {
    return;
  }
  static std::mutex diag_mutex;
  std::lock_guard<std::mutex> lock(diag_mutex);
  std::ofstream out(path, std::ios::app | std::ios::binary);
  if (!out) {
    return;
  }
  out << "{\"stage\":\"" << stage << "\",\"request_context_bytes\":"
      << request_context_bytes << ",\"history_bytes\":" << history_bytes
      << ",\"clean_context_bytes\":" << clean_context_bytes
      << ",\"reading_bytes\":" << reading_bytes
      << ",\"candidate_count\":" << candidate_count
      << ",\"history_segment_count\":" << history_segment_count
      << ",\"conversion_segment_count\":" << conversion_segment_count
      << "}\n";
}

std::string TempDir() {
#ifdef _WIN32
  std::string t = GetEnvOrEmpty("TEMP");
  if (t.empty()) {
    t = GetEnvOrEmpty("TMP");
  }
  if (t.empty()) {
    t = ".";
  }
  return t;
#else
  std::string t = GetEnvOrEmpty("TMPDIR");
  return t.empty() ? std::string("/tmp") : t;
#endif
}

std::string JoinPath(const std::string& dir, const std::string& name) {
#ifdef _WIN32
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (dir.empty()) {
    return name;
  }
  if (dir.back() == '/' || dir.back() == '\\') {
    return dir + name;
  }
  return dir + sep + name;
}

bool ReadFileToString(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

void DeleteFileQuiet(const std::string& path) {
  std::remove(path.c_str());
}

#ifdef _WIN32
constexpr RerankSocket kInvalidRerankSocket = INVALID_SOCKET;
#else
constexpr RerankSocket kInvalidRerankSocket = -1;
#endif

void CloseRerankSocket(RerankSocket s) {
#ifdef _WIN32
  if (s != INVALID_SOCKET) {
    closesocket(s);
  }
#else
  if (s >= 0) {
    close(s);
  }
#endif
}

void EnsureWinsock() {
#ifdef _WIN32
  static std::once_flag once;
  std::call_once(once, []() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
  });
#endif
}

bool WaitSocket(RerankSocket s, bool for_write, int timeout_ms) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  const int nfds =
#ifdef _WIN32
      0;
#else
      static_cast<int>(s) + 1;
#endif
  const int rc = select(nfds, for_write ? nullptr : &fds, for_write ? &fds : nullptr,
                        nullptr, &tv);
  return rc > 0;
}

bool SetNonBlocking(RerankSocket s, bool nonblock) {
#ifdef _WIN32
  u_long mode = nonblock ? 1 : 0;
  return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
  int flags = fcntl(s, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  if (nonblock) {
    flags |= O_NONBLOCK;
  } else {
    flags &= ~O_NONBLOCK;
  }
  return fcntl(s, F_SETFL, flags) == 0;
#endif
}

bool TcpExchange(const std::string& host, int port, const std::string& req_line,
                 int timeout_ms, std::string* resp) {
  EnsureWinsock();
  if (resp == nullptr || host.empty() || port <= 0 || timeout_ms <= 0) {
    return false;
  }
  RerankSocket s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == kInvalidRerankSocket) {
    return false;
  }
  const int nodelay = 1;
  setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay),
             sizeof(nodelay));
  if (!SetNonBlocking(s, true)) {
    CloseRerankSocket(s);
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    CloseRerankSocket(s);
    return false;
  }
  const int cr = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
#ifdef _WIN32
  const bool in_progress = (cr != 0) && (WSAGetLastError() == WSAEWOULDBLOCK);
#else
  const bool in_progress = (cr != 0) && (errno == EINPROGRESS);
#endif
  if (cr != 0 && !in_progress) {
    CloseRerankSocket(s);
    return false;
  }
  if (cr != 0 && !WaitSocket(s, true, timeout_ms)) {
    CloseRerankSocket(s);
    return false;
  }
  int so_error = 0;
#ifdef _WIN32
  int slen = sizeof(so_error);
#else
  socklen_t slen = sizeof(so_error);
#endif
  if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_error),
                 &slen) != 0 ||
      so_error != 0) {
    CloseRerankSocket(s);
    return false;
  }
  SetNonBlocking(s, false);
#ifdef _WIN32
  DWORD tv = static_cast<DWORD>(timeout_ms);
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv),
             sizeof(tv));
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv),
             sizeof(tv));
#else
  timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

  size_t off = 0;
  while (off < req_line.size()) {
    const int n = send(s, req_line.data() + off,
                       static_cast<int>(req_line.size() - off), 0);
    if (n <= 0) {
      CloseRerankSocket(s);
      return false;
    }
    off += static_cast<size_t>(n);
  }

  std::string buf;
  buf.reserve(1024);
  char tmp[1024];
  while (buf.find('\n') == std::string::npos) {
    const int n = recv(s, tmp, sizeof(tmp), 0);
    if (n <= 0) {
      CloseRerankSocket(s);
      return false;
    }
    buf.append(tmp, tmp + n);
    if (buf.size() > 1 << 20) {
      CloseRerankSocket(s);
      return false;
    }
  }
  CloseRerankSocket(s);
  *resp = buf.substr(0, buf.find('\n'));
  return true;
}

}  // namespace

RerankRewriter::RerankRewriter() { LoadConfigFromEnv(); }

RerankRewriter::~RerankRewriter() = default;

void RerankRewriter::LoadConfigFromEnv() {
  // v1.0 ships the loopback-only runtime in the same MSI, so reranking is on
  // by default.  An explicit false/0 value remains an administrator kill
  // switch.  If the daemon is unavailable, every request fails safe and Mozc
  // keeps its native candidate order.
  const std::string enabled = GetEnvOrEmpty("MOZC_RERANK_ENABLED");
  enabled_ = enabled.empty() ? true : EnvTruthy(enabled);
  hook_cmd_ = GetEnvOrEmpty("MOZC_RERANK_HOOK_CMD");
  daemon_addr_ = GetEnvOrEmpty("MOZC_RERANK_DAEMON_ADDR");
  if (daemon_addr_.empty()) {
    daemon_addr_ = "127.0.0.1:17890";
  }
  log_path_ = GetEnvOrEmpty("MOZC_RERANK_LOG");
  policy_path_ = GetEnvOrEmpty("MOZC_RERANK_POLICY");
  if (!policy_path_.empty()) {
    LoadPolicyFile(policy_path_);
  }

  const std::string tau_s = GetEnvOrEmpty("MOZC_RERANK_TAU");
  if (!tau_s.empty()) {
    double t = 0;
    if (absl::SimpleAtod(tau_s, &t)) {
      tau_ = static_cast<float>(t);
    }
  }
  const std::string cap_s = GetEnvOrEmpty("MOZC_RERANK_CAND_CAP");
  if (!cap_s.empty()) {
    int c = 0;
    if (absl::SimpleAtoi(cap_s, &c) && c > 0) {
      cand_cap_ = c;
    }
  }
  const std::string to_s = GetEnvOrEmpty("MOZC_RERANK_TIMEOUT_MS");
  if (!to_s.empty()) {
    int t = 0;
    if (absl::SimpleAtoi(to_s, &t) && t > 0) {
      timeout_ms_ = t;
    }
  }

  if (enabled_) {
    LOG(INFO) << "RerankRewriter enabled tau=" << tau_
              << " cand_cap=" << cand_cap_ << " timeout_ms=" << timeout_ms_
              << " daemon=" << daemon_addr_
              << " hook=" << (hook_cmd_.empty() ? "(off)" : hook_cmd_)
              << " log=" << (log_path_.empty() ? "(off)" : log_path_);
  }
}

void RerankRewriter::LoadPolicyFile(const std::string& path) {
  std::string json;
  if (!ReadFileToString(path, &json)) {
    LOG(WARNING) << "RerankRewriter policy unreadable: " << path;
    return;
  }
  auto num = [&](const char* key, double* out) {
    const std::string needle = absl::StrCat("\"", key, "\"");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
      return;
    }
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) {
      return;
    }
    ++pos;
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t')) {
      ++pos;
    }
    double v = 0;
    if (absl::SimpleAtod(json.substr(pos, 32), &v)) {
      *out = v;
    }
  };
  double tau = tau_, cap = cand_cap_, tmax = timeout_ms_, ml = max_len_,
         cc = context_chars_;
  num("tau", &tau);
  num("cand_cap", &cap);
  num("timeout_ms", &tmax);
  num("max_len", &ml);
  num("context_clip_max_chars", &cc);
  tau_ = static_cast<float>(tau);
  if (cap > 0) {
    cand_cap_ = static_cast<int>(cap);
  }
  if (tmax > 0) {
    timeout_ms_ = static_cast<int>(tmax);
  }
  if (ml > 0) {
    max_len_ = static_cast<int>(ml);
  }
  if (cc > 0) {
    context_chars_ = static_cast<int>(cc);
  }
}

void RerankRewriter::NoteTimeout() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  consecutive_ok_ = 0;
  ++consecutive_timeouts_;
  if (consecutive_timeouts_ >= 5 && degrade_tier_ < 4) {
    ++degrade_tier_;
    consecutive_timeouts_ = 0;
    LOG(WARNING) << "RerankRewriter degrade_tier=" << degrade_tier_;
  }
}

void RerankRewriter::NoteSuccess() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  consecutive_timeouts_ = 0;
  ++consecutive_ok_;
  if (consecutive_ok_ >= 5 && degrade_tier_ > 0) {
    --degrade_tier_;
    consecutive_ok_ = 0;
    LOG(INFO) << "RerankRewriter upgrade_tier=" << degrade_tier_;
  }
}

int RerankRewriter::EffectiveCandCap() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (degrade_tier_ >= 1) {
    return std::min(cand_cap_, 15);
  }
  return cand_cap_;
}

int RerankRewriter::EffectiveContextChars() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (degrade_tier_ >= 3) {
    return 0;
  }
  if (degrade_tier_ >= 2) {
    return 20;
  }
  return context_chars_;
}

bool RerankRewriter::DegradeDisabled() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  return degrade_tier_ >= 4;
}

int RerankRewriter::capability(const ConversionRequest& request) const {
  if (!enabled_) {
    return RewriterInterface::NOT_AVAILABLE;
  }
  if (request.request_type() == ConversionRequest::CONVERSION) {
    return RewriterInterface::CONVERSION;
  }
  return RewriterInterface::NOT_AVAILABLE;
}

bool RerankRewriter::Rewrite(const ConversionRequest& request,
                             Segments* segments) const {
  if (!enabled_ || segments == nullptr ||
      segments->conversion_segments_size() == 0) {
    return false;
  }
  if (DegradeDisabled()) {
    return false;
  }

  // Rerank the last conversion segment. 「駅にきしゃ」is two segments;
  // segment(0)-only left きしゃ on Mozc default (記者) with empty context.
  const int conv_n = static_cast<int>(segments->conversion_segments_size());
  const int target = conv_n - 1;
  Segment* segment = segments->mutable_conversion_segment(target);
  if (segment == nullptr || segment->candidates_size() == 0) {
    return false;
  }

  const std::string reading =
      rerank::NormalizeReading(std::string(segment->key()));
  if (reading.empty()) {
    return false;
  }

  const int cap = EffectiveCandCap();
  std::vector<std::string> nbest;
  const int n = std::min(cap, static_cast<int>(segment->candidates_size()));
  nbest.reserve(n);
  for (int i = 0; i < n; ++i) {
    nbest.push_back(std::string(segment->candidate(i).value));
  }

  // The application-provided surrounding text is authoritative and remains
  // available even when Mozc cannot reconstruct converter history.  Falling
  // back to Segments preserves headless and older-client behavior.
  std::string history = std::string(request.context().preceding_text());
  if (history.empty()) {
    for (size_t i = 0; i < segments->history_segments_size(); ++i) {
      const Segment& hs = segments->history_segment(i);
      if (hs.candidates_size() > 0) {
        history.append(hs.candidate(0).value);
      }
    }
  }
  for (int i = 0; i < target; ++i) {
    const Segment& prev = segments->conversion_segment(i);
    if (prev.candidates_size() > 0) {
      history.append(prev.candidate(0).value);
    }
  }
  const int ctx_n = EffectiveContextChars();
  const std::string context_prev =
      (ctx_n <= 0) ? std::string() : rerank::CleanContext(history, ctx_n);

  const std::string skip = rerank::RerankSkipReason(reading, context_prev);
  if (!skip.empty()) {
    AppendPrivacySafeDiag(
        "guard_skip", request.context().preceding_text().size(), history.size(),
        context_prev.size(), reading.size(), nbest.size(),
        segments->history_segments_size(), segments->conversion_segments_size());
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_log_.reading = reading;
    pending_log_.nbest = nbest;
    pending_log_.context_prev = context_prev;
    pending_log_.rerank_top1 = nbest.front();
    pending_log_.final_top1 = nbest.front();
    pending_log_.overwritten = false;
    pending_log_.tau = tau_;
    has_pending_log_ = true;
    return false;
  }

  AppendPrivacySafeDiag(
      "daemon_call", request.context().preceding_text().size(), history.size(),
      context_prev.size(), reading.size(), nbest.size(),
      segments->history_segments_size(), segments->conversion_segments_size());

  HookResult result;
  if (!CallHookWithTimeout(reading, nbest, context_prev, &result)) {
    LOG(WARNING) << "RerankRewriter hook failed/timeout for key=" << reading;
    return false;
  }
  if (result.ranked_surfaces.empty()) {
    return false;
  }
  if (result.overwritten && rerank::IsJunkSurface(result.final_top1)) {
    result.overwritten = false;
    result.final_top1 = nbest.front();
    result.ranked_surfaces = nbest;
  }

  const bool changed = ReorderSegment(segment, result.ranked_surfaces);

  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_log_.reading = reading;
    pending_log_.nbest = nbest;
    pending_log_.context_prev = context_prev;
    pending_log_.rerank_top1 = result.rerank_top1;
    pending_log_.final_top1 = result.final_top1;
    pending_log_.overwritten = result.overwritten;
    pending_log_.tau = tau_;
    has_pending_log_ = true;
  }

  return changed;
}

void RerankRewriter::Finish(const ConversionRequest& request,
                            const Segments& segments) {
  (void)request;
  if (!enabled_ || log_path_.empty()) {
    return;
  }

  PendingLog pending;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (!has_pending_log_) {
      return;
    }
    pending = pending_log_;
    has_pending_log_ = false;
  }

  std::string chosen = pending.final_top1;
  if (segments.conversion_segments_size() > 0) {
    const Segment& seg = segments.conversion_segment(0);
    if (seg.candidates_size() > 0) {
      chosen = std::string(seg.candidate(0).value);
    }
  }
  AppendConversionLog(pending, chosen);
}

void RerankRewriter::Clear() {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  has_pending_log_ = false;
  pending_log_ = PendingLog{};
}

bool RerankRewriter::CallHookWithTimeout(const std::string& reading,
                                         const std::vector<std::string>& nbest,
                                         const std::string& context_prev,
                                         HookResult* out) const {
  auto fut = std::async(std::launch::async, [this, reading, nbest, context_prev,
                                             out]() {
    if (!hook_cmd_.empty()) {
      return CallHook(reading, nbest, context_prev, out);
    }
    return CallDaemon(reading, nbest, context_prev, out);
  });
  if (fut.wait_for(std::chrono::milliseconds(timeout_ms_)) !=
      std::future_status::ready) {
    NoteTimeout();
    return false;
  }
  const bool ok = fut.get();
  if (ok) {
    NoteSuccess();
  } else {
    NoteTimeout();
  }
  return ok;
}

bool RerankRewriter::ParseDaemonAddr(const std::string& addr, std::string* host,
                                     int* port) {
  if (host == nullptr || port == nullptr || addr.empty()) {
    return false;
  }
  const size_t colon = addr.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= addr.size()) {
    return false;
  }
  *host = addr.substr(0, colon);
  int p = 0;
  if (!absl::SimpleAtoi(addr.substr(colon + 1), &p) || p <= 0 || p > 65535) {
    return false;
  }
  *port = p;
  return true;
}

bool RerankRewriter::CallDaemon(const std::string& reading,
                                const std::vector<std::string>& nbest,
                                const std::string& context_prev,
                                HookResult* out) const {
  if (out == nullptr) {
    return false;
  }
  std::string host;
  int port = 0;
  if (!ParseDaemonAddr(daemon_addr_, &host, &port)) {
    return false;
  }
  std::ostringstream ss;
  ss << "{\"reading\":\"" << EscapeJson(reading) << "\",\"context_prev\":\""
     << EscapeJson(context_prev) << "\",\"nbest\":[";
  for (size_t i = 0; i < nbest.size(); ++i) {
    if (i) {
      ss << ',';
    }
    ss << '"' << EscapeJson(nbest[i]) << '"';
  }
  ss << "]}\n";
  std::string resp;
  if (!TcpExchange(host, port, ss.str(), timeout_ms_, &resp)) {
    return false;
  }
  return ParseHookResponse(resp, out);
}

bool RerankRewriter::CallHook(const std::string& reading,
                              const std::vector<std::string>& nbest,
                              const std::string& context_prev,
                              HookResult* out) const {
  if (out == nullptr || hook_cmd_.empty()) {
    return false;
  }

  const std::string base = absl::StrCat("mozc_rerank_", MOZC_RERANK_GETPID());
  const std::string req_path = JoinPath(TempDir(), base + "_req.json");
  const std::string resp_path = JoinPath(TempDir(), base + "_resp.json");
  DeleteFileQuiet(resp_path);

  if (!WriteRequestJson(req_path, reading, nbest, context_prev)) {
    DeleteFileQuiet(req_path);
    return false;
  }

  // Bridge contract: <HOOK_CMD> <req.json> <resp.json>
  // Quote paths for Windows shells; POSIX system() also tolerates quotes.
  const std::string cmdline =
      absl::StrCat(hook_cmd_, " \"", req_path, "\" \"", resp_path, "\"");
  const int rc = std::system(cmdline.c_str());
  std::string resp;
  const bool ok_read = ReadFileToString(resp_path, &resp);
  DeleteFileQuiet(req_path);
  DeleteFileQuiet(resp_path);

  if (rc != 0) {
    LOG(WARNING) << "Rerank hook exit=" << rc << " cmd=" << hook_cmd_;
    return false;
  }
  if (!ok_read || resp.empty()) {
    LOG(WARNING) << "Rerank hook produced empty response";
    return false;
  }
  return ParseHookResponse(resp, out);
}

bool RerankRewriter::ReorderSegment(Segment* segment,
                                    const std::vector<std::string>& ranked) {
  if (segment == nullptr || ranked.empty()) {
    return false;
  }
  bool changed = false;
  const int limit =
      std::min(static_cast<int>(ranked.size()),
               static_cast<int>(segment->candidates_size()));
  for (int target = 0; target < limit; ++target) {
    int found = -1;
    for (int i = target; i < static_cast<int>(segment->candidates_size());
         ++i) {
      if (segment->candidate(i).value == ranked[target]) {
        found = i;
        break;
      }
    }
    if (found > target) {
      segment->move_candidate(found, target);
      changed = true;
    }
  }
  return changed;
}

std::string RerankRewriter::EscapeJson(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

bool RerankRewriter::WriteRequestJson(const std::string& path,
                                      const std::string& reading,
                                      const std::vector<std::string>& nbest,
                                      const std::string& context_prev) {
  std::ostringstream ss;
  ss << "{\"reading\":\"" << EscapeJson(reading) << "\",\"context_prev\":\""
     << EscapeJson(context_prev) << "\",\"nbest\":[";
  for (size_t i = 0; i < nbest.size(); ++i) {
    if (i) {
      ss << ',';
    }
    ss << '"' << EscapeJson(nbest[i]) << '"';
  }
  ss << "]}";
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out << ss.str();
  return static_cast<bool>(out);
}

bool RerankRewriter::ExtractJsonString(const std::string& json, const char* key,
                                       std::string* value) {
  const std::string needle = absl::StrCat("\"", key, "\"");
  size_t pos = json.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  while (pos < json.size() &&
         (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' ||
          json[pos] == '\t')) {
    ++pos;
  }
  if (pos >= json.size() || json[pos] != '"') {
    return false;
  }
  ++pos;
  std::string out;
  while (pos < json.size()) {
    char c = json[pos++];
    if (c == '\\') {
      if (pos >= json.size()) {
        break;
      }
      char e = json[pos++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out.push_back(e);
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        default:
          out.push_back(e);
          break;
      }
    } else if (c == '"') {
      *value = out;
      return true;
    } else {
      out.push_back(c);
    }
  }
  return false;
}

bool RerankRewriter::ExtractJsonBool(const std::string& json, const char* key,
                                     bool* value) {
  const std::string needle = absl::StrCat("\"", key, "\"");
  size_t pos = json.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  while (pos < json.size() &&
         (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' ||
          json[pos] == '\t')) {
    ++pos;
  }
  if (absl::StartsWith(absl::string_view(json).substr(pos), "true")) {
    *value = true;
    return true;
  }
  if (absl::StartsWith(absl::string_view(json).substr(pos), "false")) {
    *value = false;
    return true;
  }
  return false;
}

bool RerankRewriter::ExtractJsonStringArray(
    const std::string& json, const char* key,
    std::vector<std::string>* values) {
  const std::string needle = absl::StrCat("\"", key, "\"");
  size_t pos = json.find(needle);
  if (pos == std::string::npos) {
    return false;
  }
  pos = json.find('[', pos + needle.size());
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  values->clear();
  while (pos < json.size()) {
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' ||
            json[pos] == '\t' || json[pos] == ',')) {
      ++pos;
    }
    if (pos < json.size() && json[pos] == ']') {
      return true;
    }
    if (pos >= json.size() || json[pos] != '"') {
      return !values->empty();
    }
    ++pos;
    std::string item;
    while (pos < json.size()) {
      char c = json[pos++];
      if (c == '\\') {
        if (pos >= json.size()) {
          break;
        }
        item.push_back(json[pos++]);
      } else if (c == '"') {
        values->push_back(item);
        break;
      } else {
        item.push_back(c);
      }
    }
  }
  return !values->empty();
}

bool RerankRewriter::ParseHookResponse(const std::string& json,
                                       HookResult* out) {
  if (out == nullptr) {
    return false;
  }
  if (!ExtractJsonStringArray(json, "ranked_surfaces", &out->ranked_surfaces)) {
    return false;
  }
  ExtractJsonString(json, "rerank_top1", &out->rerank_top1);
  ExtractJsonString(json, "final_top1", &out->final_top1);
  ExtractJsonBool(json, "overwritten", &out->overwritten);
  if (out->final_top1.empty() && !out->ranked_surfaces.empty()) {
    out->final_top1 = out->ranked_surfaces.front();
  }
  return !out->ranked_surfaces.empty();
}

void RerankRewriter::AppendConversionLog(const PendingLog& pending,
                                         const std::string& chosen) const {
  if (log_path_.empty()) {
    return;
  }
  // Minimal ISO-ish UTC timestamp (second resolution).
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm);

  std::ostringstream ss;
  ss << "{\"ts\":\"" << ts << "\",\"reading\":\"" << EscapeJson(pending.reading)
     << "\",\"nbest\":[";
  for (size_t i = 0; i < pending.nbest.size(); ++i) {
    if (i) {
      ss << ',';
    }
    ss << '"' << EscapeJson(pending.nbest[i]) << '"';
  }
  ss << "],\"chosen\":\"" << EscapeJson(chosen) << "\",\"context_prev\":\""
     << EscapeJson(pending.context_prev) << "\",\"rerank_top1\":\""
     << EscapeJson(pending.rerank_top1) << "\",\"final_top1\":\""
     << EscapeJson(pending.final_top1)
     << "\",\"overwritten\":" << (pending.overwritten ? "true" : "false")
     << ",\"tau\":" << pending.tau << ",\"source\":\"ime_online\"}\n";

  std::ofstream out(log_path_, std::ios::app | std::ios::binary);
  if (!out) {
    LOG(WARNING) << "Failed to append conversion log: " << log_path_;
    return;
  }
  out << ss.str();
}

}  // namespace mozc
