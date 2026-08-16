// Copyright 2026 AI Mozc IME Project
// Procedural port of tools/rerank/usage_guard.py (Unicode code-point length).

#ifdef MOZC_RERANK_STANDALONE
#include "rerank_guard.h"
#include "rerank_eligible_readings.inc"
#else
#include "rewriter/rerank_guard.h"
#include "rewriter/rerank_eligible_readings.inc"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace mozc {
namespace rerank {
namespace {

bool IsUtf8Cont(unsigned char c) { return (c & 0xC0) == 0x80; }

bool NextCodepoint(std::string_view s, size_t* i, char32_t* cp) {
  if (i == nullptr || cp == nullptr || *i >= s.size()) {
    return false;
  }
  const unsigned char* p =
      reinterpret_cast<const unsigned char*>(s.data() + *i);
  const unsigned char* end =
      reinterpret_cast<const unsigned char*>(s.data() + s.size());
  unsigned char c = *p;
  int n = 0;
  if (c < 0x80) {
    *cp = c;
    n = 1;
  } else if ((c & 0xE0) == 0xC0 && p + 1 < end && IsUtf8Cont(p[1])) {
    *cp = ((c & 0x1F) << 6) | (p[1] & 0x3F);
    n = 2;
  } else if ((c & 0xF0) == 0xE0 && p + 2 < end && IsUtf8Cont(p[1]) &&
             IsUtf8Cont(p[2])) {
    *cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    n = 3;
  } else if ((c & 0xF8) == 0xF0 && p + 3 < end && IsUtf8Cont(p[1]) &&
             IsUtf8Cont(p[2]) && IsUtf8Cont(p[3])) {
    *cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) |
          (p[3] & 0x3F);
    n = 4;
  } else {
    *i += 1;
    *cp = 0xFFFD;
    return true;
  }
  *i += static_cast<size_t>(n);
  return true;
}

bool IsHiragana(char32_t c) { return c >= 0x3040 && c <= 0x309F; }
bool IsKatakana(char32_t c) { return c >= 0x30A0 && c <= 0x30FF; }
bool IsKanji(char32_t c) {
  return (c >= 0x3400 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF);
}
bool IsHwKanaLetter(char32_t c) { return c >= 0xFF66 && c <= 0xFF9D; }
bool IsHwKanaBlock(char32_t c) { return c >= 0xFF61 && c <= 0xFF9F; }
bool IsLatinLetter(char32_t c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= 0x00C0 && c <= 0x024F) || (c >= 0xFF21 && c <= 0xFF3A) ||
         (c >= 0xFF41 && c <= 0xFF5A);
}

bool EnvTruthy(const char* name, bool default_on) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') {
    return default_on;
  }
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
      std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "no") == 0 ||
      std::strcmp(v, "NO") == 0 || std::strcmp(v, "off") == 0 ||
      std::strcmp(v, "OFF") == 0) {
    return false;
  }
  return true;
}

bool IsKyujitaiChar(char32_t c) {
  // Must match tools/rerank/usage_guard.py _KYUJITAI_CHARS.
  switch (c) {
    case 0x5BE6:  // 實
    case 0x820A:  // 舊
    case 0x8B83:  // 讃
    case 0x8207:  // 與
    case 0x5B78:  // 學
    case 0x9AD4:  // 體
    case 0x5EE3:  // 廣
    case 0x61C9:  // 應
    case 0x85DD:  // 藝
    case 0x7E23:  // 縣
    case 0x64E7:  // 擧
    case 0x7027:  // 瀧
    case 0x975C:  // 靜
      return true;
    default:
      return false;
  }
}

}  // namespace

int Utf8CodepointLength(std::string_view s) {
  int n = 0;
  size_t i = 0;
  char32_t cp = 0;
  while (NextCodepoint(s, &i, &cp)) {
    ++n;
  }
  return n;
}

bool HasLinguisticContent(std::string_view s) {
  size_t i = 0;
  char32_t c = 0;
  while (NextCodepoint(s, &i, &c)) {
    if (IsHiragana(c) || IsKatakana(c) || IsKanji(c) || IsHwKanaLetter(c) ||
        (c >= 0x31F0 && c <= 0x31FF) || IsLatinLetter(c)) {
      return true;
    }
  }
  return false;
}

bool ContextEmptyOrSymbol(std::string_view context_prev) {
  if (context_prev.empty()) {
    return true;
  }
  bool any_non_space = false;
  size_t i = 0;
  char32_t c = 0;
  while (NextCodepoint(context_prev, &i, &c)) {
    if (c != U' ' && c != U'\t' && c != U'\u3000' && c != U'\n' &&
        c != U'\r') {
      any_non_space = true;
      break;
    }
  }
  if (!any_non_space) {
    return true;
  }
  return !HasLinguisticContent(context_prev);
}

bool IsJunkSurface(std::string_view surface) {
  if (surface.empty()) {
    return true;
  }
  int kana = 0;
  int other = 0;
  bool all_hw = true;
  size_t i = 0;
  char32_t c = 0;
  while (NextCodepoint(surface, &i, &c)) {
    if (IsKyujitaiChar(c)) {
      return true;
    }
    if (!IsHwKanaBlock(c)) {
      all_hw = false;
    }
    if (c == 0x30FC || IsKatakana(c) || IsHwKanaLetter(c)) {
      ++kana;
    } else if (c == U' ' || c == U'\t' || c == U'\u3000') {
      continue;
    } else {
      ++other;
    }
  }
  if (all_hw) {
    return true;
  }
  return kana > 0 && other == 0;
}

bool IsEligibleReading(std::string_view reading) {
  const std::string key(reading);
  const auto* begin = kEligibleReadings;
  const auto* end = kEligibleReadings + kEligibleReadingsSize;
  const auto* it = std::lower_bound(
      begin, end, key.c_str(),
      [](const char* a, const char* b) { return std::strcmp(a, b) < 0; });
  return it != end && std::strcmp(*it, key.c_str()) == 0;
}

bool GuardsEnabled() { return EnvTruthy("MOZC_RERANK_GUARD", true); }

bool StrictEligibleGuardEnabled() {
  const char* value = std::getenv("MOZC_RERANK_GUARD_MODE");
  // Existing installations remain strict.  The personalized model explicitly
  // opts into safety mode, which keeps short/context/junk guards but removes
  // the coarse static reading allowlist.
  return value == nullptr || std::strcmp(value, "safety") != 0;
}

std::string RerankSkipReason(std::string_view reading,
                             std::string_view context_prev) {
  if (!GuardsEnabled()) {
    return std::string();
  }
  if (reading.empty() || Utf8CodepointLength(reading) <= 2) {
    return "reading_too_short";
  }
  if (ContextEmptyOrSymbol(context_prev)) {
    return "context_empty_or_symbol";
  }
  if (StrictEligibleGuardEnabled() && !IsEligibleReading(reading)) {
    return "reading_not_eligible";
  }
  return std::string();
}

}  // namespace rerank
}  // namespace mozc
