// Copyright 2026 AI Mozc IME Project
// Procedural port of tools/rerank/context_clip.py (Unicode code-point length).

#ifdef MOZC_RERANK_STANDALONE
#include "context_clip.h"
#else
#include "rewriter/context_clip.h"
#endif

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mozc {
namespace rerank {
namespace {

using U32 = std::u32string;

bool IsUtf8Cont(unsigned char c) { return (c & 0xC0) == 0x80; }

U32 Utf8ToU32(std::string_view s) {
  U32 out;
  out.reserve(s.size());
  const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char* end = p + s.size();
  while (p < end) {
    unsigned char c = *p;
    char32_t cp = 0;
    int n = 0;
    if (c < 0x80) {
      cp = c;
      n = 1;
    } else if ((c & 0xE0) == 0xC0 && p + 1 < end) {
      cp = ((c & 0x1F) << 6) | (p[1] & 0x3F);
      n = 2;
    } else if ((c & 0xF0) == 0xE0 && p + 2 < end) {
      cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      n = 3;
    } else if ((c & 0xF8) == 0xF0 && p + 3 < end) {
      cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) |
           (p[3] & 0x3F);
      n = 4;
    } else {
      ++p;
      continue;
    }
    p += n;
    out.push_back(cp);
  }
  return out;
}

std::string U32ToUtf8(const U32& s) {
  std::string out;
  out.reserve(s.size() * 3);
  for (char32_t cp : s) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return out;
}

bool IsSpaceLike(char32_t c) {
  return c == U' ' || c == U'\t' || c == U'\u3000';
}

bool IsSentEnd(char32_t c) {
  return c == U'。' || c == U'！' || c == U'？' || c == U'!' || c == U'?';
}

bool Iseq(char32_t c) { return c == U'='; }

void ReplaceAll(U32* s, const U32& from, const U32& to) {
  if (from.empty()) {
    return;
  }
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if (i + from.size() <= s->size() && s->compare(i, from.size(), from) == 0) {
      out.append(to);
      i += from.size();
    } else {
      out.push_back((*s)[i]);
      ++i;
    }
  }
  *s = std::move(out);
}

bool EqualsIgnoreCaseAscii(const U32& s, size_t i, const char* lit) {
  size_t n = 0;
  while (lit[n]) {
    ++n;
  }
  if (i + n > s.size()) {
    return false;
  }
  for (size_t k = 0; k < n; ++k) {
    char32_t c = s[i + k];
    char32_t e = static_cast<unsigned char>(lit[k]);
    if (c >= 'A' && c <= 'Z') {
      c = c - 'A' + 'a';
    }
    if (e >= 'A' && e <= 'Z') {
      e = e - 'A' + 'a';
    }
    if (c != e) {
      return false;
    }
  }
  return true;
}

// \[\s*(?:edit|編集)\s*\]
void StripWikiEdit(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if ((*s)[i] != U'[') {
      out.push_back((*s)[i++]);
      continue;
    }
    size_t j = i + 1;
    while (j < s->size() && IsSpaceLike((*s)[j])) {
      ++j;
    }
    bool match = false;
    size_t after = j;
    if (EqualsIgnoreCaseAscii(*s, j, "edit")) {
      after = j + 4;
      match = true;
    } else if (j + 1 < s->size() && (*s)[j] == U'編' && (*s)[j + 1] == U'集') {
      after = j + 2;
      match = true;
    }
    if (match) {
      while (after < s->size() && IsSpaceLike((*s)[after])) {
        ++after;
      }
      if (after < s->size() && (*s)[after] == U']') {
        i = after + 1;
        continue;
      }
    }
    out.push_back((*s)[i++]);
  }
  *s = std::move(out);
}

// ={2,}[^=\n]*={2,}
void StripWikiHeading(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if (!Iseq((*s)[i])) {
      out.push_back((*s)[i++]);
      continue;
    }
    size_t a = i;
    while (a < s->size() && Iseq((*s)[a])) {
      ++a;
    }
    if (a - i < 2) {
      out.push_back((*s)[i++]);
      continue;
    }
    size_t b = a;
    while (b < s->size() && (*s)[b] != U'=' && (*s)[b] != U'\n') {
      ++b;
    }
    size_t c = b;
    while (c < s->size() && Iseq((*s)[c])) {
      ++c;
    }
    if (c - b >= 2) {
      out.push_back(U' ');
      i = c;
      continue;
    }
    out.push_back((*s)[i++]);
  }
  *s = std::move(out);
}

// \[\[(?:[^|\]]+\|)?([^\]]+)\]\]  -> group 1
void StripWikiLink(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if (i + 1 < s->size() && (*s)[i] == U'[' && (*s)[i + 1] == U'[') {
      size_t j = i + 2;
      size_t pipe = static_cast<size_t>(-1);
      while (j < s->size() && (*s)[j] != U']') {
        if ((*s)[j] == U'|' && pipe == static_cast<size_t>(-1)) {
          pipe = j;
        }
        ++j;
      }
      if (j + 1 < s->size() && (*s)[j] == U']' && (*s)[j + 1] == U']') {
        size_t start = (pipe == static_cast<size_t>(-1)) ? i + 2 : pipe + 1;
        out.append(*s, start, j - start);
        i = j + 2;
        continue;
      }
    }
    out.push_back((*s)[i++]);
  }
  *s = std::move(out);
}

// \[\d+\]  and  <ref\b[^>]*>.*?</ref>
void StripWikiRef(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if ((*s)[i] == U'[') {
      size_t j = i + 1;
      if (j < s->size() && (*s)[j] >= U'0' && (*s)[j] <= U'9') {
        while (j < s->size() && (*s)[j] >= U'0' && (*s)[j] <= U'9') {
          ++j;
        }
        if (j < s->size() && (*s)[j] == U']') {
          i = j + 1;
          continue;
        }
      }
    }
    // <ref ...> ... </ref>
    if ((*s)[i] == U'<' && EqualsIgnoreCaseAscii(*s, i + 1, "ref")) {
      size_t j = i + 4;
      if (j < s->size() &&
          ((*s)[j] == U'>' || IsSpaceLike((*s)[j]) || (*s)[j] == U'/')) {
        while (j < s->size() && (*s)[j] != U'>') {
          ++j;
        }
        if (j < s->size() && (*s)[j] == U'>') {
          ++j;
          // find </ref>
          while (j + 5 < s->size()) {
            if ((*s)[j] == U'<' && (*s)[j + 1] == U'/' &&
                EqualsIgnoreCaseAscii(*s, j + 2, "ref") &&
                j + 5 < s->size() && (*s)[j + 5] == U'>') {
              i = j + 6;
              goto next;
            }
            ++j;
          }
        }
      }
    }
    out.push_back((*s)[i++]);
  next:;
  }
  *s = std::move(out);
}

// ={2,}
void StripEqRun(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    if (Iseq((*s)[i])) {
      size_t j = i;
      while (j < s->size() && Iseq((*s)[j])) {
        ++j;
      }
      if (j - i >= 2) {
        out.push_back(U' ');
        i = j;
        continue;
      }
    }
    out.push_back((*s)[i++]);
  }
  *s = std::move(out);
}

bool IsBulletChar(char32_t c) {
  return c == U'*' || c == U'＊' || c == U'#' || c == U'＃' || c == U'・';
}

// (?:^|\s)[*＊#＃・]+(?=\s|$)
void StripBullets(U32* s) {
  U32 out;
  out.reserve(s->size());
  size_t i = 0;
  while (i < s->size()) {
    bool at_bound = (i == 0) || IsSpaceLike((*s)[i - 1]);
    if (at_bound && IsBulletChar((*s)[i])) {
      size_t j = i;
      while (j < s->size() && IsBulletChar((*s)[j])) {
        ++j;
      }
      bool after_ok = (j == s->size()) || IsSpaceLike((*s)[j]);
      if (after_ok && j > i) {
        if (i == 0) {
          out.push_back(U' ');
        } else {
          // keep the preceding space already written; add replacement space
          out.push_back(U' ');
        }
        i = j;
        continue;
      }
    }
    out.push_back((*s)[i++]);
  }
  *s = std::move(out);
}

void CollapseSpaces(U32* s) {
  U32 out;
  out.reserve(s->size());
  bool in_space = false;
  for (char32_t c : *s) {
    if (IsSpaceLike(c)) {
      if (!in_space) {
        out.push_back(U' ');
        in_space = true;
      }
    } else {
      out.push_back(c);
      in_space = false;
    }
  }
  // strip ' ' and ideographic space (already collapsed to ' ')
  size_t a = 0;
  size_t b = out.size();
  while (a < b && out[a] == U' ') {
    ++a;
  }
  while (b > a && out[b - 1] == U' ') {
    --b;
  }
  *s = out.substr(a, b - a);
}

}  // namespace

std::string CleanContext(std::string_view text, int max_chars) {
  if (text.empty()) {
    return "";
  }
  U32 s = Utf8ToU32(text);
  // \r\n / \r / \n / \t -> space
  for (char32_t& c : s) {
    if (c == U'\r' || c == U'\n' || c == U'\t') {
      c = U' ';
    }
  }
  StripWikiEdit(&s);
  StripWikiHeading(&s);
  StripWikiLink(&s);
  StripWikiRef(&s);
  StripEqRun(&s);
  StripBullets(&s);
  CollapseSpaces(&s);
  if (s.empty()) {
    return "";
  }
  int last = -1;
  for (size_t i = 0; i < s.size(); ++i) {
    if (IsSentEnd(s[i])) {
      last = static_cast<int>(i + 1);
    }
  }
  U32 sentence = (last >= 0) ? s.substr(static_cast<size_t>(last)) : s;
  size_t ls = 0;
  while (ls < sentence.size() &&
         (sentence[ls] == U' ' || sentence[ls] == U'\u3000')) {
    ++ls;
  }
  sentence = sentence.substr(ls);
  if (sentence.empty()) {
    return "";
  }
  if (static_cast<int>(sentence.size()) > max_chars && max_chars > 0) {
    sentence = sentence.substr(sentence.size() - static_cast<size_t>(max_chars));
  }
  return U32ToUtf8(sentence);
}

std::string ClipContextPrev(std::string_view full_text, int token_char_start,
                            int max_chars) {
  if (token_char_start <= 0 || full_text.empty()) {
    return "";
  }
  U32 s = Utf8ToU32(full_text);
  if (token_char_start > static_cast<int>(s.size())) {
    token_char_start = static_cast<int>(s.size());
  }
  U32 left = s.substr(0, static_cast<size_t>(token_char_start));
  return CleanContext(U32ToUtf8(left), max_chars);
}

std::string NormalizeReading(std::string_view text) {
  if (text.empty()) {
    return "";
  }
  U32 s = Utf8ToU32(text);
  U32 out;
  out.reserve(s.size());
  for (char32_t c : s) {
    // NFKC-ish: fullwidth ASCII FF01-FF5E -> 21-7E
    if (c >= 0xFF01 && c <= 0xFF5E) {
      c = c - 0xFEE0;
    } else if (c == 0x3000) {
      c = U' ';
    }
    // katakana -> hiragana
    if (c >= 0x30A1 && c <= 0x30F6) {
      c = c - 0x60;
    }
    out.push_back(c);
  }
  return U32ToUtf8(out);
}

}  // namespace rerank
}  // namespace mozc
