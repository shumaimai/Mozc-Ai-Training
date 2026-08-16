// Copyright 2026 AI Mozc IME Project
// BERT WordPiece encode matching HuggingFace BertTokenizer / ModernBERT-Ja.

#ifdef MOZC_RERANK_STANDALONE
#include "hf_tokenizer.h"
#else
#include "rewriter/hf_tokenizer.h"
#endif

#include <cctype>
#include <fstream>
#include <sstream>

namespace mozc {
namespace rerank {
namespace {

bool ReadAll(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

std::string Strip(const std::string& s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
    ++a;
  }
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' ||
                   s[b - 1] == '\n')) {
    --b;
  }
  return s.substr(a, b - a);
}

}  // namespace

std::vector<uint32_t> HfWordPieceTokenizer::Utf8ToCp(std::string_view s) {
  std::vector<uint32_t> out;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char* end = p + s.size();
  while (p < end) {
    uint32_t cp = 0;
    if (*p < 0x80) {
      cp = *p++;
    } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
      cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F);
      p += 2;
    } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
      cp = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      p += 3;
    } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
      cp = ((*p & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) |
           (p[3] & 0x3F);
      p += 4;
    } else {
      ++p;
      continue;
    }
    out.push_back(cp);
  }
  return out;
}

std::string HfWordPieceTokenizer::CpToUtf8(uint32_t cp) {
  std::string o;
  if (cp < 0x80) {
    o.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    o.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    o.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    o.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return o;
}

bool HfWordPieceTokenizer::IsCjk(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F) ||
         (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0x3040 && cp <= 0x309F) ||
         (cp >= 0x30A0 && cp <= 0x30FF) || (cp >= 0xFF00 && cp <= 0xFFEF) ||
         (cp >= 0x31F0 && cp <= 0x31FF);
}

bool HfWordPieceTokenizer::IsPunct(uint32_t cp) {
  if (cp == 0x21 || (cp >= 0x23 && cp <= 0x26) || (cp >= 0x28 && cp <= 0x2A) ||
      cp == 0x2C || cp == 0x2D || (cp >= 0x2E && cp <= 0x2F) ||
      (cp >= 0x3A && cp <= 0x3B) || (cp >= 0x3F && cp <= 0x40) ||
      (cp >= 0x5B && cp <= 0x5D) || cp == 0x5F || cp == 0x7B || cp == 0x7D) {
    return true;
  }
  // Unicode P* categories (approx via ranges used by HF BertBasicTokenizer)
  const uint32_t cat = cp;
  if ((cat >= 0x2000 && cat <= 0x206F) || (cat >= 0x3000 && cat <= 0x303F) ||
      (cat >= 0xFF01 && cat <= 0xFF0F) || (cat >= 0xFF1A && cat <= 0xFF20) ||
      (cat >= 0xFF3B && cat <= 0xFF40) || (cat >= 0xFF5B && cat <= 0xFF65)) {
    return true;
  }
  return false;
}

bool HfWordPieceTokenizer::IsWhitespace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000 ||
         cp == 0x00A0;
}

bool HfWordPieceTokenizer::LoadVocabTxt(const std::string& vocab_path) {
  std::ifstream in(vocab_path);
  if (!in) {
    return false;
  }
  token_to_id_.clear();
  std::string line;
  int32_t id = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    token_to_id_[line] = id++;
  }
  auto lookup = [&](const char* tok, int32_t fallback) {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? fallback : it->second;
  };
  pad_id_ = lookup("[PAD]", 0);
  unk_id_ = lookup("[UNK]", 1);
  cls_id_ = lookup("[CLS]", 2);
  sep_id_ = lookup("[SEP]", 3);
  return !token_to_id_.empty();
}

bool HfWordPieceTokenizer::LoadFromTokenizerDir(const std::string& dir) {
  const std::string vocab = dir + "/vocab.txt";
  if (LoadVocabTxt(vocab)) {
    return true;
  }
  // Fallback: scrape "vocab" object from tokenizer.json (token -> id).
  std::string json;
  if (!ReadAll(dir + "/tokenizer.json", &json)) {
    return false;
  }
  const std::string key = "\"vocab\"";
  const auto pos = json.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  auto brace = json.find('{', pos);
  if (brace == std::string::npos) {
    return false;
  }
  int depth = 0;
  size_t end = brace;
  for (; end < json.size(); ++end) {
    if (json[end] == '{') {
      ++depth;
    } else if (json[end] == '}') {
      --depth;
      if (depth == 0) {
        ++end;
        break;
      }
    }
  }
  const std::string obj = json.substr(brace, end - brace);
  token_to_id_.clear();
  size_t i = 1;
  while (i < obj.size()) {
    auto q1 = obj.find('"', i);
    if (q1 == std::string::npos) {
      break;
    }
    auto q2 = obj.find('"', q1 + 1);
    if (q2 == std::string::npos) {
      break;
    }
    std::string tok = obj.substr(q1 + 1, q2 - q1 - 1);
    auto colon = obj.find(':', q2);
    if (colon == std::string::npos) {
      break;
    }
    size_t n0 = colon + 1;
    while (n0 < obj.size() && (obj[n0] == ' ' || obj[n0] == '\n')) {
      ++n0;
    }
    size_t n1 = n0;
    while (n1 < obj.size() && (std::isdigit(static_cast<unsigned char>(obj[n1])) ||
                               obj[n1] == '-')) {
      ++n1;
    }
    int32_t id = std::stoi(obj.substr(n0, n1 - n0));
    token_to_id_[tok] = id;
    i = n1;
  }
  auto lookup = [&](const char* tok, int32_t fallback) {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? fallback : it->second;
  };
  pad_id_ = lookup("[PAD]", 0);
  unk_id_ = lookup("[UNK]", 1);
  cls_id_ = lookup("[CLS]", 2);
  sep_id_ = lookup("[SEP]", 3);
  return !token_to_id_.empty();
}

std::vector<std::string> HfWordPieceTokenizer::BasicTokenize(
    std::string_view text) const {
  // BertBasicTokenizer: whitespace split, isolate CJK + punctuation.
  std::string spaced;
  for (uint32_t cp : Utf8ToCp(text)) {
    if (IsCjk(cp) || IsPunct(cp)) {
      spaced.push_back(' ');
      spaced += CpToUtf8(cp);
      spaced.push_back(' ');
    } else if (IsWhitespace(cp)) {
      spaced.push_back(' ');
    } else {
      spaced += CpToUtf8(cp);
    }
  }
  std::vector<std::string> toks;
  std::string cur;
  for (char c : spaced) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      if (!cur.empty()) {
        toks.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) {
    toks.push_back(cur);
  }
  return toks;
}

std::vector<std::string> HfWordPieceTokenizer::WordPiece(
    const std::string& token) const {
  const auto cps = Utf8ToCp(token);
  if (static_cast<int>(cps.size()) > max_input_chars_per_word_) {
    return {"[UNK]"};
  }
  std::vector<std::string> sub;
  int start = 0;
  const int n = static_cast<int>(cps.size());
  while (start < n) {
    int end = n;
    std::string cur;
    bool found = false;
    while (start < end) {
      std::string substr;
      for (int i = start; i < end; ++i) {
        substr += CpToUtf8(cps[i]);
      }
      if (start > 0) {
        substr = "##" + substr;
      }
      if (token_to_id_.find(substr) != token_to_id_.end()) {
        cur = substr;
        found = true;
        break;
      }
      --end;
    }
    if (!found) {
      return {"[UNK]"};
    }
    sub.push_back(cur);
    start = end;
  }
  return sub;
}

std::vector<int32_t> HfWordPieceTokenizer::Encode(std::string_view text,
                                                  int max_len) const {
  std::vector<int32_t> ids;
  ids.push_back(cls_id_);
  for (const auto& tok : BasicTokenize(text)) {
    for (const auto& wp : WordPiece(tok)) {
      auto it = token_to_id_.find(wp);
      ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
    }
  }
  ids.push_back(sep_id_);
  if (max_len > 0 && static_cast<int>(ids.size()) > max_len) {
    ids.resize(static_cast<size_t>(max_len - 1));
    ids.push_back(sep_id_);
  }
  return ids;
}

}  // namespace rerank
}  // namespace mozc
