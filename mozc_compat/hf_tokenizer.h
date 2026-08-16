// Copyright 2026 AI Mozc IME Project
// BERT WordPiece loader for tokenizer.json / vocab.txt (ModernBERT-Ja).
// Goal: same string → same token ids as HuggingFace AutoTokenizer.

#ifndef MOZC_RERANK_HF_TOKENIZER_H_
#define MOZC_RERANK_HF_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mozc {
namespace rerank {

class HfWordPieceTokenizer {
 public:
  bool LoadFromTokenizerDir(const std::string& dir);
  bool LoadVocabTxt(const std::string& vocab_path);

  std::vector<int32_t> Encode(std::string_view text, int max_len) const;

  int32_t cls_id() const { return cls_id_; }
  int32_t sep_id() const { return sep_id_; }
  int32_t unk_id() const { return unk_id_; }
  int32_t pad_id() const { return pad_id_; }
  size_t vocab_size() const { return token_to_id_.size(); }

 private:
  std::vector<std::string> BasicTokenize(std::string_view text) const;
  std::vector<std::string> WordPiece(const std::string& token) const;
  static bool IsCjk(uint32_t cp);
  static bool IsPunct(uint32_t cp);
  static bool IsWhitespace(uint32_t cp);
  static std::vector<uint32_t> Utf8ToCp(std::string_view s);
  static std::string CpToUtf8(uint32_t cp);

  std::unordered_map<std::string, int32_t> token_to_id_;
  int32_t cls_id_ = 2;
  int32_t sep_id_ = 3;
  int32_t unk_id_ = 1;
  int32_t pad_id_ = 0;
  int max_input_chars_per_word_ = 100;
};

}  // namespace rerank
}  // namespace mozc

#endif  // MOZC_RERANK_HF_TOKENIZER_H_
