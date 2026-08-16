// Standalone CLI: encode stdin (or --text) to token ids.
// g++ -std=c++17 -DMOZC_RERANK_STANDALONE -O2 hf_tokenizer.cc tokenize_cli.cc -o tokenize_cli

#include "hf_tokenizer.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string dir;
  std::string text;
  int max_len = 128;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tokenizer" && i + 1 < argc) {
      dir = argv[++i];
    } else if (a == "--text" && i + 1 < argc) {
      text = argv[++i];
    } else if (a == "--max-len" && i + 1 < argc) {
      max_len = std::stoi(argv[++i]);
    }
  }
  if (dir.empty()) {
    std::cerr << "usage: tokenize_cli --tokenizer DIR [--text STR] [--max-len N]\n";
    return 2;
  }
  if (text.empty()) {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!text.empty()) {
        text.push_back('\n');
      }
      text += line;
    }
  }
  mozc::rerank::HfWordPieceTokenizer tok;
  if (!tok.LoadFromTokenizerDir(dir)) {
    std::cerr << "failed to load tokenizer from " << dir << "\n";
    return 1;
  }
  const auto ids = tok.Encode(text, max_len);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      std::cout << ' ';
    }
    std::cout << ids[i];
  }
  std::cout << '\n';
  return 0;
}
