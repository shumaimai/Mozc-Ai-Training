// Standalone CLI for Python↔C++ context_clip parity.
// Build: g++ -std=c++17 -DMOZC_RERANK_STANDALONE -O2 \
//   context_clip.cc context_clip_cli.cc -o context_clip_cli

#include "context_clip.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string op = "clean";
  int max_chars = 50;
  int start = -1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--op" && i + 1 < argc) {
      op = argv[++i];
    } else if (a == "--max" && i + 1 < argc) {
      max_chars = std::stoi(argv[++i]);
    } else if (a == "--start" && i + 1 < argc) {
      start = std::stoi(argv[++i]);
    }
  }
  std::string input;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!input.empty()) {
      input.push_back('\n');
    }
    input += line;
  }
  std::string out;
  if (op == "clean") {
    out = mozc::rerank::CleanContext(input, max_chars);
  } else if (op == "clip") {
    out = mozc::rerank::ClipContextPrev(input, start, max_chars);
  } else if (op == "reading") {
    out = mozc::rerank::NormalizeReading(input);
  } else {
    std::cerr << "unknown op\n";
    return 2;
  }
  std::cout << out;
  return 0;
}
