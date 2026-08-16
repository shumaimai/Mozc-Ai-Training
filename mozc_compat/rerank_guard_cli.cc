// Standalone CLI for Python↔C++ usage_guard parity.
// Build: g++ -std=c++17 -DMOZC_RERANK_STANDALONE -O2 \
//   rerank_guard.cc rerank_guard_cli.cc -o rerank_guard_cli

#include "rerank_guard.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string op = "skip";
  std::string reading;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--op" && i + 1 < argc) {
      op = argv[++i];
    } else if (a == "--reading" && i + 1 < argc) {
      reading = argv[++i];
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
  if (op == "skip") {
    std::cout << mozc::rerank::RerankSkipReason(reading, input);
  } else if (op == "eligible") {
    std::cout << (mozc::rerank::IsEligibleReading(input.empty() ? reading : input)
                      ? "1"
                      : "0");
  } else if (op == "junk") {
    std::cout << (mozc::rerank::IsJunkSurface(input) ? "1" : "0");
  } else if (op == "ctx") {
    std::cout << (mozc::rerank::ContextEmptyOrSymbol(input) ? "1" : "0");
  } else {
    std::cerr << "unknown op\n";
    return 2;
  }
  return 0;
}
