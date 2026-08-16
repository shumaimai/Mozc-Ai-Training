// Copyright 2026 AI Mozc IME Project
// Port of tools/rerank/margin.py apply_margin (top-1 gate only).

#ifndef MOZC_RERANK_MARGIN_H_
#define MOZC_RERANK_MARGIN_H_

#include <algorithm>
#include <string>
#include <vector>

namespace mozc {
namespace rerank {

struct MarginDecision {
  std::string final_top1;
  std::string rerank_top1;
  std::string mozc_top1;
  float score_rerank = 0.0f;
  float score_mozc = 0.0f;
  float margin = 0.0f;
  bool overwritten = false;
  bool mozc_missing = false;
};

inline int ArgMax(const std::vector<float>& scores) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(scores.size()); ++i) {
    if (scores[i] > scores[best]) {
      best = i;
    }
  }
  return best;
}

inline MarginDecision ApplyMargin(const std::vector<std::string>& candidates,
                                  const std::vector<float>& scores,
                                  const std::string& mozc_top1, float tau) {
  MarginDecision d;
  d.mozc_top1 = mozc_top1;
  if (candidates.empty() || candidates.size() != scores.size()) {
    d.final_top1 = mozc_top1;
    d.mozc_missing = true;
    return d;
  }
  const int best = ArgMax(scores);
  d.rerank_top1 = candidates[best];
  d.score_rerank = scores[best];
  int mozc_i = -1;
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    if (candidates[i] == mozc_top1) {
      mozc_i = i;
      break;
    }
  }
  if (mozc_i < 0) {
    d.final_top1 = mozc_top1;
    d.mozc_missing = true;
    return d;
  }
  d.score_mozc = scores[mozc_i];
  d.margin = d.score_rerank - d.score_mozc;
  d.overwritten = (d.rerank_top1 != mozc_top1) && (d.margin >= tau);
  d.final_top1 = d.overwritten ? d.rerank_top1 : mozc_top1;
  return d;
}

// Rank by score desc; put final_top1 first (same as phase3_hook.rerank_one).
inline std::vector<std::string> RankedSurfaces(
    const std::vector<std::string>& candidates, const std::vector<float>& scores,
    const std::string& final_top1) {
  std::vector<int> order(candidates.size());
  for (int i = 0; i < static_cast<int>(order.size()); ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });
  std::vector<std::string> out;
  out.reserve(candidates.size());
  out.push_back(final_top1);
  for (int i : order) {
    if (candidates[i] != final_top1) {
      out.push_back(candidates[i]);
    }
  }
  return out;
}

}  // namespace rerank
}  // namespace mozc

#endif  // MOZC_RERANK_MARGIN_H_
