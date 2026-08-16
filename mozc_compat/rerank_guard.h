// Copyright 2026 AI Mozc IME Project
// Port of tools/rerank/usage_guard.py (NEXT_TASK_USAGE_GUARD).
// Python is the source of truth. Keep this file Mozc-free for standalone parity.

#ifndef MOZC_RERANK_GUARD_H_
#define MOZC_RERANK_GUARD_H_

#include <string>
#include <string_view>

namespace mozc {
namespace rerank {

// Empty = call the model. Otherwise a reason code matching Python usage_guard.
std::string RerankSkipReason(std::string_view reading,
                             std::string_view context_prev);

bool GuardsEnabled();
bool StrictEligibleGuardEnabled();
bool IsEligibleReading(std::string_view reading);
bool ContextEmptyOrSymbol(std::string_view context_prev);
bool IsJunkSurface(std::string_view surface);
int Utf8CodepointLength(std::string_view s);
bool HasLinguisticContent(std::string_view s);

}  // namespace rerank
}  // namespace mozc

#endif  // MOZC_RERANK_GUARD_H_
