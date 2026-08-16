// Copyright 2026 AI Mozc IME Project
// Port of tools/rerank/context_clip.py (NEXT_TASK_PHASE3_CTX).
// Python is the source of truth. Keep this file Mozc-free for standalone parity.

#ifndef MOZC_RERANK_CONTEXT_CLIP_H_
#define MOZC_RERANK_CONTEXT_CLIP_H_

#include <cstddef>
#include <string>
#include <string_view>

namespace mozc {
namespace rerank {

// Same as Python clean_context: strip markup/newlines, cut at last 。！？!?,
// keep in-progress sentence, last max_chars code points. No NFKC.
std::string CleanContext(std::string_view text, int max_chars = 50);

// Left context of full_text[:token_char_start] then CleanContext.
std::string ClipContextPrev(std::string_view full_text, int token_char_start,
                            int max_chars = 50);

// Hiragana + a practical NFKC subset (fullwidth ASCII). Not used on context.
std::string NormalizeReading(std::string_view text);

}  // namespace rerank
}  // namespace mozc

#endif  // MOZC_RERANK_CONTEXT_CLIP_H_
