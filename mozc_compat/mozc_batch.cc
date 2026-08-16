// Copyright 2024 AI Mozc IME Project
// mozc_batch - batch converter that emits Mozc's top-N candidates per reading.
//
// This binary is built INSIDE a google/mozc checkout (see integrate_mozc.py),
// not in this standalone repo. Prefer a vanilla Mozc engine without AIRewriter
// so baseline N-best is not polluted by network AI calls.
//
// I/O is deliberately plain TSV so the binary needs no JSON dependency. The
// JSONL glue (extracting readings, joining candidates back onto records) lives
// on the Python side in tools/dataset/mozc_batch.py.
//
//   Input : one reading (hiragana key) per line, read from --input or stdin.
//   Output: "<key>\t<cand1>\t<cand2>\t..." per line, to --output or stdout.
//
// Candidate construction (Phase0 fix):
//   1) Best-path concat of top-1 across all conversion segments
//   2) Per-segment alternates with other segments fixed to top-1
//   3) Force whole-key into a single segment via ResizeSegment, then take
//      that segment's candidates (so long compounds can appear as full strings)

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_stream.h"
#include "base/init_mozc.h"
#include "composer/composer.h"
#include "config/config_handler.h"
#include "converter/candidate.h"
#include "converter/converter_interface.h"
#include "converter/segments.h"
#include "data_manager/data_manager.h"
#include "engine/engine.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"

ABSL_FLAG(std::string, engine_data_path, "",
          "Path to the Mozc engine data file (e.g. the built mozc.data).");
ABSL_FLAG(std::string, input, "", "Input file with one reading per line; stdin if empty.");
ABSL_FLAG(std::string, output, "", "Output TSV file; stdout if empty.");
ABSL_FLAG(int, max_candidates, 100, "Maximum candidates to emit per reading.");
ABSL_FLAG(int, alts_per_segment, 8,
          "How many alternate candidates to expand per segment when building "
          "full-path strings.");

namespace mozc {
namespace {

std::string Top1Value(const Segment &segment) {
  if (segment.candidates_size() == 0) {
    return std::string(segment.key());
  }
  return segment.candidate(0).value;
}

std::string ConcatTop1Path(const Segments &segments) {
  std::string out;
  for (size_t i = 0; i < segments.conversion_segments_size(); ++i) {
    out.append(Top1Value(segments.conversion_segment(i)));
  }
  return out;
}

void PushUnique(std::vector<std::string> *values,
                absl::flat_hash_set<std::string> *seen,
                const std::string &value, int max_candidates) {
  if (value.empty() || static_cast<int>(values->size()) >= max_candidates) {
    return;
  }
  if (!seen->insert(value).second) {
    return;
  }
  values->push_back(value);
}

std::vector<std::string> ConvertOne(const ConverterInterface &converter,
                                    const commands::Request &request,
                                    const config::Config &config,
                                    absl::string_view key, int max_candidates,
                                    int alts_per_segment) {
  composer::Composer composer(request, config);
  composer.SetPreeditTextForTestOnly(std::string(key));

  ConversionRequest::Options options = {
      .request_type = ConversionRequest::CONVERSION,
      .max_conversion_candidates_size = std::max(max_candidates, 50),
      .use_actual_converter_for_realtime_conversion = true,
  };

  const ConversionRequest conversion_request =
      ConversionRequestBuilder()
          .SetComposer(composer)
          .SetRequestView(request)
          .SetConfigView(config)
          .SetOptions(std::move(options))
          .Build();

  Segments segments;
  converter.StartConversion(conversion_request, &segments);

  std::vector<std::string> values;
  absl::flat_hash_set<std::string> seen;
  if (segments.conversion_segments_size() == 0) {
    return values;
  }

  // 1) Default best path across all segments.
  PushUnique(&values, &seen, ConcatTop1Path(segments), max_candidates);

  // 2) Expand alternates per segment (others fixed to top-1).
  const size_t n = segments.conversion_segments_size();
  std::vector<std::string> top1(n);
  for (size_t i = 0; i < n; ++i) {
    top1[i] = Top1Value(segments.conversion_segment(i));
  }
  for (size_t i = 0; i < n; ++i) {
    const Segment &segment = segments.conversion_segment(i);
    const int alt_n =
        std::min(alts_per_segment, static_cast<int>(segment.candidates_size()));
    for (int c = 0; c < alt_n; ++c) {
      std::string path;
      for (size_t j = 0; j < n; ++j) {
        if (j == i) {
          path.append(segment.candidate(c).value);
        } else {
          path.append(top1[j]);
        }
      }
      PushUnique(&values, &seen, path, max_candidates);
      if (static_cast<int>(values.size()) >= max_candidates) {
        return values;
      }
    }
  }

  // 3) Force whole reading into one segment, then take its candidate list.
  size_t total_key_len = 0;
  for (size_t i = 0; i < n; ++i) {
    total_key_len += segments.conversion_segment(i).key_len();
  }
  const size_t seg0_len = segments.conversion_segment(0).key_len();
  const int offset =
      static_cast<int>(total_key_len) - static_cast<int>(seg0_len);
  if (n > 1 && offset != 0) {
    Segments single = segments;
    if (converter.ResizeSegment(&single, conversion_request, 0, offset) &&
        single.conversion_segments_size() >= 1) {
      const Segment &whole = single.conversion_segment(0);
      // Prefer whole-segment candidates near the front of the list.
      const int count = std::min(static_cast<int>(whole.candidates_size()),
                                 max_candidates);
      // Insert after best-path by rebuilding: keep current order uniqueness.
      for (int i = 0; i < count; ++i) {
        PushUnique(&values, &seen, whole.candidate(i).value, max_candidates);
        if (static_cast<int>(values.size()) >= max_candidates) {
          break;
        }
      }
    }
  } else if (n == 1) {
    const Segment &whole = segments.conversion_segment(0);
    const int count =
        std::min(static_cast<int>(whole.candidates_size()), max_candidates);
    for (int i = 0; i < count; ++i) {
      PushUnique(&values, &seen, whole.candidate(i).value, max_candidates);
    }
  }

  return values;
}

int RunBatch() {
  const std::string data_path = absl::GetFlag(FLAGS_engine_data_path);
  CHECK(!data_path.empty()) << "--engine_data_path is required";

  absl::StatusOr<std::unique_ptr<const DataManager>> data_manager =
      DataManager::CreateFromFile(data_path);
  CHECK_OK(data_manager) << "failed to load engine data from " << data_path;

  std::unique_ptr<Engine> engine =
      Engine::CreateEngine(*std::move(data_manager)).value();
  std::shared_ptr<const ConverterInterface> converter = engine->GetConverter();

  commands::Request request;
  config::Config config = config::ConfigHandler::DefaultConfig();

  const int max_candidates = absl::GetFlag(FLAGS_max_candidates);
  const int alts_per_segment = absl::GetFlag(FLAGS_alts_per_segment);

  std::istream *in = &std::cin;
  std::unique_ptr<InputFileStream> input_file;
  if (const std::string path = absl::GetFlag(FLAGS_input); !path.empty()) {
    input_file = std::make_unique<InputFileStream>(path);
    in = input_file.get();
  }

  std::ostream *out = &std::cout;
  std::unique_ptr<OutputFileStream> output_file;
  if (const std::string path = absl::GetFlag(FLAGS_output); !path.empty()) {
    output_file = std::make_unique<OutputFileStream>(path);
    out = output_file.get();
  }

  std::string line;
  while (std::getline(*in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> values = ConvertOne(
        *converter, request, config, line, max_candidates, alts_per_segment);
    *out << line;
    for (const std::string &value : values) {
      *out << '\t' << value;
    }
    *out << '\n';
  }
  return 0;
}

}  // namespace
}  // namespace mozc

int main(int argc, char **argv) {
  mozc::InitMozc(argv[0], &argc, &argv);
  return mozc::RunBatch();
}
