# Public-data listwise optimization and CPU validation

The selected ten-layer ModernBERT-ja-30m FP32 model improves CPU latency and
guarded selection on the same 5,974 document-separated validation groups.
See the paired [runtime improvement report](https://github.com/shumaimai/Mozc-Ai/blob/perf/listwise-cpu-20261001/MozcAI_improvement_20260930.md).

| TCP measurement on Intel i7-1060NG7 | Original shipping model/runtime | Selected model/runtime |
| --- | ---: | ---: |
| All requests p50 / p95 | 33.509 / 187.377 ms | 20.442 / 125.938 ms |
| Successful scored requests p50 | 47.440 ms | 30.517 ms |
| Failures at 200 ms deadline | 283 / 5,974 | 145 / 5,974 |
| Accuracy including native fallback | 76.3140% | 76.9166% |
| Helped / hurt | 145 / 42 | 168 / 29 |

Each conversion opens a new TCP connection. Failed requests remain in overall
latency and use native Mozc selection for accuracy. C++ HARD_PROTECT is replayed.
Successful scored percentiles exclude failures and have a censored tail;
all-request p50 includes 1,610 guard skips. The original daemon uses eight
ORT threads; the selected configuration uses four. CPU governor stays powersave.

Uncensored CPU evaluation gives 77.1175% guarded accuracy at tau=1.5,
helped 180 / hurt 29. Scored requests after the first 20 validation rows have
p50 30.463 ms, p95 123.808 ms, p99 288.954 ms; 78 of 4,353 exceed 200 ms.
The long tail remains and is handled by the total communication deadline.
These are validation results. Windows installation and new field logs have
not been evaluated; final_test remains untouched.

## Training and model identity

The public training split has 47,589 rows. Of 31,665 NEURAL_ELIGIBLE rows,
30,411 have at least two unique surfaces and the gold within the original
first 30 candidates. Gold is never added. Protected and coverage-limited
rows remain evaluation-only. Reading and context normalization match the daemon.

The selected checkpoint starts from epoch 1 of the fresh listwise run and
uses one refinement epoch, learning rate 2e-6, group CE + 0.15 BCE and teacher
KL weight 2 at temperature 2. Batches pack 384 candidate sequences with dynamic
padding; seed is 20260930. The teacher fresh run uses learning rate 2e-5.
Training ran on Modal L4; final CPU inference uses ORT 1.30.0 / SentencePiece 0.2.2.

- Selected ONNX: `b2971435a9934c71d8fecbdfd50611c355c4dd12cba32b3244073e4fb8702a6c`
- Selected checkpoint: `ea935215b2c0148bae56277ddd49b3735c04f678b9c996f5d2f273ac281bbd87`
- Parent teacher: `4908941648a9ee9bd08a8ed61bef0f75cbd6f37a6b36b973d05452daa6d78041`
- Original shipping ONNX: `00738ecfd6ee63e25cbc9cb6cfdb2976c381c930e423fc1401d356ef62a04a25`
- Public train: `b8ae9a55f2bb083fe0b15eb8f79be19d3bdfaeb16de85cbc00da64cc7c844d29`
- Public validation: `087008d04e769006561721ec10306c7edb48811abf645bc9d2f7cbc1023a628b`

The launcher mounts only public train and validation files, excludes final_test,
and enforces the public-path privacy gate. No local user input or selection
logs were uploaded. The [completed Modal run](https://modal.com/apps/syuhei2009/main/ap-mGLb6AkBkLSKrhpGGg2XjS)
stores the checkpoint in `mozc-artifacts:/optimization_20260930/listwise_refine/epoch1`.
The selected ONNX is versioned with Git LFS in the public runtime repository.

## Evidence

- [Original TCP benchmark](daemon_original.json) / [selected TCP benchmark](daemon_selected.json).
- [Uncensored CPU validation and GPU score comparison](selected_cpu_report.json).
- [CPU PyTorch versus ONNX parity](refine_parity_cpu.json): all 9,158 token sequences
  and all 800 argmax / tau=1.5 selections agree; maximum score difference 2.122e-5.
- [Original authored diagnostic](everyday_baseline.json) / [selected diagnostic](everyday_selected.json):
  54 cases / 27 context contrast pairs, not training data or real Mozc N-best.
  Final correct counts improve 38 to 44; pairs with both final choices correct 11 to 17.
- [Model manifest](model_manifest.json): training history, dataset hashes and CPU metrics.
  GPU TF32 versus CPU argmax agrees on 5,971/5,974; tau=1.5 selection on 5,973/5,974.
  Reported final quality uses the CPU results.

Other experiments include continued BCE initialization, fresh listwise,
canonical-context fresh listwise, a seven-layer distilled student and INT8.
Smaller/quantized alternatives lose quality; the selected model retains ten
layers and FP32. Intermediate checkpoints and per-example scores remain in
ignored `artifacts/optimization_20260930/`. Older Phase 2 validation evidence
is under `../phase2_format_v2_validation/`.

## Reproduction

Run Modal from the repository root with its SDK authenticated and the public
dataset files present. Fresh training must finish before refinement:

```bash
modal run scripts/modal_listwise_optimization.py --variant fresh
modal run scripts/modal_listwise_optimization.py --variant refine
```

The launcher uses fixed experiment output paths; change the output directory
for an independent rerun to preserve the recorded checkpoint identities.
Evaluation tools accept explicit data, model and runtime paths:

```bash
python -m tools.rerank.evaluate_cpu_models \
  --runtime ../Mozc-Ai/runtime/rerank_daemon.py \
  --data data/public/contextual_ranking_v2_production_runtime_context/dataset/validation.jsonl.gz \
  --model ../Mozc-Ai/runtime/model/cross_encoder_fp32.onnx \
  --tokenizer ../Mozc-Ai/runtime/model/tokenizer/tokenizer.model \
  --threads 4 --out-dir artifacts/recheck_cpu

# Start the paired resident runtime before benchmarking.
python -m tools.rerank.daemon_latency_bench \
  --data data/public/contextual_ranking_v2_production_runtime_context/dataset/validation.jsonl.gz \
  --timeout-ms 200 --out artifacts/recheck_tcp.json
```

All 53 reranker tests and 14 dataset tests pass locally. Runtime and TCP tests
use the sibling `../Mozc-Ai` checkout by default; set `MOZCAI_RUNTIME_ROOT`
to another checkout root when needed. CI checks out the paired public runtime
commit and installs only CPU test dependencies. Candidate preparation contract
tests need no model, GPU or torch import.

## Review follow-up (2026-10-01)

The paired runtime now discards unknown diagnostic string values at the logging
boundary, reads numeric policy tokens correctly (including the shipped tau=1.5),
and verifies the installed ONNX hash against both ping and scored responses.
The same identity check is also enforced after MSI extraction in release CI.

`test_cpp_review_contract.py` compiles the actual diagnostic and guard modules,
file reader, JSON helpers and `LoadPolicyFile` with the real Abseil library.
Only the Mozc class shell and LOG macro are isolated from the full Mozc build.
Nine tests cover response text/control characters, known and unknown reason
codes, null values, shipped policy values, whitespace/scientific numbers and
fallback defaults. The original reviewed commit fails the tau and privacy
regressions; the repaired CPU suite passes all 62 reranker and 14 dataset tests.
Install `g++`, `pkg-config` and `libabsl-dev` before running these C++ tests.

`test_windows_smoke.ps1` parses the actual runtime smoke and loads only its
SHA validator. Eight cases cover matching hashes, either response identifying
another model, both responses agreeing on another model, missing/malformed
hashes and hex case. CI runs this on Windows PowerShell and PowerShell 7 without
installing an IME or downloading the model. Full MSI/IME validation remains
separate from these contract tests.
The full-script parse also caught Windows PowerShell 5.1 misreading the existing
UTF-8 Japanese literals; the runtime smoke now carries a UTF-8 BOM so direct
`powershell -File` execution uses the correct encoding.
