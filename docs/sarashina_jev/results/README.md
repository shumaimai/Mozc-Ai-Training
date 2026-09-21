# Sarashina-JEV QAT and vocabulary-ablation snapshot

Snapshot date: 2026-09-21. All accuracy values use the same 204-page public
proxy evaluation set with five candidates per page. One page is approximately
0.49 percentage points.

## Authoritative comparison

| Vocabulary | FP32 Hit@1 | Dynamic INT8 + FP16 embedding Hit@1 | Model size | p50 | Token-length ratio | Changed inputs | New UNK |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 102,400 | 84.8039% | 81.8627% | 419.8135 MiB | 134.43 ms | 1.0000 | 0% | 0 |
| **64,000** | **84.3137%** | **81.3725%** | **326.0635 MiB** | **162.73 ms** | **1.02827** | 48.24% | 0 |
| 48,000 | 83.8235% | 78.9216% | 287.0010 MiB | 163.02 ms | 1.08183 | 100% | 0 |

The 64k vocabulary is the current recommendation. It saves 93.75 MiB from the
FP16 embedding relative to 102.4k while losing one of 204 INT8 decisions. The
48k vocabulary loses another five decisions versus 64k and does not improve
latency in the fresh-container measurements, so the experiment did not proceed
to 32k. A 56k compromise is the next useful vocabulary point if this line of
work continues.

Latency across the historical 102.4k run and fresh 64k/48k revalidations is not
strictly comparable because Modal assigned different CPU classes. The 64k and
48k fresh revalidations were both about 163 ms, so vocabulary reduction should
be treated as a size optimization rather than a latency optimization.

## Important ONNX Runtime anomaly

The original 64k and 48k export containers reported catastrophic INT8 Hit@1 of
25.49% and 19.61%, while FP32 stayed at 84.31% and 83.82%. The exact same ONNX
files evaluated in fresh Modal containers produced 81.37% and 78.92%.

For the 64k model, comparison against the 102.4k baseline showed identical graph
structure, identical non-embedding INT8 initializers, and identical embedding
rows for every token ID used by the evaluation set. This points to a
CPU/ONNX-Runtime quantized-kernel issue rather than checkpoint or export
corruption.

The JSON files under `raw/vocab_64000` and `raw/vocab_48000` are preserved
unchanged for forensic analysis. Their embedded 25.49%/19.61% INT8 measurements
are the anomalous runs and are **not** the authoritative accuracy values. Use
[`summary.json`](summary.json) for the fresh-container results.

## Files

- `raw/qat_v2_repro/`: QAT v2 training arguments, history, model metadata, and
  the original 102.4k ONNX benchmark report.
- `raw/vocab_64000/`: unmodified 64k prune/export reports.
- `raw/vocab_48000/`: unmodified 48k prune/export reports.
- `raw/fresh_revalidation.json`: exact fresh-container evaluation stdout values.
- `raw/onnx_compare_64000.json`: graph/initializer/embedding comparison used to
  isolate the CPU-dependent INT8 anomaly.
- `summary.json`: normalized comparison including fresh revalidation metrics.
- `artifact_manifest.json`: large Modal artifacts intentionally excluded from Git.
- `PRIVACY_AUDIT.md`: publication privacy/security review.

Model weights and ONNX files are not committed here. The largest artifact is
1.1 GiB and normal GitHub files are limited to 100 MiB. The base
[`sbintuitions/sarashina2.2-0.5b`](https://huggingface.co/sbintuitions/sarashina2.2-0.5b)
model is MIT-licensed; derived artifacts remain in the private Modal volume and
can be regenerated with the committed scripts and proxy data.
