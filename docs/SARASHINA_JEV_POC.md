# Sarashina-JEV IME PoC

Status: experimental. This path does not replace the v1.0 ModernBERT reranker.

## Goal

Test whether a Japanese generative model keeps useful difficult-language knowledge after removing generation and physically pruning transformer layers.

Base model: `sbintuitions/sarashina2.2-0.5b` (Llama, 24 layers, hidden 1280, vocab 102400).

PoC rules:

- load `AutoModel`, not `AutoModelForCausalLM` -> no LM head;
- no text generation, sampling, beam search, or token-by-token decoding;
- prune 24 layers to 12 by uniform layer selection for the first experiment;
- score only the currently visible Mozc page, default 5 candidates;
- one forward receives the five candidate sequences as one batch;
- train with listwise cross entropy across the five scores;
- a gold-containing page learns the gold position;
- when the first page does not contain gold, a low-weight anchor teaches "keep Mozc #1" rather than making a random reorder.

The first question is not "is this the final shippable model?" It is:

> Does a brutally reduced Sarashina retain better context judgment than the current small reranker on hard Japanese phrasing?

## Layout

- `tools/sarashina_jev/model.py`: load backbone, remove LM head by construction, uniform layer pruning, scalar score head.
- `tools/sarashina_jev/data.py`: convert existing `reading/context_prev/mozc_nbest/gold` JSONL into 5-candidate pages.
- `tools/sarashina_jev/train.py`: partial/full fine-tuning with listwise CE.
- `tools/sarashina_jev/eval.py`: page Hit@1, Mozc baseline on gold pages, anchor keep rate, and p50/p95 latency.
- `tools/sarashina_jev/test_core.py`: light tests for pruning/page logic.

## Smoke test

```bash
python -m unittest tools.sarashina_jev.test_core -v
```

## First GPU run

Use a small limit first:

```bash
python -m tools.sarashina_jev.train \
  --train data/rerank_ctx/train_v2_clean.jsonl \
  --eval data/rerank_ctx/unseen.jsonl \
  --model sbintuitions/sarashina2.2-0.5b \
  --keep-layers 12 \
  --page-size 5 \
  --train-last-n-layers 4 \
  --gradient-checkpointing \
  --bf16 \
  --limit 2000 \
  --out artifacts/sarashina_jev/12l_smoke
```

If bf16 is not stable on the target ROCm GPU, replace `--bf16` with `--fp16`.

Then evaluate:

```bash
python -m tools.sarashina_jev.eval \
  --artifact artifacts/sarashina_jev/12l_smoke \
  --data data/rerank_ctx/unseen.jsonl
```

## Full experiment matrix

Run the same data and settings for:

- 24 layers: upper bound after LM-head removal.
- 12 layers: first main PoC.
- 8 layers: aggressive speed target.
- current ModernBERT 30M: existing baseline.

Report at minimum:

- gold-page Hit@1;
- gold-page Mozc Hit@1;
- improvement / regression counts;
- anchor keep rate;
- p50 / p95 latency;
- total and trainable parameter counts.

Do not shrink the tokenizer/vocabulary in this first experiment. Layer pruning and LM-head removal should be isolated first; vocabulary compression comes only after we know whether language understanding survives.


## Modal (recommended when away from the GPU machine)

The PoC branch includes `scripts/modal_sarashina_jev.py`.

Keep training data in a persistent Modal Volume instead of embedding it in every image:

```bash
modal volume create mozc-training-data

modal volume put mozc-training-data \
  data/public/rerank_ctx/train_v2.jsonl \
  /train_v2.jsonl

modal volume put mozc-training-data \
  data/public/rerank_ctx/eval_unseen_v2.jsonl \
  /eval_unseen_v2.jsonl
```

Then launch the 12-layer smoke run:

```bash
modal run scripts/modal_sarashina_jev.py \
  --keep-layers 12 \
  --train-last-n-layers 4 \
  --page-size 5 \
  --limit 2000 \
  --out /artifacts/sarashina_jev/12l_smoke
```

The local entrypoint uses `spawn()`, so the GPU function continues independently after the launcher exits. The default GPU is L4. Artifacts persist in `mozc-artifacts`, model downloads in `hf-cache`, and datasets in `mozc-training-data`.

Inspect later:

```bash
modal volume ls mozc-artifacts /sarashina_jev/12l_smoke
modal volume get mozc-artifacts /sarashina_jev/12l_smoke ./12l_smoke
```

If the dataset filenames differ, override `--train-path` and `--eval-path`.


## Cloud-only bootstrap (no home PC required)

For the pruning experiment, Modal can generate a proxy dataset itself from the public Japanese Wikipedia dataset. It finds surface forms sharing a Sudachi reading, ranks five candidates by corpus frequency, and uses the real surrounding sentence as context.

This is only for answering the pruning question: whether difficult Japanese context knowledge survives 24 -> 12/8 layers. It is not a replacement for the final real Mozc N-best dataset.

```bash
modal run scripts/modal_sarashina_jev.py \
  --bootstrap \
  --keep-layers 12 \
  --train-last-n-layers 4 \
  --page-size 5 \
  --limit 2000 \
  --out /artifacts/sarashina_jev/12l_public_smoke
```

The launcher first runs a CPU bootstrap function, commits the generated JSONL to `mozc-training-data`, then spawns the L4 training function. Default bootstrap scans 6000 Wikipedia articles and caps the generated dataset at 6000 examples.

For a smaller first trial:

```bash
modal run scripts/modal_sarashina_jev.py \
  --bootstrap \
  --bootstrap-scan-articles 1500 \
  --bootstrap-example-articles 1500 \
  --bootstrap-max-examples 2000 \
  --limit 1500 \
  --out /artifacts/sarashina_jev/12l_public_quick
```
