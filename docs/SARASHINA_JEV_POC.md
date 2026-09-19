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
