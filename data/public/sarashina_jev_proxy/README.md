# Sarashina-JEV public proxy dataset

This directory contains the exact public proxy data used by the Sarashina-JEV
QAT and vocabulary-pruning experiments. It is **not** real Mozc N-best data and
must not be interpreted as production IME accuracy data.

## Contents

- `train.jsonl`: 1,796 rows
- `eval.jsonl`: 204 rows
- `meta.json`: source snapshot and generation parameters
- `CHECKSUMS.sha256`: SHA-256 checksums for the three files

Every row has exactly five candidates and the gold surface is present in those
five candidates. Candidate order is corpus-frequency order, so candidate 0 is a
frequency-biased proxy for the Mozc baseline.

## Provenance and license

- Source dataset: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
- Snapshot/config: `20231101.ja`
- Upstream text license recorded by the generator: `CC-BY-SA-3.0/GFDL`
- Reading generation: `SudachiPy` with `SudachiDict-core`
- Generator: [`tools/sarashina_jev/bootstrap_public.py`](../../../tools/sarashina_jev/bootstrap_public.py)

The Wikipedia-derived text remains subject to the upstream terms:

- [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html)
- [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)

Attribution is provided through the source name, source URL, snapshot, license
identifier, and source article ID stored in each JSONL row. Users redistributing
modified versions must preserve the applicable attribution and share-alike terms.

## Reproduction

The exact dataset parameters were:

```text
scan_articles=1500
example_articles=1500
max_examples=2000
eval_ratio=0.1
```

With Modal configured, regenerate it with:

```bash
modal run scripts/modal_sarashina_jev.py::bootstrap_public_data \
  --scan-articles 1500 \
  --example-articles 1500 \
  --max-examples 2000
```

The split is deterministic from the Wikipedia article ID, sentence index, and
morpheme index. Upstream streaming order or dependency-version changes can still
change a newly generated snapshot; use the committed files and checksums for an
exact analysis of this run.

## Privacy review

These files were generated only from the public Wikipedia dataset. They do not
contain IME usage logs, chats, local documents, API credentials, or local file
paths. A pre-publication scan found no email addresses, phone/postal-number
patterns, IP addresses, credentials, or user-home paths. Public encyclopedia
text can naturally mention public people and places; that is upstream public
content rather than private user data.
