# Sarashina-JEV 6L fresh public holdout

This is a public Wikipedia-derived holdout generated for the 64k-vocabulary
8L→6L experiment. It contains **800 pages** and is not production Mozc N-best
data.

- Source: `wikimedia/wikipedia`, config `20231101.ja`
- Candidate readings: built from Wikipedia articles `[0,1500)`
- Holdout examples: separate articles `[1500,4500)`
- Split: `eval_ratio=1.0`, deterministic streaming order, seed-free because the article range is the split key
- `source_id_overlap_count` with the existing train/eval: `0`
- Generator: [`tools/sarashina_jev/bootstrap_public.py`](../../../tools/sarashina_jev/bootstrap_public.py)
- Checksum: [`CHECKSUMS.sha256`](CHECKSUMS.sha256)

The upstream text remains subject to CC-BY-SA-3.0/GFDL and Wikimedia terms.
Public names and places may occur naturally; no private IME logs, chats, local
documents, credentials, or local paths were used.
