# Publication privacy and secret audit

Audit date: 2026-09-21

## Scope

- `data/public/sarashina_jev_proxy/{train,eval}.jsonl`
- `data/public/sarashina_jev_6l_holdout/holdout.jsonl` and provenance/checksum
- dataset metadata and checksums
- QAT v2, 64k, 48k, and 6L JSON reports/tokenizer metadata
- repository working tree and full Git patch history for high-confidence secret patterns

## Findings

- Credential/token/private-key patterns: **0**
- Email-address patterns in the added dataset/reports: **0**
- Phone-number patterns: **0**
- Japanese postal-code patterns: **0**
- IPv4-address patterns: **0**
- User-home paths (`/home/<user>`, `C:\\Users\\<user>`): **0**
- IME usage logs, chats, local documents, or private fine-tuning data: **not present**
- Holdout source-id overlap with existing train/eval: **0**

The reports contain generic Modal paths such as `/artifacts/...` and `/data/...`;
these do not identify a local user. The dataset contains public Wikipedia text,
which can mention public people and places. No non-public/user-originated data
was used.

GitHub secret scanning is not enabled for this repository, so the pre-push audit
used local pattern scanning and JSON/schema checks. All newly created commits use
the GitHub noreply author address. Existing public history was not rewritten.

## Integrity checks

- Both JSONL files parse successfully as UTF-8 JSON.
- Train/eval row counts are 1,796/204.
- The fresh holdout has 800 rows and its SHA-256 matches `CHECKSUMS.sha256`.
- Every row has exactly five candidates.
- Every row has `gold_in_nbest=true`.
- All result JSON files parse successfully.
- No added file exceeds GitHub's 100 MiB per-file limit.
