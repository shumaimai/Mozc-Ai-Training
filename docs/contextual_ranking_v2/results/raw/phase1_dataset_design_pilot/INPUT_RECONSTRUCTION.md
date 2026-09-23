# Pilot input reconstruction note

The 100 `source_id`/title/page identities are the same as the Phase 1 pilot
manifest. The original temporary `wiki_documents.jsonl` was not retained after
the previous session's `/tmp` directory was reclaimed. For this pilot, those
same page identities were reconstructed from the Japanese Wikipedia API on
2026-09-22 using plain-text extracts capped at 1,500 characters per document.

Therefore this result is **document-identity reproducible**, but it is not a
byte-identical replay of the original 2023-11-01 Wikimedia dump text. A final
Dataset v2 run must freeze and checksum the actual dump-derived input shards
before extraction. No model training or production-scale generation was run
here.
