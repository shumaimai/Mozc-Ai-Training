# Modal public-data staging area

Only files placed deliberately in this directory are uploaded by the v1 Modal
launchers. Generated public train/evaluation JSONL files may be copied here
immediately before a cloud run.

Never place conversion logs, personal usage data, chats, API credentials, or a
private fine-tune in this directory. The launchers also reject paths containing
`private`, `personal`, `usage`, or `log`.

Inside a Modal container this directory is mounted as `data/rerank_ctx`, so the
existing command-line defaults remain unchanged.

