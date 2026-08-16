"""Python HF AutoTokenizer vs C++ token-id parity.

ModernBERT-Ja (`sbintuitions/modernbert-ja-70m`) is LlamaTokenizer +
SentencePiece (`tokenizer.model`), NOT WordPiece. The standalone
`mozc_compat/tokenize_cli` WordPiece path will not match; native ORT
must load tokenizer.model (bos=<s> id 1, eos=</s> id 2). Pair text uses
a literal ` [SEP] ` string from build_pair_text — that is not `<sep>`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.rerank.train_cross_encoder import build_pair_text

CASES = [
    "読み: きしゃ [SEP] 文脈: 新聞の [SEP] 候補: 記者",
    "読み: きしゃ [SEP] 文脈: 駅に [SEP] 候補: 汽車",
    "読み: とうきょう [SEP] 文脈:  [SEP] 候補: 東京",
    build_pair_text("きしゃ", "新聞社の", "記者"),
    build_pair_text("きしゃ", "駅に停まった", "汽車"),
    build_pair_text("あ", "", "亜"),
    "Hello, world!",
    "漢字ひらがなカタカナABC123",
]


def hf_ids(tokenizer, text: str, max_len: int) -> list[int]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    return [int(x) for x in enc["input_ids"]]


def cpp_ids(cli: Path, tok_dir: Path, text: str, max_len: int) -> list[int]:
    p = subprocess.run(
        [str(cli), "--tokenizer", str(tok_dir), "--text", text, "--max-len", str(max_len)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    line = p.stdout.decode("utf-8").strip()
    if not line:
        return []
    return [int(x) for x in line.split()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir (vocab.txt / tokenizer.json)")
    ap.add_argument("--cli", default="", help="C++ tokenize_cli path")
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok_dir = Path(args.tokenizer)
    hf = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)

    n_ok = 0
    n = 0
    cli = Path(args.cli) if args.cli else None
    for text in CASES:
        n += 1
        a = hf_ids(hf, text, args.max_len)
        if cli is None:
            print("HF", a[:16], "...", text[:40])
            n_ok += 1
            continue
        b = cpp_ids(cli, tok_dir, text, args.max_len)
        if a != b:
            print("FAIL", repr(text[:80]))
            print("  hf ", a)
            print("  cpp", b)
        else:
            n_ok += 1
            print("OK", text[:40], "n=", len(a))
    print(f"tokenize parity {n_ok}/{n}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
