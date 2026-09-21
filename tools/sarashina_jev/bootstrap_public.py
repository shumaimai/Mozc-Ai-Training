"""Build a public homophone-ranking proxy dataset entirely in the cloud.

The dataset is intentionally a PoC proxy, not a replacement for real Mozc N-best.
It uses Japanese Wikipedia context and Sudachi readings, then creates five surface
forms sharing one reading. Candidate order is global frequency, approximating a
frequency-biased IME baseline.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from sudachipy import dictionary, tokenizer as sudachi_tokenizer


DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.ja"
SOURCE_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
LICENSE_ID = "CC-BY-SA-3.0/GFDL"
SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def kata_to_hira(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def valid_surface(text: str) -> bool:
    if not (1 <= len(text) <= 10):
        return False
    if any(ch.isspace() for ch in text):
        return False
    # Prefer conversion-worthy forms; skip pure punctuation/ASCII and pure hiragana.
    has_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
    has_katakana = any(0x30A0 <= ord(ch) <= 0x30FF for ch in text)
    return has_kanji or has_katakana


def article_stream(max_articles: int, *, start_article: int = 0) -> Iterable[dict]:
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )
    for i, row in enumerate(ds):
        if i < start_article:
            continue
        if i >= start_article + max_articles:
            break
        yield row


def iter_morphemes(text: str, tok):
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C
    for sent_index, match in enumerate(SENTENCE_RE.finditer(text)):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        for morph_index, morph in enumerate(tok.tokenize(sentence, mode)):
            surface = morph.surface()
            if not valid_surface(surface):
                continue
            pos0 = morph.part_of_speech()[0]
            if pos0 not in {"名詞", "動詞", "形容詞", "形状詞"}:
                continue
            reading = kata_to_hira(morph.reading_form())
            if not reading or reading == "*":
                continue
            try:
                begin = int(morph.begin())
            except Exception:
                begin = sentence.find(surface)
            context = sentence[: max(0, begin)][-80:]
            yield sent_index, morph_index, reading, surface, context


def build_candidate_map(
    max_articles: int,
    min_forms: int = 5,
    *,
    start_article: int = 0,
) -> dict[str, list[str]]:
    tok = dictionary.Dictionary(dict="core").create()
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in article_stream(max_articles, start_article=start_article):
        text = str(row.get("text") or "")
        for _, _, reading, surface, _ in iter_morphemes(text, tok):
            counts[reading][surface] += 1

    out: dict[str, list[str]] = {}
    for reading, counter in counts.items():
        if len(counter) < min_forms:
            continue
        # frequency-desc, lexical tie break for reproducibility
        ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        out[reading] = [surface for surface, _ in ranked[:5]]
    return out


def stable_eval(article_id: str, sent_index: int, morph_index: int, eval_ratio: float) -> bool:
    key = f"{article_id}:{sent_index}:{morph_index}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return bucket < eval_ratio


def generate_rows(
    candidates: dict[str, list[str]],
    *,
    max_articles: int,
    max_examples: int,
    eval_ratio: float,
    start_article: int = 0,
):
    tok = dictionary.Dictionary(dict="core").create()
    train: list[dict] = []
    eval_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for row in article_stream(max_articles, start_article=start_article):
        article_id = str(row.get("id") or "")
        text = str(row.get("text") or "")
        for sent_i, morph_i, reading, surface, context in iter_morphemes(text, tok):
            cands = candidates.get(reading)
            if not cands or surface not in cands:
                continue
            dedup_key = (reading, surface, context)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            example = {
                "reading": reading,
                "context_prev": context,
                "gold": surface,
                "mozc_nbest": cands,
                "gold_in_nbest": True,
                "source": "wikimedia/wikipedia",
                "source_id": article_id,
                "source_url": SOURCE_URL,
                "license_id": LICENSE_ID,
                "reading_source": "SudachiDict-core",
                "reading_confidence": 1.0,
                "proxy_dataset": True,
            }
            target = eval_rows if stable_eval(article_id, sent_i, morph_i, eval_ratio) else train
            target.append(example)
            if len(train) + len(eval_rows) >= max_examples:
                return train, eval_rows
    return train, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_public_proxy(
    out_dir: str | Path,
    *,
    scan_articles: int = 6000,
    example_articles: int = 6000,
    max_examples: int = 6000,
    eval_ratio: float = 0.1,
    scan_start_article: int = 0,
    example_start_article: int = 0,
) -> dict:
    out = Path(out_dir)
    candidate_map = build_candidate_map(
        scan_articles,
        start_article=scan_start_article,
    )
    train, eval_rows = generate_rows(
        candidate_map,
        max_articles=example_articles,
        max_examples=max_examples,
        eval_ratio=eval_ratio,
        start_article=example_start_article,
    )
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", eval_rows)
    meta = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "source_url": SOURCE_URL,
        "license_id": LICENSE_ID,
        "scan_articles": scan_articles,
        "scan_start_article": scan_start_article,
        "example_articles": example_articles,
        "example_start_article": example_start_article,
        "candidate_readings": len(candidate_map),
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "max_examples": max_examples,
        "purpose": "Sarashina-JEV pruning PoC proxy only; not final Mozc N-best training data",
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/sarashina_jev/public_proxy")
    parser.add_argument("--scan-articles", type=int, default=6000)
    parser.add_argument("--example-articles", type=int, default=6000)
    parser.add_argument("--max-examples", type=int, default=6000)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--scan-start-article", type=int, default=0)
    parser.add_argument("--example-start-article", type=int, default=0)
    args = parser.parse_args()
    meta = build_public_proxy(
        args.out,
        scan_articles=args.scan_articles,
        example_articles=args.example_articles,
        max_examples=args.max_examples,
        eval_ratio=args.eval_ratio,
        scan_start_article=args.scan_start_article,
        example_start_article=args.example_start_article,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
