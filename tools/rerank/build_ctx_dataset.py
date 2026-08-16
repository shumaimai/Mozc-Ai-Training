"""Build contextual reranker datasets (PLAN_CONTEXTUAL_RERANKER.md).

Pipeline (ROCm training later; this script is CPU/WSL + Windows mozc_batch):
  1) fetch corpus texts (wiki / news)
  2) Sudachi tokenize → reading/gold/context_prev records
  3) Mozc N-best attach
  4) ambiguous map + split + sample

Example (small scale):
  python -m tools.rerank.build_ctx_dataset fetch-wiki --out data/rerank_ctx/raw/wiki_docs.jsonl --n 400
  python -m tools.rerank.build_ctx_dataset extract --docs data/rerank_ctx/raw/wiki_docs.jsonl --out data/rerank_ctx/work/extracted.jsonl
  python -m tools.rerank.build_ctx_dataset mozc-attach --extracted data/rerank_ctx/work/extracted.jsonl --work-dir data/rerank_ctx/work/mozc
  python -m tools.rerank.build_ctx_dataset assemble-small --attached data/rerank_ctx/work/mozc/attached.jsonl --out-dir data/rerank_ctx
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.dataset.jsonl import read_jsonl, write_jsonl
from tools.dataset.mozc_batch import (
    MozcBatchModeARequired,
    join_reading_candidates,
    load_candidates_map,
    normalize_mozc_key,
    resolve_batch_config,
    run_mozc_batch_boundary,
    unique_keys,
    write_keys_txt,
)
from tools.dataset.normalize import katakana_to_hiragana, normalize_reading, normalize_surface
from tools.rerank.context_clip import (
    clean_context,
    clip_context_prev,
    has_kanji,
    sanitize_nbest,
)
from tools.rerank.prepare import find_gold_rank

# Sudachi POS prefixes to keep as content words (exclude particles/symbols/etc.)
_CONTENT_POS1 = {
    "名詞",
    "動詞",
    "形容詞",
    "形状詞",  # Sudachi adjectival noun
    "副詞",
    "連体詞",
}
_FUNC_POS1 = {"助詞", "助動詞", "接続詞", "記号", "補助記号", "フィラー", "感動詞"}
_SKIP_POS2 = {"数詞", "非自立可能"}  # soft skip for pure counters when surface is digits-only
_DIGITS_ONLY = re.compile(r"^[\d０-９一二三四五六七八九十百千万億兆\.\,．]+$")
_URL_OR_AT = re.compile(r"(https?://|www\.|@)")

# Track B base note (1-var A/B vs current CE)
_CE_BASE_NOTE = (
    "Track A/B base: sbintuitions/modernbert-ja-70m continued from "
    "artifacts/rerank/modernbert70m_ce_v3"
)


def _hiragana_reading(raw: str) -> str:
    return normalize_reading(katakana_to_hiragana(unicodedata.normalize("NFKC", raw or "")))


def _is_content_token(pos: list[str], surface: str) -> bool:
    if not pos:
        return False
    if pos[0] not in _CONTENT_POS1:
        return False
    if pos[0] == "名詞" and len(pos) > 1 and pos[1] in _SKIP_POS2 and _DIGITS_ONLY.match(surface or ""):
        return False
    if _DIGITS_ONLY.match(surface or ""):
        return False
    if _URL_OR_AT.search(surface or ""):
        return False
    return True


def is_content_pos(pos: list[str] | None) -> bool:
    """True for noun/verb/adj/adv-family; excludes particles/aux/conj/symbol/filler/interjection."""
    if not pos:
        return False
    p0 = pos[0] if isinstance(pos, list) else str(pos)
    if p0 in _FUNC_POS1:
        return False
    return p0 in _CONTENT_POS1


def _qc_ok(reading: str, gold: str) -> bool:
    if not reading or not gold:
        return False
    if not (1 <= len(reading) <= 16):
        return False
    if not (1 <= len(gold) <= 12):
        return False
    # reading should be hiragana-ish
    if not re.fullmatch(r"[\u3041-\u3096ー]+", reading):
        return False
    return True


def fetch_wiki_from_abstract_dump(out_path: Path, n: int = 400) -> int:
    """Download jawiki abstracts dump and take first n usable articles (no API rate limit)."""
    import gzip
    import xml.etree.ElementTree as ET

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path = out_path.parent / "jawiki-latest-abstract.xml.gz"
    url = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-abstract.xml.gz"
    if not dump_path.exists() or dump_path.stat().st_size < 1_000_000:
        print(f"downloading {url} -> {dump_path}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "mozc-ai-ctx-builder/0.1"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(dump_path, "wb") as fout:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
                if dump_path.stat().st_size % (20 * 1024 * 1024) < 1024 * 1024:
                    print(f"  downloaded_mb={dump_path.stat().st_size/1024/1024:.1f}", flush=True)
    docs: list[dict[str, Any]] = []
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as fin:
        # Stream doc by doc: <doc>...</doc>
        buf: list[str] = []
        in_doc = False
        for line in fin:
            if "<doc>" in line:
                in_doc = True
                buf = [line]
                continue
            if in_doc:
                buf.append(line)
                if "</doc>" in line:
                    in_doc = False
                    block = "".join(buf)
                    try:
                        # abstracts use <title> <url> <abstract>
                        title_m = re.search(r"<title>(.*?)</title>", block, re.S)
                        abs_m = re.search(r"<abstract>(.*?)</abstract>", block, re.S)
                        url_m = re.search(r"<url>(.*?)</url>", block, re.S)
                        title = re.sub(r"^Wikipedia:\s*", "", (title_m.group(1) if title_m else "").strip())
                        abstract = (abs_m.group(1) if abs_m else "").strip()
                        abstract = re.sub(r"\s+", " ", abstract)
                        if len(abstract) < 80 or abstract.count("。") < 1:
                            continue
                        # skip non-article namespaces
                        if ":" in title and not title.startswith("Wikipedia:"):
                            # allow normal titles; skip Talk: etc if present
                            if any(title.startswith(p) for p in ("利用者:", "Template:", "ファイル:", "Category:", "Wikipedia:")):
                                continue
                        pid = url_m.group(1).rstrip("/").split("/")[-1] if url_m else str(len(docs))
                        docs.append(
                            {
                                "doc_id": f"wiki_{pid}",
                                "source": "wiki",
                                "title": title,
                                "text": abstract + ("。" if not abstract.endswith("。") else ""),
                            }
                        )
                    except Exception:
                        pass
                    if len(docs) >= n:
                        break
            if len(docs) >= n:
                break
    write_jsonl(out_path, docs)
    print(f"wiki_abstract_docs={len(docs)} -> {out_path}", flush=True)
    return len(docs)


def fetch_aozora_docs(out_path: Path, n: int = 80) -> int:
    """Build docs from public-domain aozora works listed in interim index (best-effort)."""
    from tools.dataset.aozora import download_work_text, public_domain_works, strip_boilerplate
    from tools.dataset.jsonl import read_jsonl as _rj

    index_path = Path("data/interim/aozora_index.jsonl")
    # Prefer ruby interim contexts if full fetch is slow: expand metadata.context
    ruby_path = Path("data/interim/aozora_ruby.jsonl")
    docs: list[dict[str, Any]] = []
    by_work: dict[str, list[str]] = defaultdict(list)
    if ruby_path.exists():
        for row in _rj(ruby_path):
            meta = row.get("metadata") or {}
            wid = str(meta.get("work_id") or "unk")
            ctx = (meta.get("context") or "").strip()
            if ctx:
                by_work[wid].append(ctx)
        for wid, ctxs in by_work.items():
            text = "。".join(dict.fromkeys(ctxs))  # unique preserve order
            if len(text) < 80:
                continue
            docs.append(
                {
                    "doc_id": f"aozora_{wid}",
                    "source": "news_old",  # treat as older literary corpus in train pool
                    "title": f"aozora:{wid}",
                    "text": text if text.endswith("。") else text + "。",
                }
            )
            if len(docs) >= n:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, docs[:n])
    print(f"aozora_docs={min(len(docs), n)} -> {out_path}", flush=True)
    return min(len(docs), n)


def fetch_wiki_from_hf(out_path: Path, n: int = 800) -> int:
    """Stream Japanese Wikipedia from HuggingFace (wikimedia/wikipedia)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = []
    # 20231101.ja is a common snapshot id; try a few
    errors: list[str] = []
    ds = None
    for config in ("20231101.ja", "20220301.ja"):
        try:
            ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
            print(f"hf_wikipedia_config={config}", flush=True)
            break
        except Exception as exc:
            errors.append(f"{config}:{exc}")
    if ds is None:
        raise RuntimeError("hf wikipedia load failed: " + " | ".join(errors))
    for i, row in enumerate(ds):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not text or len(text) < 120:
            continue
        # keep first ~1500 chars (enough sentences for extraction)
        text = text[:1500]
        if text.count("。") < 2:
            continue
        docs.append(
            {
                "doc_id": f"wiki_hf_{i}",
                "source": "wiki",
                "title": title,
                "text": text,
            }
        )
        if len(docs) % 50 == 0:
            write_jsonl(out_path, docs)
            print(f"hf_wiki_docs={len(docs)}/{n}", flush=True)
        if len(docs) >= n:
            break
    write_jsonl(out_path, docs)
    print(f"hf_wiki_docs={len(docs)} -> {out_path}", flush=True)
    return len(docs)


def fetch_wiki_docs(n: int, out_path: Path, seed: int = 42) -> int:
    """Prefer HF stream, then abstract dump, then API."""
    try:
        return fetch_wiki_from_hf(out_path, n=n)
    except Exception as exc:
        print(f"hf_wiki_failed: {exc}", flush=True)
    try:
        return fetch_wiki_from_abstract_dump(out_path, n=n)
    except Exception as exc:
        print(f"abstract_dump_failed: {exc}; falling back to API", flush=True)
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            docs.append(row)
            try:
                seen_ids.add(int(str(row.get("doc_id", "")).replace("wiki_", "")))
            except ValueError:
                pass
        print(f"resume wiki_docs={len(docs)}", flush=True)
    fails = 0
    while len(docs) < n:
        need = min(5, n - len(docs))
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": str(need),
            "prop": "extracts|info",
            "explaintext": "1",
            "exlimit": str(need),
            "exchars": "1200",
        }
        url = "https://ja.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "mozc-ai-ctx-builder/0.1 (research; local)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            fails = 0
        except Exception as exc:
            fails += 1
            wait = min(180.0, 3.0 ** min(fails, 5) + rng.random())
            print(f"wiki_fetch_err {exc}; sleep {wait:.1f}s", flush=True)
            time.sleep(wait)
            if fails > 40:
                break
            continue
        pages = (payload.get("query") or {}).get("pages") or {}
        added = 0
        for _, page in pages.items():
            pid = int(page.get("pageid") or 0)
            if not pid or pid in seen_ids:
                continue
            title = (page.get("title") or "").strip()
            text = (page.get("extract") or "").strip()
            if not text or len(text) < 80:
                continue
            seen_ids.add(pid)
            docs.append(
                {
                    "doc_id": f"wiki_{pid}",
                    "source": "wiki",
                    "title": title,
                    "text": text,
                }
            )
            added += 1
            if len(docs) >= n:
                break
        write_jsonl(out_path, docs)
        print(f"wiki_docs={len(docs)}/{n} (+{added})", flush=True)
        time.sleep(3.0 + rng.random() * 2.0)
    write_jsonl(out_path, docs)
    return len(docs)


def fetch_news_rss(out_path: Path, *, fresh: bool, limit: int = 80) -> int:
    """Fetch Japanese news RSS items as crude news docs."""
    feeds = [
        "https://www.nhk.or.jp/rss/news/cat0.xml",
        "https://www.nhk.or.jp/rss/news/cat1.xml",
        "https://www.nhk.or.jp/rss/news/cat2.xml",
        "https://www.nhk.or.jp/rss/news/cat3.xml",
        "https://www.nhk.or.jp/rss/news/cat4.xml",
        "https://www.nhk.or.jp/rss/news/cat5.xml",
        "https://www.nhk.or.jp/rss/news/cat6.xml",
        "https://news.yahoo.co.jp/rss/topics/top-picks.xml",
        "https://news.yahoo.co.jp/rss/topics/domestic.xml",
        "https://news.yahoo.co.jp/rss/topics/world.xml",
        "https://news.yahoo.co.jp/rss/topics/business.xml",
        "https://news.yahoo.co.jp/rss/topics/entertainment.xml",
        "https://news.yahoo.co.jp/rss/topics/sports.xml",
        "https://news.yahoo.co.jp/rss/topics/it.xml",
        "https://news.yahoo.co.jp/rss/topics/science.xml",
        "https://www.asahi.com/rss/asahi/newsheadlines.rdf",
        "https://www.yomiuri.co.jp/feed/",
    ]
    import xml.etree.ElementTree as ET

    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    source = "news_fresh" if fresh else "news_old"
    for feed in feeds:
        try:
            req = urllib.request.Request(feed, headers={"User-Agent": "mozc-ai-ctx-builder/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
        except Exception as exc:
            print(f"rss_fail {feed}: {exc}", flush=True)
            continue
        items = root.findall(".//item") or root.findall(".//{http://purl.org/rss/1.0/}item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        for item in items:
            title = (
                item.findtext("title")
                or item.findtext("{http://purl.org/rss/1.0/}title")
                or item.findtext("{http://www.w3.org/2005/Atom}title")
                or ""
            ).strip()
            desc = (
                item.findtext("description")
                or item.findtext("{http://purl.org/rss/1.0/}description")
                or item.findtext("{http://www.w3.org/2005/Atom}summary")
                or item.findtext("{http://www.w3.org/2005/Atom}content")
                or ""
            ).strip()
            desc = re.sub(r"<[^>]+>", "", desc)
            link = (
                item.findtext("link")
                or item.findtext("{http://purl.org/rss/1.0/}link")
                or ""
            ).strip()
            if not link:
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
                if link_el is not None:
                    link = (link_el.get("href") or "").strip()
            text = f"{title}。{desc}".strip("。") + "。"
            key = link or title
            if len(text) < 40 or key in seen:
                continue
            seen.add(key)
            docs.append(
                {
                    "doc_id": f"{source}_{len(docs):05d}",
                    "source": source,
                    "title": title,
                    "text": text,
                    "url": link,
                }
            )
            if len(docs) >= limit:
                break
        if len(docs) >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, docs)
    print(f"news_docs={len(docs)} source={source} -> {out_path}", flush=True)
    return len(docs)


def _sudachi_tokenizer():
    from sudachipy import Dictionary, SplitMode

    dict_obj = Dictionary(dict="full")
    return dict_obj.create(mode=SplitMode.B), SplitMode


def extract_from_docs(docs: Iterable[dict[str, Any]], max_ctx: int = 50) -> list[dict[str, Any]]:
    tok, _ = _sudachi_tokenizer()
    out: list[dict[str, Any]] = []
    for doc in docs:
        text = doc.get("text") or ""
        doc_id = doc.get("doc_id") or "doc"
        source = doc.get("source") or "unknown"
        if not text:
            continue
        # Sentence-ish split for sent_id only (tokenization is on full text offsets)
        # Rebuild char alignment via Sudachi on whole text.
        try:
            morphemes = tok.tokenize(text)
        except Exception as exc:
            print(f"sudachi_fail {doc_id}: {exc}", flush=True)
            continue
        # Track char cursor: Sudachi surfaces concatenate ≈ original for plain text
        # Use begin/end from morpheme if available
        sent_idx = 0
        for m in morphemes:
            surface = m.surface()
            pos = list(m.part_of_speech())
            reading_raw = m.reading_form() or ""
            try:
                begin = m.begin()
                end = m.end()
            except Exception:
                # Fallback: skip if no offsets
                continue
            if not _is_content_token(pos, surface):
                # Update sentence index on punctuation in non-content too
                if surface in ("。", "！", "？", "!", "?"):
                    sent_idx += 1
                continue
            reading = _hiragana_reading(reading_raw)
            gold = normalize_surface(surface)
            if not _qc_ok(reading, gold):
                continue
            ctx = clip_context_prev(text, begin, max_chars=max_ctx)
            out.append(
                {
                    "reading": reading,
                    "context_prev": ctx,
                    "gold": gold,
                    "source": source,
                    "doc_id": doc_id,
                    "sent_id": f"{doc_id}_s{sent_idx:04d}",
                    "char_begin": begin,
                    "char_end": end,
                    "pos": pos[:3],
                }
            )
            if surface in ("。", "！", "？"):
                sent_idx += 1
        print(f"extract {doc_id}: tokens_kept cumulative={len(out)}", flush=True)
    return out


def attach_mozc(
    extracted: list[dict[str, Any]],
    work_dir: Path,
    *,
    env_file: Path | None,
    max_candidates: int = 80,
    mode: str = "auto",
    force_rerun: bool = False,
) -> list[dict[str, Any]]:
    """Attach Mozc N-best via shared boundary helper (Mode A/B).

    Keys are unique hiragana/NFKC readings only (no context). Under WSL, Mode B
    launches the Windows PE with ``wslpath -w`` file args; on failure Mode A
    prints a pasteable PowerShell command.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Normalize readings on records so join keys match keys.txt / TSV.
    for r in extracted:
        r["reading"] = normalize_mozc_key(r.get("reading", ""))
    keys = unique_keys(r["reading"] for r in extracted)
    keys_path = (work_dir / "keys.txt").resolve()
    cand_path = (work_dir / "candidates.tsv").resolve()
    write_keys_txt(keys, keys_path, normalize=False)  # already normalized
    # Join-only: reuse existing candidates.tsv without needing MOZC_BATCH_EXE.
    if mode == "join" and cand_path.is_file() and cand_path.stat().st_size > 0:
        print(f"mozc_batch join: using existing {cand_path} keys={len(keys)}", flush=True)
        used = "a"
        max_c = max_candidates
    else:
        exe, engine, max_c = resolve_batch_config(
            env_file=env_file or Path("config/mozc_batch.env"),
            max_candidates=max_candidates,
        )
        max_c = max_candidates or max_c
        print(f"mozc_batch unique_keys={len(keys)} mode={mode} max={max_c}", flush=True)
        try:
            used = run_mozc_batch_boundary(
                exe,
                engine,
                keys_path,
                cand_path,
                max_c,
                mode=mode,  # type: ignore[arg-type]
                allow_existing_candidates=not force_rerun,
            )
            print(f"mozc_batch used_mode={used}", flush=True)
        except MozcBatchModeARequired as exc:
            print(str(exc), flush=True)
            raise SystemExit(2) from exc

    key_to_cands = load_candidates_map(cand_path)
    per_row = join_reading_candidates(extracted, key_to_cands, reading_field="reading")
    attached: list[dict[str, Any]] = []
    for r, cands in zip(extracted, per_row):
        dedup: list[str] = []
        seen_c: set[str] = set()
        for c in cands:
            if not c or c in seen_c:
                continue
            seen_c.add(c)
            dedup.append(c)
        rank = find_gold_rank(r["gold"], dedup)
        top1 = dedup[0] if dedup else ""
        hit1 = bool(top1) and (
            top1 == r["gold"] or normalize_surface(top1) == normalize_surface(r["gold"])
        )
        attached.append(
            {
                **r,
                "mozc_nbest": dedup,
                "gold_in_nbest": rank is not None,
                "mozc_top1": top1,
                "mozc_hit1": hit1,
                "gold_rank": rank,
            }
        )
    out_path = work_dir / "attached.jsonl"
    write_jsonl(out_path, attached)
    print(f"attached={len(attached)} -> {out_path}", flush=True)
    return attached


def build_reading_gold_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["reading"]][r["gold"]] += 1
    ambiguous_readings: dict[str, Any] = {}
    for reading, gold_c in counts.items():
        # Ambiguous: ≥2 golds each with freq ≥5
        strong = {g: n for g, n in gold_c.items() if n >= 5}
        if len(strong) >= 2:
            ambiguous_readings[reading] = {
                "golds": dict(gold_c),
                "strong_golds": strong,
                "reading_gold_count": len(strong),
            }
    return {
        "n_readings": len(counts),
        "n_ambiguous_readings": len(ambiguous_readings),
        "ambiguous": ambiguous_readings,
        "all_counts": {k: dict(v) for k, v in counts.items()},
    }


def annotate_ambiguous(rows: list[dict[str, Any]], reading_map: dict[str, Any]) -> list[dict[str, Any]]:
    amb = reading_map.get("ambiguous") or {}
    out = []
    for r in rows:
        info = amb.get(r["reading"])
        out.append(
            {
                **r,
                "ambiguous": bool(info),
                "reading_gold_count": int(info["reading_gold_count"]) if info else 1,
            }
        )
    return out


def build_context_sensitive_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Define context_sensitive readings from the full attached pool.

    Criteria (AND): total≥8, top_gold_share<0.70, ≥2 kanji golds (gold≠reading,
    each freq≥3), content POS only. Rows with empty clean_context are excluded
    from aggregation.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if not clean_context(r.get("context_prev") or ""):
            continue
        if not is_content_pos(r.get("pos") or []):
            continue
        counts[r["reading"]][r["gold"]] += 1

    cs_readings: dict[str, Any] = {}
    for reading, gold_c in counts.items():
        total = sum(gold_c.values())
        if total < 8:
            continue
        top_n = max(gold_c.values())
        top_share = top_n / total
        if top_share >= 0.70:
            continue
        kanji_golds = {
            g: n for g, n in gold_c.items() if has_kanji(g) and g != reading and n >= 3
        }
        if len(kanji_golds) < 2:
            continue
        cs_readings[reading] = {
            "golds": dict(gold_c),
            "kanji_golds": kanji_golds,
            "total": total,
            "top_gold_share": round(top_share, 6),
            "n_kanji_golds": len(kanji_golds),
        }
    return {
        "n_readings_scored": len(counts),
        "n_context_sensitive_readings": len(cs_readings),
        "context_sensitive": cs_readings,
    }


def _recompute_mozc_fields(row: dict[str, Any]) -> dict[str, Any]:
    cands = sanitize_nbest(list(row.get("mozc_nbest") or []))
    rank = find_gold_rank(row["gold"], cands)
    top1 = cands[0] if cands else ""
    hit1 = bool(top1) and (
        top1 == row["gold"] or normalize_surface(top1) == normalize_surface(row["gold"])
    )
    out = dict(row)
    out["mozc_nbest"] = cands
    out["gold_in_nbest"] = rank is not None
    out["gold_rank"] = rank
    out["mozc_top1"] = top1
    out["mozc_hit1"] = hit1
    return out


def annotate_context_sensitive(
    rows: list[dict[str, Any]],
    cs_map: dict[str, Any],
    *,
    apply_clean: bool = True,
    apply_nbest_hygiene: bool = True,
) -> list[dict[str, Any]]:
    """Attach context_sensitive / top_gold_share / n_kanji_golds; clean context."""
    cs = cs_map.get("context_sensitive") or {}
    out: list[dict[str, Any]] = []
    for r in rows:
        info = cs.get(r["reading"])
        cleaned = clean_context(r.get("context_prev") or "") if apply_clean else (r.get("context_prev") or "")
        gold = r.get("gold") or ""
        reading = r.get("reading") or ""
        row_cs = bool(
            info
            and is_content_pos(r.get("pos") or [])
            and has_kanji(gold)
            and gold != reading
            and bool(cleaned)
        )
        item = {
            **r,
            "context_prev": cleaned if apply_clean else r.get("context_prev") or "",
            "context_sensitive": row_cs,
            "top_gold_share": float(info["top_gold_share"]) if info else 1.0,
            "n_kanji_golds": int(info["n_kanji_golds"]) if info else 0,
        }
        if apply_nbest_hygiene:
            item = _recompute_mozc_fields(item)
        out.append(item)
    return out


def _row_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (
        r.get("doc_id"),
        r.get("sent_id"),
        r.get("char_begin"),
        r.get("char_end"),
        r.get("reading"),
        r.get("gold"),
    )


def make_corrupted_anchor(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Light context corruption for a small robustness slice (anchor, not CS)."""
    r = dict(row)
    ctx = r.get("context_prev") or ""
    if len(ctx) >= 4:
        cut = max(1, len(ctx) // 2)
        if rng.random() < 0.5:
            r["context_prev"] = ctx[:cut]
        else:
            r["context_prev"] = "".join(ch for i, ch in enumerate(ctx) if i % 2 == 0)
    r["context_sensitive"] = False
    r["corrupted"] = True
    return r


def _pick_anchors(
    anchors: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    prefer_hit1: bool,
) -> list[dict[str, Any]]:
    if prefer_hit1:
        hit = [r for r in anchors if r.get("mozc_hit1")]
        miss = [r for r in anchors if not r.get("mozc_hit1")]
        rng.shuffle(hit)
        rng.shuffle(miss)
        pool = hit + miss
    else:
        pool = list(anchors)
        rng.shuffle(pool)
    return pool[:n]


def assemble_train_v2(
    train_rows: list[dict[str, Any]],
    *,
    cs_frac: float,
    seed: int,
    min_cs: int,
    corrupt_frac: float,
    reserve_cs_for_seen: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[Any, ...]]]:
    """Oversample all train-doc CS (minus seen reserve); fill anchors to 30–40% CS.

    Returns (train, eval_seen_cs_reserve, reserved_keys).
    """
    rng = random.Random(seed)
    usable = [r for r in train_rows if r.get("gold_in_nbest")]
    cs = [r for r in usable if r.get("context_sensitive")]
    anchors = [r for r in usable if not r.get("context_sensitive")]
    rng.shuffle(cs)

    n_reserve = min(reserve_cs_for_seen, max(0, len(cs) - min_cs))
    # Prefer leaving ≥min_cs for train
    if len(cs) - reserve_cs_for_seen >= min_cs:
        n_reserve = reserve_cs_for_seen
    else:
        n_reserve = max(0, len(cs) - min_cs)

    seen_cs = cs[:n_reserve]
    train_cs = cs[n_reserve:]
    reserved_keys = {_row_key(r) for r in seen_cs}

    # Size so CS lands in [30%, 40%]
    n_cs = len(train_cs)
    if n_cs == 0:
        train: list[dict[str, Any]] = []
    else:
        n_lo = int(math.ceil(n_cs / 0.40))  # CS = 40%
        n_hi = int(math.floor(n_cs / 0.30))  # CS = 30%
        n_target = int(round(n_cs / cs_frac))
        n_total = min(max(n_target, n_lo), n_hi)
        n_anchor = max(0, n_total - n_cs)
        take_anchor = _pick_anchors(anchors, n_anchor, rng, prefer_hit1=True)
        take_anchor = [r for r in take_anchor if _row_key(r) not in reserved_keys]
        if len(take_anchor) < n_anchor:
            taken = {_row_key(r) for r in take_anchor}

            def _ok_anchor(r: dict[str, Any]) -> bool:
                k = _row_key(r)
                return k not in reserved_keys and k not in taken

            extra = [r for r in anchors if _ok_anchor(r)]
            rng.shuffle(extra)
            take_anchor.extend(extra[: n_anchor - len(take_anchor)])
        train = list(train_cs) + take_anchor[:n_anchor]

        # Small corrupted slice from unused anchors
        n_corrupt = int(round(len(train) * corrupt_frac))
        used = {_row_key(r) for r in train} | reserved_keys
        corrupt_src = [r for r in anchors if _row_key(r) not in used and (r.get("context_prev") or "")]
        rng.shuffle(corrupt_src)
        for r in corrupt_src[:n_corrupt]:
            train.append(make_corrupted_anchor(r, rng))

    rng.shuffle(train)
    return train, seen_cs, reserved_keys


def assemble_eval_v2(
    pool_rows: list[dict[str, Any]],
    *,
    n: int,
    cs_frac: float,
    min_cs: int,
    seed: int,
    exclude_keys: set[tuple[Any, ...]] | None = None,
    prefer_all_cs: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    excl = exclude_keys or set()
    usable = [r for r in pool_rows if r.get("gold_in_nbest") and _row_key(r) not in excl]
    cs = [r for r in usable if r.get("context_sensitive")]
    anchors = [r for r in usable if not r.get("context_sensitive")]
    rng.shuffle(cs)
    n_cs = min(len(cs), max(min_cs, int(round(n * cs_frac))))
    if prefer_all_cs:
        n_cs = len(cs)
        # bump n if needed to keep CS ≤40%
        if n_cs > 0 and n_cs / max(n, 1) > 0.40:
            n = int(math.ceil(n_cs / 0.35))
    take_cs = cs[:n_cs]
    n_anchor = max(0, n - len(take_cs))
    take_anchor = _pick_anchors(anchors, n_anchor, rng, prefer_hit1=True)
    take = take_cs + take_anchor
    if len(take) < n:
        leftover = cs[n_cs:] + [r for r in anchors if r not in take_anchor]
        rng.shuffle(leftover)
        take.extend(leftover[: n - len(take)])
    rng.shuffle(take)
    return take[:n]


def _subset_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rs)
    cs_rows = [r for r in rs if r.get("context_sensitive")]
    amb = sum(1 for r in rs if r.get("ambiguous"))
    shares = sorted(float(r.get("top_gold_share") or 1.0) for r in cs_rows)
    median_share = shares[len(shares) // 2] if shares else None
    ctx_nonempty = sum(1 for r in rs if (r.get("context_prev") or "").strip())
    nl_residue = sum(1 for r in rs if "\n" in (r.get("context_prev") or "") or "\r" in (r.get("context_prev") or ""))
    markup_residue = sum(
        1
        for r in rs
        if "==" in (r.get("context_prev") or "")
        or "[edit]" in (r.get("context_prev") or "").lower()
        or "[編集]" in (r.get("context_prev") or "")
    )
    gold_eq = sum(1 for r in cs_rows if r.get("gold") == r.get("reading"))
    return {
        "n": n,
        "context_sensitive": len(cs_rows),
        "context_sensitive_frac": round(len(cs_rows) / max(1, n), 4),
        "context_sensitive_distinct_readings": len({r["reading"] for r in cs_rows}),
        "ambiguous": amb,
        "ambiguous_frac": round(amb / max(1, n), 4),
        "top_gold_share_median_cs": round(median_share, 4) if median_share is not None else None,
        "context_nonempty_frac": round(ctx_nonempty / max(1, n), 4),
        "context_nonempty_frac_cs": round(
            sum(1 for r in cs_rows if (r.get("context_prev") or "").strip()) / max(1, len(cs_rows)), 4
        )
        if cs_rows
        else None,
        "newline_residue_frac": round(nl_residue / max(1, n), 4),
        "markup_residue_frac": round(markup_residue / max(1, n), 4),
        "gold_eq_reading_frac_cs": round(gold_eq / max(1, len(cs_rows)), 4) if cs_rows else None,
        "gold_in_nbest": sum(1 for r in rs if r.get("gold_in_nbest")),
        "gold_in_nbest_frac": round(sum(1 for r in rs if r.get("gold_in_nbest")) / max(1, n), 4),
        "mozc_hit1": round(sum(1 for r in rs if r.get("mozc_hit1")) / max(1, n), 4),
        "corrupted": sum(1 for r in rs if r.get("corrupted")),
    }


def command_assemble_v2(args: argparse.Namespace) -> int:
    """Rebuild train/eval with context_sensitive primary flag (*_v2.jsonl)."""
    attached_path = Path(args.attached)
    rows = list(read_jsonl(attached_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep legacy ambiguous for comparison
    if not rows or "ambiguous" not in rows[0]:
        mp = build_reading_gold_map(rows)
        compact = {
            "n_readings": mp["n_readings"],
            "n_ambiguous_readings": mp["n_ambiguous_readings"],
            "ambiguous": mp["ambiguous"],
        }
        rows = annotate_ambiguous(rows, compact)

    if getattr(args, "cs_map_from", None):
        cs_src_path = Path(args.cs_map_from)
    else:
        cs_src_path = attached_path
    if cs_src_path.resolve() == attached_path.resolve():
        cs_src_rows = rows
    else:
        cs_src_rows = list(read_jsonl(cs_src_path))
        if cs_src_rows and "ambiguous" not in cs_src_rows[0]:
            mp0 = build_reading_gold_map(cs_src_rows)
            cs_src_rows = annotate_ambiguous(
                cs_src_rows,
                {
                    "n_readings": mp0["n_readings"],
                    "n_ambiguous_readings": mp0["n_ambiguous_readings"],
                    "ambiguous": mp0["ambiguous"],
                },
            )
    cs_map = build_context_sensitive_map(cs_src_rows)
    (out_dir / "context_sensitive_map.json").write_text(
        json.dumps(
            {
                "n_readings_scored": cs_map["n_readings_scored"],
                "n_context_sensitive_readings": cs_map["n_context_sensitive_readings"],
                "context_sensitive": {
                    k: {
                        "total": v["total"],
                        "top_gold_share": v["top_gold_share"],
                        "n_kanji_golds": v["n_kanji_golds"],
                        "kanji_golds": v["kanji_golds"],
                    }
                    for k, v in cs_map["context_sensitive"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = annotate_context_sensitive(
        rows,
        cs_map,
        apply_clean=True,
        apply_nbest_hygiene=True,
    )

    split = doc_split(rows, seed=args.seed, train_frac=args.train_frac)
    n_seen_cs_target = max(args.min_eval_cs, int(round(args.eval_n * args.cs_frac)))
    train, seen_cs_reserve, reserved_keys = assemble_train_v2(
        split["train"],
        cs_frac=args.cs_frac,
        seed=args.seed,
        min_cs=args.min_train_cs,
        corrupt_frac=args.corrupt_frac,
        reserve_cs_for_seen=n_seen_cs_target,
    )
    # eval_seen: reserved CS + anchors from train docs (no train row overlap)
    rng_seen = random.Random(args.seed + 1)
    seen_cs = list(seen_cs_reserve)
    rng_seen.shuffle(seen_cs)
    seen_cs = seen_cs[:n_seen_cs_target]
    # Any unused reserved CS go back to train (maximize train CS).
    unused_reserve = seen_cs_reserve[n_seen_cs_target:]
    if unused_reserve:
        train.extend(unused_reserve)
        reserved_keys = {_row_key(r) for r in seen_cs}
    train_keys = {_row_key(r) for r in train}
    seen_anchors = [
        r
        for r in split["seen_pool"]
        if r.get("gold_in_nbest")
        and not r.get("context_sensitive")
        and _row_key(r) not in reserved_keys
        and _row_key(r) not in train_keys
    ]
    n_seen_anchor = max(0, args.eval_n - len(seen_cs))
    take_seen_anchor = _pick_anchors(seen_anchors, n_seen_anchor, rng_seen, prefer_hit1=True)
    eval_seen = seen_cs + take_seen_anchor
    rng_seen.shuffle(eval_seen)
    eval_seen = eval_seen[: args.eval_n]

    eval_unseen = assemble_eval_v2(
        split["unseen"],
        n=args.eval_n,
        cs_frac=args.cs_frac,
        min_cs=args.min_eval_cs,
        seed=args.seed + 2,
    )
    fresh_cs_avail = sum(
        1 for r in split["fresh"] if r.get("context_sensitive") and r.get("gold_in_nbest")
    )
    eval_fresh = assemble_eval_v2(
        split["fresh"],
        n=args.eval_fresh_n,
        cs_frac=args.cs_frac,
        min_cs=args.min_eval_cs,
        seed=args.seed + 3,
        # Take all CS only when the pool is too small to hit min_eval_cs.
        prefer_all_cs=fresh_cs_avail < args.min_eval_cs,
    )

    write_jsonl(out_dir / "train_v2.jsonl", train)
    write_jsonl(out_dir / "eval_seen_v2.jsonl", eval_seen)
    write_jsonl(out_dir / "eval_unseen_v2.jsonl", eval_unseen)
    write_jsonl(out_dir / "eval_fresh_v2.jsonl", eval_fresh)
    write_audit_sample(train + eval_unseen, out_dir / "audit_sample_v2_300.json", n=300, seed=args.seed)

    # Leak checks
    train_docs = {r["doc_id"] for r in train}
    fresh_docs = {r["doc_id"] for r in eval_fresh}
    unseen_docs = {r["doc_id"] for r in eval_unseen}
    leak_fresh_train = sorted(train_docs & fresh_docs)
    leak_unseen_train = sorted(train_docs & unseen_docs)
    train_keys = {_row_key(r) for r in train}
    seen_overlap = sum(1 for r in eval_seen if _row_key(r) in train_keys)

    summary = {
        "scale": "v2_context_sensitive",
        "ce_base_note": _CE_BASE_NOTE,
        "split_meta": split["meta"],
        "train_frac": args.train_frac,
        "cs_frac_target": args.cs_frac,
        "cs_map_from": str(cs_src_path),
        "n_context_sensitive_readings": cs_map["n_context_sensitive_readings"],
        "train": _subset_stats(train),
        "eval_seen": _subset_stats(eval_seen),
        "eval_unseen": _subset_stats(eval_unseen),
        "eval_fresh": _subset_stats(eval_fresh),
        "leak_checks": {
            "fresh_docs_in_train": leak_fresh_train,
            "unseen_docs_in_train": leak_unseen_train,
            "eval_seen_row_overlap_with_train": seen_overlap,
        },
        "success_criteria": {
            "train_cs_ge_15000": _subset_stats(train)["context_sensitive"] >= 15000,
            "train_cs_distinct_ge_200": _subset_stats(train)["context_sensitive_distinct_readings"]
            >= 200,
            "eval_seen_cs_ge_500": _subset_stats(eval_seen)["context_sensitive"] >= 500,
            "eval_unseen_cs_ge_500": _subset_stats(eval_unseen)["context_sensitive"] >= 500,
            "eval_fresh_cs_ge_500": _subset_stats(eval_fresh)["context_sensitive"] >= 500,
            "median_top_gold_share_cs_lt_0_6": (
                (_subset_stats(train)["top_gold_share_median_cs"] or 1.0) < 0.6
            ),
            "newline_residue_near_0": _subset_stats(train)["newline_residue_frac"] < 0.01,
            "cs_gold_eq_reading_0": (_subset_stats(train)["gold_eq_reading_frac_cs"] or 0) == 0,
            "no_fresh_in_train": len(leak_fresh_train) == 0,
        },
    }
    crit = summary["success_criteria"]
    summary["go_train_track_ab"] = all(
        [
            crit["train_cs_ge_15000"],
            crit["train_cs_distinct_ge_200"],
            crit["eval_seen_cs_ge_500"],
            crit["eval_unseen_cs_ge_500"],
            crit["median_top_gold_share_cs_lt_0_6"],
            crit["newline_residue_near_0"],
            crit["cs_gold_eq_reading_0"],
            crit["no_fresh_in_train"],
        ]
    )
    # Fresh CS≥500 is required for full go; surface separately if short.
    summary["go_train_track_ab_full"] = bool(summary["go_train_track_ab"] and crit["eval_fresh_cs_ge_500"])
    (out_dir / "assemble_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# (removed stray noop)


def doc_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float = 0.8,
) -> dict[str, list[dict[str, Any]]]:
    """Document-level split into train / seen / unseen / fresh."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    # Separate fresh news docs
    fresh_docs = [d for d, rs in by_doc.items() if (rs[0].get("source") or "").startswith("news_fresh")]
    pool_docs = [d for d in by_doc if d not in set(fresh_docs)]
    rng = random.Random(seed)
    rng.shuffle(pool_docs)
    n_train = max(1, int(len(pool_docs) * train_frac))
    train_docs = set(pool_docs[:n_train])
    hold_docs = pool_docs[n_train:]
    # unseen = half of hold docs; remaining hold docs contribute to seen sampling from train
    rng.shuffle(hold_docs)
    mid = max(1, len(hold_docs) // 2) if hold_docs else 0
    unseen_docs = set(hold_docs[:mid])
    # seen samples come from train_docs (different instances)
    train_rows = [r for d in train_docs for r in by_doc[d]]
    unseen_rows = [r for d in unseen_docs for r in by_doc[d]]
    seen_pool = list(train_rows)
    fresh_rows = [r for d in fresh_docs for r in by_doc[d]]
    return {
        "train": train_rows,
        "seen_pool": seen_pool,
        "unseen": unseen_rows,
        "fresh": fresh_rows,
        "meta": {
            "n_train_docs": len(train_docs),
            "n_unseen_docs": len(unseen_docs),
            "n_fresh_docs": len(fresh_docs),
            "n_pool_docs": len(pool_docs),
        },
    }


def sample_with_ambiguous_ratio(
    rows: list[dict[str, Any]],
    *,
    n: int,
    amb_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    # Prefer gold_in_nbest for CE trainability
    usable = [r for r in rows if r.get("gold_in_nbest")]
    amb = [r for r in usable if r.get("ambiguous")]
    non = [r for r in usable if not r.get("ambiguous")]
    rng.shuffle(amb)
    rng.shuffle(non)
    n_amb = int(n * amb_frac)
    n_non = n - n_amb
    take = amb[:n_amb] + non[:n_non]
    if len(take) < n:
        leftover = amb[n_amb:] + non[n_non:]
        rng.shuffle(leftover)
        take.extend(leftover[: n - len(take)])
    rng.shuffle(take)
    return take[:n]


def write_audit_sample(rows: list[dict[str, Any]], out_path: Path, n: int = 300, seed: int = 0) -> None:
    rng = random.Random(seed)
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[:n]
    slim = [
        {
            "reading": r["reading"],
            "context_prev": r.get("context_prev") or "",
            "gold": r["gold"],
            "mozc_top1": r.get("mozc_top1") or "",
            "gold_in_nbest": r.get("gold_in_nbest"),
            "ambiguous": r.get("ambiguous"),
            "source": r.get("source"),
            "doc_id": r.get("doc_id"),
            "audit_ok": None,
            "audit_note": "",
        }
        for r in sample
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also TSV for easier human marking
    tsv = out_path.with_suffix(".tsv")
    lines = ["idx\treading\tcontext_prev\tgold\tmozc_top1\tgold_in_nbest\tambiguous\tsource\taudit_ok\taudit_note"]
    for i, r in enumerate(slim):
        ctx = (r["context_prev"] or "").replace("\t", " ").replace("\n", " ")
        lines.append(
            f"{i}\t{r['reading']}\t{ctx}\t{r['gold']}\t{r['mozc_top1']}\t{r['gold_in_nbest']}\t{r['ambiguous']}\t{r['source']}\t\t"
        )
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_fetch_wiki(args: argparse.Namespace) -> int:
    n = fetch_wiki_docs(args.n, Path(args.out), seed=args.seed)
    print(f"wrote {n} -> {args.out}")
    return 0


def command_fetch_news(args: argparse.Namespace) -> int:
    n = fetch_news_rss(Path(args.out), fresh=bool(args.fresh), limit=args.n)
    print(f"wrote {n} -> {args.out}")
    return 0


def command_extract(args: argparse.Namespace) -> int:
    docs = list(read_jsonl(Path(args.docs)))
    rows = extract_from_docs(docs, max_ctx=args.max_ctx)
    write_jsonl(Path(args.out), rows)
    print(f"extracted={len(rows)} -> {args.out}")
    return 0


def command_mozc_attach(args: argparse.Namespace) -> int:
    extracted = list(read_jsonl(Path(args.extracted)))
    attach_mozc(
        extracted,
        Path(args.work_dir),
        env_file=Path(args.env_file) if args.env_file else None,
        max_candidates=args.max_candidates,
        mode=getattr(args, "mode", "auto") or "auto",
        force_rerun=bool(getattr(args, "force_rerun", False)),
    )
    return 0


def command_ambiguous(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.attached)))
    mp = build_reading_gold_map(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Store compact map (ambiguous + summary); full all_counts can be huge
    compact = {
        "n_readings": mp["n_readings"],
        "n_ambiguous_readings": mp["n_ambiguous_readings"],
        "ambiguous": mp["ambiguous"],
    }
    out.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    ann = annotate_ambiguous(rows, compact)
    write_jsonl(Path(args.annotated_out), ann)
    print(
        f"ambiguous_readings={compact['n_ambiguous_readings']} annotated={len(ann)} -> {args.annotated_out}"
    )
    return 0


def command_assemble_small(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.attached)))
    # Ensure ambiguous fields
    if not rows or "ambiguous" not in rows[0]:
        mp = build_reading_gold_map(rows)
        compact = {
            "n_readings": mp["n_readings"],
            "n_ambiguous_readings": mp["n_ambiguous_readings"],
            "ambiguous": mp["ambiguous"],
        }
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / "reading_gold_map.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows = annotate_ambiguous(rows, compact)
    else:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    split = doc_split(rows, seed=args.seed, train_frac=args.train_frac)
    train = sample_with_ambiguous_ratio(
        split["train"], n=args.train_n, amb_frac=args.amb_frac, seed=args.seed
    )
    eval_seen = sample_with_ambiguous_ratio(
        split["seen_pool"], n=args.eval_n, amb_frac=args.amb_frac, seed=args.seed + 1
    )
    eval_unseen = sample_with_ambiguous_ratio(
        split["unseen"], n=args.eval_n, amb_frac=args.amb_frac, seed=args.seed + 2
    )
    # fresh may be small — take all usable then pad from unseen if needed
    fresh_usable = [r for r in split["fresh"] if r.get("gold_in_nbest")]
    if len(fresh_usable) >= args.eval_fresh_n:
        eval_fresh = sample_with_ambiguous_ratio(
            split["fresh"], n=args.eval_fresh_n, amb_frac=args.amb_frac, seed=args.seed + 3
        )
    else:
        eval_fresh = fresh_usable
        need = args.eval_fresh_n - len(eval_fresh)
        if need > 0:
            # Mark padded rows for transparency
            pad = sample_with_ambiguous_ratio(
                split["unseen"], n=need, amb_frac=args.amb_frac, seed=args.seed + 4
            )
            for r in pad:
                r = dict(r)
                r["fresh_pad_from_unseen"] = True
                eval_fresh.append(r)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval_seen.jsonl", eval_seen)
    write_jsonl(out_dir / "eval_unseen.jsonl", eval_unseen)
    write_jsonl(out_dir / "eval_fresh.jsonl", eval_fresh)
    write_audit_sample(train + eval_unseen, out_dir / "audit_sample_300.json", n=300, seed=args.seed)

    def amb_stats(name: str, rs: list[dict[str, Any]]) -> dict[str, Any]:
        amb = sum(1 for r in rs if r.get("ambiguous"))
        gin = sum(1 for r in rs if r.get("gold_in_nbest"))
        return {
            "n": len(rs),
            "ambiguous": amb,
            "ambiguous_frac": round(amb / max(1, len(rs)), 4),
            "gold_in_nbest": gin,
            "gold_in_nbest_frac": round(gin / max(1, len(rs)), 4),
            "mozc_hit1": round(sum(1 for r in rs if r.get("mozc_hit1")) / max(1, len(rs)), 4),
        }

    summary = {
        "scale": "small",
        "split_meta": split["meta"],
        "train": amb_stats("train", train),
        "eval_seen": amb_stats("seen", eval_seen),
        "eval_unseen": amb_stats("unseen", eval_unseen),
        "eval_fresh": amb_stats("fresh", eval_fresh),
        "amb_frac_target": args.amb_frac,
        "note": _CE_BASE_NOTE,
    }
    (out_dir / "assemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_smoke_one_doc(args: argparse.Namespace) -> int:
    """End-to-end on a single synthetic/wiki doc for wiring check."""
    text = (
        "朝日新聞の記者が駅に向かった。"
        "ホームでは汽車が到着するのを待っていた。"
        "貴社の製品についても後日取材する予定だ。"
    )
    docs = [{"doc_id": "smoke_001", "source": "wiki", "title": "smoke", "text": text}]
    rows = extract_from_docs(docs)
    print("extracted", len(rows))
    for r in rows[:12]:
        print(json.dumps({k: r[k] for k in ("reading", "context_prev", "gold")}, ensure_ascii=False))
    # context_prev identity check
    from tools.rerank.context_clip import clip_context_prev as clip2

    assert clip_context_prev(text, 0) == ""
    # Find 記者 offset
    idx = text.index("記者")
    ctx = clip_context_prev(text, idx, 50)
    assert "朝日新聞の" in ctx or ctx.endswith("の") or "新聞" in ctx
    assert clip2(text, idx, 50) == ctx
    print("smoke_ok context_clip shared")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contextual reranker dataset builder")
    sub = p.add_subparsers(dest="command", required=True)

    sm = sub.add_parser("smoke", help="one-doc extract + context_clip check")
    sm.set_defaults(func=command_smoke_one_doc)

    fw = sub.add_parser("fetch-wiki")
    fw.add_argument("--out", default="data/rerank_ctx/raw/wiki_docs.jsonl")
    fw.add_argument("--n", type=int, default=400)
    fw.add_argument("--seed", type=int, default=42)
    fw.set_defaults(func=command_fetch_wiki)

    fn = sub.add_parser("fetch-news")
    fn.add_argument("--out", default="data/rerank_ctx/raw/news_fresh.jsonl")
    fn.add_argument("--n", type=int, default=80)
    fn.add_argument("--fresh", action="store_true", default=True)
    fn.add_argument("--old", action="store_true", help="label as news_old")
    fn.set_defaults(func=command_fetch_news)

    ex = sub.add_parser("extract")
    ex.add_argument("--docs", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--max-ctx", type=int, default=50)
    ex.set_defaults(func=command_extract)

    mz = sub.add_parser("mozc-attach")
    mz.add_argument("--extracted", required=True)
    mz.add_argument("--work-dir", default="data/rerank_ctx/work/mozc")
    mz.add_argument("--env-file", default="config/mozc_batch.env")
    mz.add_argument("--max-candidates", type=int, default=80)
    mz.add_argument(
        "--mode",
        choices=("auto", "a", "b", "join"),
        default="auto",
        help="auto=Mode B under WSL (fallback A); a=emit PowerShell; b=interop; join=existing TSV",
    )
    mz.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing candidates.tsv and re-run mozc_batch",
    )
    mz.set_defaults(func=command_mozc_attach)

    am = sub.add_parser("ambiguous")
    am.add_argument("--attached", required=True)
    am.add_argument("--out", default="data/rerank_ctx/reading_gold_map.json")
    am.add_argument("--annotated-out", default="data/rerank_ctx/work/annotated.jsonl")
    am.set_defaults(func=command_ambiguous)

    asm = sub.add_parser("assemble-small", help="small-scale train 20k / eval 1k splits")
    asm.add_argument("--attached", required=True)
    asm.add_argument("--out-dir", default="data/rerank_ctx")
    asm.add_argument("--train-n", type=int, default=20000)
    asm.add_argument("--eval-n", type=int, default=1000)
    asm.add_argument("--eval-fresh-n", type=int, default=1000)
    asm.add_argument("--amb-frac", type=float, default=0.35)
    asm.add_argument("--train-frac", type=float, default=0.8)
    asm.add_argument("--seed", type=int, default=42)
    asm.set_defaults(func=command_assemble_small)

    as2 = sub.add_parser(
        "assemble-v2",
        help="context_sensitive rebuild → train_v2 / eval_*_v2 (no Mozc rerun)",
    )
    as2.add_argument("--attached", required=True)
    as2.add_argument("--out-dir", default="data/rerank_ctx")
    as2.add_argument("--eval-n", type=int, default=1500)
    as2.add_argument("--eval-fresh-n", type=int, default=1500)
    as2.add_argument("--cs-frac", type=float, default=0.35)
    as2.add_argument(
        "--train-frac",
        type=float,
        default=0.92,
        help="doc-level train frac (higher to hit CS≥15k after seen reserve)",
    )
    as2.add_argument("--min-train-cs", type=int, default=15000)
    as2.add_argument("--min-eval-cs", type=int, default=500)
    as2.add_argument("--corrupt-frac", type=float, default=0.02)
    as2.add_argument("--seed", type=int, default=42)
    as2.add_argument(
        "--cs-map-from",
        default="",
        help="JSONL used only to DEFINE context_sensitive readings (freeze original pool)",
    )
    as2.set_defaults(func=command_assemble_v2)
    return p


def main(argv: list[str] | None = None) -> int:
    # fix news --old
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "old", False):
        args.fresh = False
        if "news_fresh" in (args.out or ""):
            args.out = args.out.replace("news_fresh", "news_old")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
