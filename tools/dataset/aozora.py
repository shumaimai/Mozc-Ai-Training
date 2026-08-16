from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.request import urlopen

from .normalize import normalize_reading, normalize_surface
from .records import Provenance, TermRecord

EXPLICIT_RUBY = re.compile(r"｜(?P<surface>[^《\n]+)《(?P<reading>[^》\n]+)》")
IMPLICIT_RUBY = re.compile(r"(?P<surface>[一-鿿々〆ヶ]{1,20})《(?P<reading>[^》\n]+)》")

# GitHub mirror of Aozora Bunko. The extended author/work index carries the
# per-work and per-person copyright flags used to keep only public-domain texts.
INDEX_URL = "https://github.com/aozorabunko/aozorabunko/raw/master/index_pages/list_person_all_extended_utf8.zip"

# A row of ASCII hyphens delimits the header notes block that explains the ruby
# markup itself; that block must be dropped before ruby extraction.
DELIMITER = re.compile(r"^-{10,}$")


def _read_single_member(data: bytes, suffix: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"expected one {suffix} member, found {names}")
        return archive.read(names[0])


def download_index(url: str = INDEX_URL) -> list[dict[str, str]]:
    payload = urlopen(url, timeout=120).read()
    text = _read_single_member(payload, ".csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def public_domain_works(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    works: list[dict[str, str]] = []
    for row in rows:
        if row.get("作品著作権フラグ") != "なし" or row.get("人物著作権フラグ") != "なし":
            continue
        text_url = (row.get("テキストファイルURL") or "").strip()
        if not text_url.lower().endswith(".zip"):
            continue
        works.append(
            {
                "work_id": row.get("作品ID", ""),
                "title": row.get("作品名", ""),
                "title_reading": row.get("作品名読み", ""),
                "author_id": row.get("人物ID", ""),
                "author": (row.get("姓", "") + row.get("名", "")).strip(),
                "kana_type": row.get("文字遣い種別", ""),
                "text_url": text_url,
                "encoding": row.get("テキストファイル符号化方式", ""),
                "card_url": row.get("図書カードURL", ""),
                "release_date": row.get("公開日", ""),
                "base_book": row.get("底本名1", ""),
            }
        )
    return works


def strip_boilerplate(text: str) -> str:
    lines = text.splitlines()
    delimiters = [index for index, line in enumerate(lines) if DELIMITER.match(line.strip())]
    body = lines[delimiters[1] + 1 :] if len(delimiters) >= 2 else lines
    cleaned: list[str] = []
    for line in body:
        if line.startswith("底本："):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def download_work_text(url: str, encoding: str = "ShiftJIS") -> str:
    payload = urlopen(url, timeout=120).read()
    raw = _read_single_member(payload, ".txt")
    codec = "shift_jis" if encoding.lower().replace("_", "") in {"shiftjis", "sjis"} else encoding
    try:
        text = raw.decode(codec)
    except (UnicodeDecodeError, LookupError):
        text = raw.decode("cp932")
    return strip_boilerplate(text)


def ruby_records(
    text: str,
    source_id: str,
    source_url: str,
    source_version: str = "",
    retrieved_at: str | None = None,
    metadata: dict[str, object] | None = None,
    category: str = "literary_ruby",
    capture_context: bool = False,
) -> list[TermRecord]:
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    provenance = Provenance(
        source_id=source_id,
        source_url=source_url,
        license_id="per_work_public_domain_or_explicit_license",
        retrieved_at=timestamp,
        source_version=source_version,
    )
    records: dict[tuple[str, str], TermRecord] = {}
    for line in text.splitlines():
        for pattern in (EXPLICIT_RUBY, IMPLICIT_RUBY):
            for match in pattern.finditer(line):
                surface = normalize_surface(match.group("surface"))
                reading = normalize_reading(match.group("reading"))
                if not surface or not reading:
                    continue
                key = (surface, reading)
                if key in records:
                    continue
                entry_metadata = dict(metadata or {})
                if capture_context:
                    entry_metadata["context"] = line.strip()
                records[key] = TermRecord(
                    surface=surface,
                    reading=reading,
                    category=category,
                    provenance=provenance,
                    reading_source="ruby",
                    reading_confidence="ruby",
                    metadata=entry_metadata,
                )
    return list(records.values())


def work_records(work: dict[str, str], capture_context: bool = True) -> list[TermRecord]:
    text = download_work_text(work["text_url"], work.get("encoding", "ShiftJIS"))
    metadata = {
        "work_id": work.get("work_id", ""),
        "title": work.get("title", ""),
        "author": work.get("author", ""),
        "card_url": work.get("card_url", ""),
        "kana_type": work.get("kana_type", ""),
    }
    return ruby_records(
        text,
        source_id="aozora_public_domain",
        source_url=work["text_url"],
        source_version=work.get("release_date", ""),
        metadata=metadata,
        capture_context=capture_context,
    )
