from __future__ import annotations

import re
from datetime import UTC, datetime

from .normalize import normalize_reading, normalize_surface
from .records import Provenance, TermRecord

EXPLICIT_RUBY = re.compile(r"｜(?P<surface>[^《\n]+)《(?P<reading>[^》\n]+)》")
IMPLICIT_RUBY = re.compile(r"(?P<surface>[一-鿿々〆ヶ]{1,20})《(?P<reading>[^》\n]+)》")


def ruby_records(text: str, source_id: str, source_url: str, source_version: str = "", retrieved_at: str | None = None) -> list[TermRecord]:
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    provenance = Provenance(
        source_id=source_id,
        source_url=source_url,
        license_id="per_work_public_domain_or_explicit_license",
        retrieved_at=timestamp,
        source_version=source_version,
    )
    records: dict[tuple[str, str], TermRecord] = {}
    for pattern in (EXPLICIT_RUBY, IMPLICIT_RUBY):
        for match in pattern.finditer(text):
            surface = normalize_surface(match.group("surface"))
            reading = normalize_reading(match.group("reading"))
            if not surface or not reading:
                continue
            records[(surface, reading)] = TermRecord(
                surface=surface,
                reading=reading,
                category="literary_ruby",
                provenance=provenance,
                reading_source="ruby",
                reading_confidence="ruby",
            )
    return list(records.values())
