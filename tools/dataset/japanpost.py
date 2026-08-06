from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from .normalize import is_valid_reading, normalize_reading, normalize_surface
from .records import Provenance, TermRecord

DEFAULT_URL = "https://www.post.japanpost.jp/service/search/zipcode/download/kogaki/zip/ken_all.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def records_from_zip(path: Path, source_url: str, retrieved_at: str | None = None) -> list[TermRecord]:
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    provenance = Provenance(
        source_id="japanpost_zipcode",
        source_url=source_url,
        license_id="JapanPost-terms-review-required",
        retrieved_at=timestamp,
    )
    records: dict[tuple[str, str], TermRecord] = {}
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV file in {path}")
        raw_content = archive.read(csv_names[0])
    try:
        content = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw_content.decode("cp932")
    for row in csv.reader(io.StringIO(content)):
        if len(row) < 9:
            continue
        prefecture_kana, city_kana, town_kana = row[3:6]
        prefecture, city, town = row[6:9]
        metadata = {"postal_code": row[2], "prefecture": prefecture, "city": city, "town": town}
        town_surface = normalize_surface(town)
        town_is_usable = (
            town_surface not in {"以下に掲載がない場合", "一円", "番地がくる場合"}
            and not any(marker in town_surface for marker in ("(", ")", "~", "（", "）", "〜"))
        )
        parts = [
            (prefecture, prefecture_kana),
            (city, city_kana),
            (prefecture + city, prefecture_kana + city_kana),
        ]
        if town_is_usable:
            parts.extend(((town, town_kana), (city + town, city_kana + town_kana)))
        for raw_surface, raw_reading in parts:
            surface = normalize_surface(raw_surface)
            reading = normalize_reading(raw_reading)
            if not surface or not is_valid_reading(reading):
                continue
            record = TermRecord(
                surface=surface,
                reading=reading,
                category="address",
                provenance=provenance,
                reading_source="official_kana",
                reading_confidence="official",
                metadata=metadata,
            )
            records[(surface, reading)] = record
    return list(records.values())
