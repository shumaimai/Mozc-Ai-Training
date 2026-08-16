from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime

from .normalize import is_valid_reading, normalize_reading, normalize_surface
from .records import Provenance, TermRecord

# Official Wikidata Query Service. Data is CC0; readings come from P1814
# (name in kana) and are treated as unverified until the Mozc/DeepSeek stages
# confirm them. Override with --endpoint if WQS is throttled or unavailable.
DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MozcAiTraining/0.1 (IME dataset pipeline; github.com/shumaimai)"

# Entities located in Japan (P17=Q17) that carry a kana reading (P1814). The
# country filter keeps places, facilities, and organizations while excluding
# bare person entities, which do not take P17.
QUERY_TEMPLATE = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?itemLabel ?kana (SAMPLE(?tl) AS ?typeLabel) WHERE {{
  ?item wdt:P1814 ?kana .
  ?item wdt:P17 wd:Q17 .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "ja")
  OPTIONAL {{ ?item wdt:P31 ?type . ?type rdfs:label ?tl . FILTER(LANG(?tl) = "ja") }}
}}
GROUP BY ?item ?itemLabel ?kana
ORDER BY ?item
LIMIT {limit} OFFSET {offset}
"""


def run_query(query: str, endpoint: str = DEFAULT_ENDPOINT, retries: int = 5) -> list[dict]:
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            return payload["results"]["bindings"]
        except urllib.error.HTTPError as error:
            if error.code in (429, 503) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise
    return []


def records_from_bindings(
    bindings: Iterable[dict],
    retrieved_at: str | None = None,
) -> list[TermRecord]:
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    provenance = Provenance(
        source_id="wikidata",
        source_url="https://query.wikidata.org/sparql",
        license_id="CC0-1.0",
        retrieved_at=timestamp,
    )
    records: dict[tuple[str, str], TermRecord] = {}
    for binding in bindings:
        surface = normalize_surface(binding.get("itemLabel", {}).get("value", ""))
        reading = normalize_reading(binding.get("kana", {}).get("value", ""))
        if not surface or not is_valid_reading(reading):
            continue
        key = (surface, reading)
        if key in records:
            continue
        qid = binding.get("item", {}).get("value", "").rsplit("/", 1)[-1]
        wikidata_type = binding.get("typeLabel", {}).get("value", "")
        records[key] = TermRecord(
            surface=surface,
            reading=reading,
            category="place_or_facility",
            provenance=provenance,
            reading_source="wikidata_kana",
            reading_confidence="unverified",
            metadata={"qid": qid, "wikidata_type": wikidata_type},
        )
    return list(records.values())


def fetch_places(
    endpoint: str = DEFAULT_ENDPOINT,
    page_size: int = 10000,
    max_pages: int = 20,
    retrieved_at: str | None = None,
) -> list[TermRecord]:
    timestamp = retrieved_at or datetime.now(UTC).isoformat()
    records: dict[tuple[str, str], TermRecord] = {}
    for page in range(max_pages):
        query = QUERY_TEMPLATE.format(limit=page_size, offset=page * page_size)
        bindings = run_query(query, endpoint)
        if not bindings:
            break
        for record in records_from_bindings(bindings, retrieved_at=timestamp):
            records[(record.surface, record.reading)] = record
        print(f"  page {page + 1}: {len(bindings)} rows, {len(records)} unique so far")
        if len(bindings) < page_size:
            break
    return list(records.values())
