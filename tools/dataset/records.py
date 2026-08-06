from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provenance:
    source_id: str
    source_url: str
    license_id: str
    retrieved_at: str
    source_version: str = ""


@dataclass(frozen=True)
class TermRecord:
    surface: str
    reading: str
    category: str
    provenance: Provenance
    reading_source: str
    reading_confidence: str
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TermRecord:
        return cls(
            surface=value["surface"],
            reading=value["reading"],
            category=value["category"],
            provenance=Provenance(**value["provenance"]),
            reading_source=value["reading_source"],
            reading_confidence=value["reading_confidence"],
            aliases=tuple(value.get("aliases", ())),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class CandidateComparison:
    record: TermRecord
    context: tuple[str, ...]
    candidates: tuple[str, ...]
    gold_rank: int | None
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["record"] = self.record.to_dict()
        return value
