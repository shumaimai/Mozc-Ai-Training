"""Phase 1 production-contract schema and validation.

This module validates records; it does not generate Dataset v2.  A record must
be reconstructable from actual Mozc candidates and must not silently downgrade
to surface-only candidates.
"""
from __future__ import annotations

from typing import Any

FORMAT_VERSION = "contextual-ranking-v2-format-v1"
SCHEMA_VERSION = "contextual-ranking-v2-record-v1"
PROTECTIONS = {"HARD_PROTECT", "DELTA_ONLY", "NORMAL"}
CATEGORIES = {"DEFAULT", "SYMBOL", "OTHER"}
EXAMPLE_STATUSES = {"NEURAL_ELIGIBLE", "PROTECTED_EVAL_ONLY", "COVERAGE_FAILURE"}

CANDIDATE_REQUIRED = {
    "surface", "rank", "cost", "cost_delta", "lid", "rid",
    "attributes", "category", "converted_segment_count", "protection",
}
RECORD_REQUIRED = {
    "schema_version", "format_version", "source_id", "reading",
    "context_prev", "gold", "target_segment_index", "candidates",
}


def validate_candidate(candidate: Any, expected_rank: int | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return ["candidate_not_object"]
    missing = CANDIDATE_REQUIRED - candidate.keys()
    errors.extend(f"candidate_missing:{key}" for key in sorted(missing))
    if missing:
        return errors
    if not isinstance(candidate["surface"], str) or not candidate["surface"]:
        errors.append("candidate_surface_invalid")
    for key in ("rank", "cost", "cost_delta", "lid", "rid", "attributes", "converted_segment_count"):
        if not isinstance(candidate[key], int):
            errors.append(f"candidate_{key}_not_int")
    if expected_rank is not None and candidate["rank"] != expected_rank:
        errors.append("candidate_rank_not_contiguous")
    if candidate["category"] not in CATEGORIES:
        errors.append("candidate_category_invalid")
    if candidate["protection"] not in PROTECTIONS:
        errors.append("candidate_protection_invalid")
    if candidate["converted_segment_count"] < 1:
        errors.append("candidate_segment_count_invalid")
    if "wcost" in candidate and not isinstance(candidate["wcost"], int):
        errors.append("candidate_wcost_not_int")
    return errors


def validate_record(record: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record_not_object"]
    errors.extend(f"record_missing:{key}" for key in sorted(RECORD_REQUIRED - record.keys()))
    if errors:
        return errors
    if record["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if record["format_version"] != FORMAT_VERSION:
        errors.append("format_version_invalid")
    for key in ("source_id", "reading", "gold"):
        if not isinstance(record[key], str) or not record[key]:
            errors.append(f"{key}_invalid")
    if not isinstance(record["context_prev"], str) or len(record["context_prev"]) > 50:
        errors.append("context_prev_invalid")
    if not isinstance(record["target_segment_index"], int) or record["target_segment_index"] < 0:
        errors.append("target_segment_index_invalid")
    if "example_status" in record and record["example_status"] not in EXAMPLE_STATUSES:
        errors.append("example_status_invalid")
    if "example_reason" in record and (not isinstance(record["example_reason"], str) or not record["example_reason"]):
        errors.append("example_reason_invalid")
    candidates = record["candidates"]
    if not isinstance(candidates, list) or not candidates:
        return errors + ["candidates_invalid"]
    for rank, candidate in enumerate(candidates):
        errors.extend(validate_candidate(candidate, rank))
    if candidates and record["gold"] not in {c["surface"] for c in candidates if isinstance(c, dict) and "surface" in c} and record.get("example_status") != "COVERAGE_FAILURE":
        errors.append("gold_not_in_candidates")
    return errors


def is_valid_record(record: Any) -> bool:
    return not validate_record(record)


__all__ = ["FORMAT_VERSION", "SCHEMA_VERSION", "EXAMPLE_STATUSES", "validate_candidate", "validate_record", "is_valid_record"]
