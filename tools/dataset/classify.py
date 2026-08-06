from __future__ import annotations

from .normalize import is_meaningful_surface, is_valid_reading, normalize_surface
from .records import CandidateComparison, TermRecord


DICTIONARY_CATEGORIES = {
    "address",
    "geographic_name",
    "public_facility",
    "technical_term",
}


def candidate_rank(gold: str, candidates: list[str] | tuple[str, ...]) -> int | None:
    normalized_gold = normalize_surface(gold)
    for index, candidate in enumerate(candidates, start=1):
        if normalize_surface(candidate) == normalized_gold:
            return index
    return None


def classify(record: TermRecord, candidates: list[str] | tuple[str, ...], context: list[str] | tuple[str, ...] = (), top_k: int = 5, max_rank: int = 50) -> CandidateComparison:
    normalized_candidates = tuple(normalize_surface(candidate) for candidate in candidates if candidate)
    if not is_meaningful_surface(record.surface):
        return CandidateComparison(record, tuple(context), normalized_candidates, None, "reject", "invalid_surface")
    if not is_valid_reading(record.reading):
        return CandidateComparison(record, tuple(context), normalized_candidates, None, "reject", "invalid_reading")
    rank = candidate_rank(record.surface, normalized_candidates)
    if rank is not None and rank <= top_k:
        return CandidateComparison(record, tuple(context), normalized_candidates, rank, "abstain", "already_in_top_k")
    if rank is not None and rank <= max_rank:
        return CandidateComparison(record, tuple(context), normalized_candidates, rank, "rerank", "present_below_top_k")
    if record.category in DICTIONARY_CATEGORIES and record.reading_confidence == "official":
        return CandidateComparison(record, tuple(context), normalized_candidates, rank, "dictionary_gap", "stable_term_missing")
    return CandidateComparison(record, tuple(context), normalized_candidates, rank, "generation_gap", "missing_candidate")
