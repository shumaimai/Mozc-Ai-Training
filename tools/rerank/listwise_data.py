"""Public candidate-set preparation without GPU or model dependencies."""
from tools.rerank.contextual_ranking_v2_contract import format_v2
from tools.rerank.context_clip import normalize_reading, clean_context


def prepare(rows, tokenizer, train=False):
    groups = []
    for i, row in enumerate(rows):
        if train and row.get("eligibility_status") != "NEURAL_ELIGIBLE":
            continue
        candidates = list(dict.fromkeys(c["surface"] for c in
            row["candidates"][:30] if c.get("surface")))
        gold = row["gold"]
        if train and (gold not in candidates or len(candidates) < 2):
            continue
        reading = normalize_reading(row["reading"])
        # C++ clipping can leave leading whitespace; reproduce the daemon's
        # final clean before formatting both training and validation inputs.
        context = clean_context(row["context_prev"] or "")
        texts = [format_v2(reading, context, c) for c in candidates]
        tokens = [[1] + tokenizer.encode(t, out_type=int)[:126] + [2] for t in texts]
        groups.append({"row_index": i, "reading": reading, "context": context,
            "candidates": candidates, "gold": gold, "gold_index":
            candidates.index(gold) if gold in candidates else -1,
            "tokens": tokens, "eligibility_status": row.get("eligibility_status"),
            "hard_protect": row["candidates"][0].get("protection") == "HARD_PROTECT"})
    return groups
