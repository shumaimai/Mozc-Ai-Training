"""Pure helpers shared by Sarashina-JEV training and tests."""

from __future__ import annotations


def select_even_layers(total_layers: int, keep_layers: int) -> list[int]:
    """Select layers approximately uniformly while keeping first and last."""
    if keep_layers < 1:
        raise ValueError("keep_layers must be >= 1")
    if keep_layers > total_layers:
        raise ValueError("keep_layers cannot exceed total_layers")
    if keep_layers == total_layers:
        return list(range(total_layers))
    if keep_layers == 1:
        return [total_layers - 1]

    raw = [round(i * (total_layers - 1) / (keep_layers - 1)) for i in range(keep_layers)]
    out: list[int] = []
    used: set[int] = set()
    for idx in raw:
        if idx not in used:
            out.append(idx)
            used.add(idx)
    if len(out) < keep_layers:
        for idx in range(total_layers):
            if idx not in used:
                out.append(idx)
                used.add(idx)
                if len(out) == keep_layers:
                    break
        out.sort()
    return out
