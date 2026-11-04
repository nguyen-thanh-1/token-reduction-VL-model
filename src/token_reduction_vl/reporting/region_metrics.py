"""Shape metrics of a kept-token set on the merged visual grid (goal 2: sensible regions).

Tokens are indexed row-major on an ``H × W`` grid (``index = row * W + col``).
Neighbours are the 8 surrounding cells, so diagonal contact counts as connected,
as it looks on an overlay.

* ``components``      number of connected groups of kept tokens (lower = fewer blobs)
* ``neighbor_fraction`` share of kept tokens with at least one kept neighbour
* ``coverage``        share of ``block × block`` cells holding at least one kept token
* ``jaccard``         overlap of two keep sets; across questions on the same image,
                      high values mean the selection ignores the question
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from statistics import mean
from typing import Any

OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def _neighbors(index: int, height: int, width: int) -> Iterable[int]:
    row, col = divmod(index, width)
    for dr, dc in OFFSETS:
        r, c = row + dr, col + dc
        if 0 <= r < height and 0 <= c < width:
            yield r * width + c


def components(keep: Iterable[int], grid_hw: Sequence[int]) -> int:
    height, width = grid_hw
    remaining = set(int(i) for i in keep)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            for other in _neighbors(stack.pop(), height, width):
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
    return count


def neighbor_fraction(keep: Iterable[int], grid_hw: Sequence[int]) -> float:
    height, width = grid_hw
    kept = set(int(i) for i in keep)
    if not kept:
        return 0.0
    return sum(any(other in kept for other in _neighbors(i, height, width)) for i in kept) / len(kept)


def coverage(keep: Iterable[int], grid_hw: Sequence[int], block: int = 3) -> float:
    height, width = grid_hw
    cells = {((i // width) // block, (i % width) // block) for i in keep}
    total = -(-height // block) * -(-width // block)
    return len(cells) / total


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def question_overlap(rows: Iterable[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Mean pairwise Jaccard of keep sets between questions on the same image, and the image count."""

    by_image: dict[str, list[set[int]]] = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), []).append(set(row["keep_indices"]))
    per_image = [mean(jaccard(a, b) for a, b in combinations(sets, 2)) for sets in by_image.values() if len(sets) >= 2]
    return (mean(per_image) if per_image else None), len(per_image)


def summarize(rows: Sequence[Mapping[str, Any]], block: int = 3) -> dict[str, float | int | None]:
    """Mean shape metrics over rows that carry ``keep_indices`` and ``grid_hw``."""

    rows = [row for row in rows if row.get("keep_indices") is not None and row.get("grid_hw")]
    if not rows:
        return {"rows": 0}
    overlap, images = question_overlap(rows)
    return {
        "rows": len(rows),
        "keep_ratio": mean(row["K"] / row["N"] for row in rows),
        "components": mean(components(row["keep_indices"], row["grid_hw"]) for row in rows),
        "components_per_10": mean(10 * components(row["keep_indices"], row["grid_hw"]) / row["K"] for row in rows),
        "neighbor_fraction": mean(neighbor_fraction(row["keep_indices"], row["grid_hw"]) for row in rows),
        "coverage": mean(coverage(row["keep_indices"], row["grid_hw"], block) for row in rows),
        "question_overlap": overlap,
        "overlap_images": images,
    }
