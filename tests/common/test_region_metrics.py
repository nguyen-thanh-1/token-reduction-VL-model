import pytest

from token_reduction_vl.reporting.region_metrics import (
    components,
    coverage,
    jaccard,
    neighbor_fraction,
    question_overlap,
    summarize,
)

GRID = (4, 6)  # index = row * 6 + col


def test_components_join_diagonal_neighbours() -> None:
    assert components([0, 7, 14], GRID) == 1  # (0,0) (1,1) (2,2)
    assert components([0, 2, 4], GRID) == 3
    assert components([], GRID) == 0


def test_neighbor_fraction_and_coverage() -> None:
    assert neighbor_fraction([0, 1, 23], GRID) == pytest.approx(2 / 3)
    # 3x3 blocks on a 4x6 grid: 2 x 2 = 4 cells.
    assert coverage([0, 23], GRID) == pytest.approx(0.5)
    assert coverage(range(24), GRID) == 1.0


def test_question_overlap_is_per_image() -> None:
    rows = [
        {"image_id": "a", "keep_indices": [0, 1]},
        {"image_id": "a", "keep_indices": [0, 1]},
        {"image_id": "b", "keep_indices": [0, 1]},
        {"image_id": "b", "keep_indices": [2, 3]},
        {"image_id": "c", "keep_indices": [5]},
    ]
    overlap, images = question_overlap(rows)
    assert images == 2 and overlap == pytest.approx(0.5)
    assert jaccard([], []) == 1.0


def test_summarize_skips_rows_without_keep_indices() -> None:
    rows = [
        {"image_id": "a", "keep_indices": [0, 1, 6], "grid_hw": [4, 6], "K": 3, "N": 24},
        {"image_id": "a", "keep_indices": None, "grid_hw": [4, 6], "K": 24, "N": 24},
    ]
    summary = summarize(rows)
    assert summary["rows"] == 1 and summary["components"] == 1 and summary["question_overlap"] is None
