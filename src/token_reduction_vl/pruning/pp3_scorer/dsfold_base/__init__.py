"""PP1 (DSFold) re-implemented by the PP3 member from ``pp1-dsfold/method.md``.

Must pass ``tests/test_pp1_conformance.py``; do not import another method's copy.
"""

from .dpp import SelectionResult, greedy_dpp, select_order, weighted_maxmin
from .fold import assign_dropped, dropped_indices, fold_deepstack, grid_coordinates
from .params import DSFoldParams, load_dsfold_params
from .pruner import DSFoldBase, DSFoldRun
from .scoring import ScoreBundle, compute_scores, unit_rows, zscore

__all__ = [
    "DSFoldBase",
    "DSFoldParams",
    "DSFoldRun",
    "ScoreBundle",
    "SelectionResult",
    "assign_dropped",
    "compute_scores",
    "dropped_indices",
    "fold_deepstack",
    "greedy_dpp",
    "grid_coordinates",
    "load_dsfold_params",
    "select_order",
    "unit_rows",
    "weighted_maxmin",
    "zscore",
]
