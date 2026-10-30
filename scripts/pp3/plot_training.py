"""Plot PP3 training curves for every history_*.json in outputs/pp3_scorer/training/.

Can be run while training is in progress; the history file is updated after each epoch.

Example:
    uv run python scripts/pp3/plot_training.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.pruning.pp3_scorer.plots import plot_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "outputs" / "pp3_scorer" / "training")
    args = parser.parse_args()
    paths = sorted(args.dir.glob("history_*.json"))
    if not paths:
        raise SystemExit(f"No history_*.json in {args.dir}")
    for path in paths:
        print(f"wrote {plot_history(path)}")


if __name__ == "__main__":
    main()
