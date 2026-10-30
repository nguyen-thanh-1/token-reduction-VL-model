"""Plot PP4 training curves for every lambda*.jsonl log in outputs/pp4_gate/training/.

Can be run while training is in progress; logs are appended every 50 steps.

Example:
    uv run python scripts/pp4/plot_training.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.pruning.pp4_gate.plots import plot_training_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "outputs" / "pp4_gate" / "training")
    args = parser.parse_args()
    paths = sorted(path for path in args.dir.glob("lambda*.jsonl") if not path.stem.endswith("_errors"))
    if not paths:
        raise SystemExit(f"No lambda*.jsonl logs in {args.dir}")
    for path in paths:
        print(f"wrote {plot_training_log(path)}")


if __name__ == "__main__":
    main()
