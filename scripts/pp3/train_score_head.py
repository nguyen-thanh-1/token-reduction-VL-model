"""PP3 step 4: route-B training of the score head from saliency shards (no VLM needed).

Example:
    uv run python scripts/pp3/train_score_head.py --arch 3a
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.data import load_records  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.features import FeatureCompressor  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.train import RouteBConfig, train_route_b  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp3_scorer.yaml")
    parser.add_argument("--arch", choices=["3a", "3b", "3c"], default=None)
    parser.add_argument("--seed", type=int, action="append", default=None, help="Repeatable; defaults to config seeds.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap loaded samples to limit RAM.")
    parser.add_argument("--resume", action="store_true", help="Continue from score_head_<arch>_seed<s>_last.pt.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    route_b = dict(config["route_b"])
    seeds = args.seed or route_b.pop("seeds", [0])
    route_b.pop("seeds", None)
    output_dir = PROJECT_ROOT / route_b.pop("output_dir")
    base = RouteBConfig(**route_b)
    if args.arch:
        base = replace(base, arch=args.arch)

    records = load_records(PROJECT_ROOT / config["labels"]["output_dir"], config["labels"]["prefix"], args.max_samples)
    if not records:
        raise SystemExit("No saliency shards found; run scripts/pp3/make_saliency_labels.py first.")
    dims = FeatureCompressor.load(PROJECT_ROOT / config["compressor"]).dims
    print(f"Loaded {len(records):,} samples; feature dims {dims}")
    for seed in seeds:
        checkpoint = train_route_b(
            records,
            dims,
            replace(base, seed=seed),
            output_dir,
            PROJECT_ROOT / "outputs" / "pp3_scorer" / "training",
            resume=args.resume,
        )
        print(f"seed {seed}: best checkpoint {checkpoint}")


if __name__ == "__main__":
    main()
