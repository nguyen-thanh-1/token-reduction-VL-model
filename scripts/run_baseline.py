"""Run the full-token Qwen3-VL baseline on a saved dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.sources import iter_source_samples  # noqa: E402
from token_reduction_vl.evaluation.runner import run_baseline  # noqa: E402
from token_reduction_vl.models.qwen3_vl import Qwen3VLBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml",
    )
    parser.add_argument("--dataset", help="Configured source name, e.g. mme_test.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    run_config = config["run"]
    sources = config["data"]["sources"]
    dataset_name = args.dataset or run_config["default_dataset"]
    if dataset_name not in sources:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Available sources: {', '.join(sources)}"
        )

    model = Qwen3VLBaseline(
        model_id=args.model_id or model_config["id"],
        dtype=model_config.get("dtype"),
        device_map=model_config.get("device_map", "auto"),
        attn_implementation=model_config.get("attn_implementation"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        max_new_tokens=model_config.get("max_new_tokens", 32),
        do_sample=model_config.get("do_sample", False),
    )
    samples = iter_source_samples(sources[dataset_name], project_root=PROJECT_ROOT)
    limit = args.limit if args.limit is not None else run_config.get("limit")
    output_path = args.output or (
        PROJECT_ROOT / run_config.get("output_dir", "outputs/baseline") / f"{dataset_name}.jsonl"
    )
    count = run_baseline(samples, model, output_path, limit=limit)
    print(f"Wrote {count:,} predictions to {output_path}")


if __name__ == "__main__":
    main()
