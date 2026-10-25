"""PP3 step 2b: route-B labels (gradient x input saliency) for the training split.

For each sample: full-token greedy answer (no EOS, <= 8 tokens), one backward
pass for input gradients, then compressed features + saliency are appended to
shards in data/features/pp3_scorer/. Use --start/--end to split the work.

Example:
    uv run python scripts/pp3/make_saliency_labels.py --start 0 --end 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.data import SaliencyRecord, ShardWriter  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase, load_dsfold_params  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.features import FeatureCompressor, question_rows  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.saliency import full_token_answer, gradient_x_input_saliency  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp3_scorer.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    labels = config["labels"]
    model = config["model"]

    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    base = DSFoldBase(load_dsfold_params(PROJECT_ROOT / config["dsfold_params"]))
    compressor = FeatureCompressor.load(PROJECT_ROOT / config["compressor"])
    ids = read_split(PROJECT_ROOT / labels["split"])[args.start : args.end]
    writer = ShardWriter(
        PROJECT_ROOT / labels["output_dir"],
        f"{labels['prefix']}_from{args.start:06d}",
        shard_size=labels.get("shard_size", 1000),
    )

    skipped = 0
    samples = iter_split_samples(PROJECT_ROOT / labels["canonical"], ids, project_root=PROJECT_ROOT)
    for sample in tqdm(samples, total=len(ids), desc="saliency"):
        text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
        encoded = runner.encode(sample.image, sample.question, text)
        answer = full_token_answer(runner, encoded)
        if answer.numel() == 0:
            skipped += 1
            continue
        saliency = gradient_x_input_saliency(runner, encoded, answer)
        x = runner.pruner_input(encoded, 1.0)
        with torch.no_grad():
            scores = base.scores(x)
            v = compressor.v(scores.v_unit)
            levels = [projection(level) for projection, level in zip(compressor.levels, scores.f_hat, strict=True)]
            q_raw = question_rows(x, base.params.stopwords)
            q = compressor.q(q_raw) if q_raw.numel() else torch.zeros((0, compressor.q.output_dim))
        writer.add(
            SaliencyRecord(
                sample_id=sample.sample_id,
                image_id=str(sample.metadata.get("image_id")),
                grid_hw=encoded.grid_hw,
                v=v.half().cpu().numpy(),
                f=torch.stack(levels).half().cpu().numpy(),
                z_rel=scores.z_rel.cpu().numpy(),
                z_ms=scores.z_ms.cpu().numpy(),
                q=q.half().cpu().numpy(),
                saliency=saliency.cpu().numpy(),
            )
        )
    writer.flush()
    print(f"Done; skipped {skipped} samples with an empty answer.")


if __name__ == "__main__":
    main()
