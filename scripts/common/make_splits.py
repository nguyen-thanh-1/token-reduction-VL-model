"""Create the shared sample-id lists in configs/splits/ (METHODS.md section 9.3).

Sampling is by image so that MME question pairs and GQA images stay together.
Reads processed canonical JSONL only; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.data.canonical import iter_canonical_records  # noqa: E402
from token_reduction_vl.data.splits import write_split  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed" / "v1"
SPLITS = PROJECT_ROOT / "configs" / "splits"


def ids_by_image(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in iter_canonical_records(path):
        groups[record.image_id].append(record.sample_id)
    return groups


def sample_by_image(groups: dict[str, list[str]], target: int, seed: int, per_image: int | None = None) -> list[str]:
    images = sorted(groups)
    random.Random(seed).shuffle(images)
    chosen: list[str] = []
    for image in images:
        members = sorted(groups[image])
        if per_image is not None:
            members = random.Random(f"{seed}:{image}").sample(members, min(per_image, len(members)))
        chosen.extend(members)
        if len(chosen) >= target:
            break
    return chosen[:target] if per_image is not None else chosen


def conformance_ids(seed: int) -> list[str]:
    """40 GQA testdev + 30 MMBench val + 30 MME test, with >= 3 MME images above N_cap."""

    rng = random.Random(seed)
    gqa = sorted(r.sample_id for r in iter_canonical_records(PROCESSED / "gqa" / "testdev.jsonl"))
    mmb = sorted(r.sample_id for r in iter_canonical_records(PROCESSED / "mmbench" / "validation.jsonl"))
    large, small = [], []
    baseline = PROJECT_ROOT / "outputs" / "baseline" / "mme_test.jsonl"
    for line in baseline.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        (large if row["input_tokens"] > 4096 + 64 else small).append(row["sample_id"])
    mme = rng.sample(sorted(large), 3) + rng.sample(sorted(small), 27)
    return rng.sample(gqa, 40) + rng.sample(mmb, 30) + mme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    train = sample_by_image(ids_by_image(PROCESSED / "gqa" / "train.jsonl"), 50_000, seed, per_image=3)
    write_split(SPLITS / "gqa_train_50k.txt", train, comment=f"seed={seed}; GQA train balanced; <=3 questions per image")

    val_groups = ids_by_image(PROCESSED / "gqa" / "val.jsonl")
    val = sample_by_image(val_groups, 5_000, seed, per_image=3)
    write_split(SPLITS / "gqa_val_5k.txt", val, comment=f"seed={seed}; GQA val balanced; threshold selection")
    tune = sample_by_image(val_groups, 2_000, seed + 1, per_image=3)
    write_split(SPLITS / "gqa_val_tune_2k.txt", tune, comment=f"seed={seed + 1}; GQA val balanced; PP1 tuning")

    for name, path in (
        ("dev_1k_gqa", PROCESSED / "gqa" / "testdev.jsonl"),
        ("dev_1k_mmb", PROCESSED / "mmbench" / "validation.jsonl"),
        ("dev_1k_mme", PROCESSED / "mme" / "test.jsonl"),
    ):
        ids = sample_by_image(ids_by_image(path), 1_000, seed)
        write_split(SPLITS / f"{name}.txt", ids, comment=f"seed={seed}; whole images; {path.relative_to(PROJECT_ROOT)}")

    write_split(SPLITS / "pp1_conformance_100.txt", conformance_ids(seed), comment=f"seed={seed}; PP1 conformance")
    print(f"Wrote splits to {SPLITS}")


if __name__ == "__main__":
    main()
