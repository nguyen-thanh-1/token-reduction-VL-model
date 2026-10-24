# ADR 0001: Canonical VQA data schema v1

- Status: Proposed
- Date: 2026-10-08

## Context and decision drivers

GQA, MMBench, and MME expose different identifiers, labels, prompt context,
answer formats, and image layouts. Training, full-token baselines, future
pruning experiments, inference, and benchmark scoring need to consume the same
examples without embedding Qwen-specific chat tokens into persisted data.

Important source constraints are:

- GQA is restricted to balanced configurations and stores images separately
  from instructions.
- MMBench validation is labelled, while its test split is unlabelled.
- MME has two yes/no questions per image and repeats `question_id`, so that
  source field cannot be the canonical unique sample key.

## Options considered

1. Read every raw dataset directly in each training/evaluation script. This
   avoids an intermediate file but duplicates joins, prompts, and split rules.
2. Persist Qwen chat messages. This is immediately convenient but couples the
   datasets to one processor and makes later pruning/model comparisons harder.
3. Persist a versioned, model-agnostic JSONL record and construct model
   messages at runtime.

## Decision

Use option 3. Canonical records use schema version `1.0` and contain stable
sample and image identifiers, dataset/split/task fields, an image reference,
question and optional hint, structured choices, answers, category, label
availability, and source metadata.

Processed records reference images under `data/raw` instead of copying them.
Qwen messages and supervised targets are constructed at runtime. Split use is
fixed initially as follows:

- training: GQA balanced train;
- validation: GQA balanced val;
- benchmark evaluation: GQA balanced testdev, MMBench validation, MME test;
- inference/submission only: GQA balanced test and MMBench test.

MMBench and MME are not used as training data in the initial protocol, which
avoids benchmark leakage. Benchmark scoring remains separate from generation.

## Consequences, risks, and validation

The same processed files can support training, baseline inference, and future
pruning methods. Dataset-specific prompt and metric logic remains explicit.
Raw images must stay at their recorded paths; moving or deleting `data/raw`
invalidates processed image references. Source manifests and file sizes are
copied into the processed manifest, but upstream Hugging Face revisions should
also be pinned before publishing final thesis results.

Validation consists of schema checks, unique `sample_id` checks, referenced
image existence checks, fixture-based adapter tests, and smoke preprocessing
with `--limit`. Full dataset conversion and model inference remain explicit
commands rather than test side effects.

## Affected files

- `configs/data_v1.yaml`
- `scripts/preprocess_datasets.py`
- `scripts/run_baseline.py`
- `scripts/evaluate_predictions.py`
- `src/token_reduction_vl/data/schema.py`
- `src/token_reduction_vl/data/adapters/`
- `src/token_reduction_vl/data/messages.py`
- `src/token_reduction_vl/evaluation/metrics.py`
