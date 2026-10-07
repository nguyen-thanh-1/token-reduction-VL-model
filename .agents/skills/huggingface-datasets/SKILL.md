---
name: huggingface-datasets
description: Implement safe and reproducible downloads for the project datasets.
---

# Hugging Face datasets

## Sources

- GQA: `lmms-lab-encoder/GQA`
- MMBench: `HuggingFaceM4/MMBench`
- MME: `lmms-lab-encoder/MME`

## GQA restriction

Only use these balanced configurations:

- `test_balanced_images`
- `test_balanced_instructions`
- `testdev_balanced_images`
- `testdev_balanced_instructions`
- `train_balanced_images`
- `train_balanced_instructions`
- `val_balanced_image`
- `val_balanced_instructions`

Never add or silently fall back to GQA `*_all_*` configurations. Keep the
configuration list explicit so an upstream change cannot trigger a full
dataset download accidentally.

## Storage and credentials

- Use `datasets.load_dataset` and persist completed datasets with
  `save_to_disk`.
- Store outputs under separate `data/raw/GQA`, `data/raw/MMB`, and
  `data/raw/MME` trees.
- Keep downloaded data ignored by Git; commit only scripts and metadata.
- Use Hugging Face CLI authentication or `HF_TOKEN`; never hard-code tokens.
- Keep bulk downloads explicit and avoid triggering them during validation.
