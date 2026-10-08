# Adaptive Visual Token Pruning for Vision Language Models

This repository is the initial implementation scaffold for a thesis project on
adaptive visual-token pruning in vision-language models. The proposal is still
exploratory, so the code does not commit to a pruning method yet. The first
milestone is reproducible data preparation and a full-token baseline with
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

## Current scope

The current code prepares three things:

1. Downloading the selected Hugging Face datasets into separate local folders.
2. Running ordinary full-token Qwen3-VL inference and writing JSONL results.
3. Providing a method-agnostic pruning interface for future Random-K,
   question-aware, adaptive-budget, diversity-aware, or spatial methods.

No pruning algorithm, training loop, or model-weight download is executed by
the repository setup itself.

## Project structure

```text
.
├── configs/
│   └── baseline_qwen3_vl.yaml       # model, data sources, and run defaults
├── data/
│   ├── raw/                         # downloaded Hugging Face datasets
│   │   ├── GQA/
│   │   ├── MMB/
│   │   └── MME/
│   ├── processed/                   # normalized or filtered samples
│   ├── features/                    # cached visual features
│   ├── predictions/                 # optional per-example predictions
│   └── metrics/                     # evaluation summaries
├── outputs/
│   └── baseline/                    # baseline JSONL outputs
├── scripts/
│   ├── download_datasets.py         # dataset downloader
│   └── run_baseline.py              # full-token baseline runner
├── src/token_reduction_vl/
│   ├── data/                        # canonical sample and source adapters
│   ├── evaluation/                  # JSONL runner and metrics
│   ├── models/                      # Qwen3-VL wrapper
│   └── pruning/                     # future pruning interfaces
└── .agents/                         # project-specific agent instructions
```

Downloaded data, model caches, virtual environments, and experiment outputs
are ignored by Git. Only code, configuration, and metadata should be committed.

## Environment setup

The project uses Python 3.12 and `uv`:

```powershell
uv sync
```

For private or gated Hugging Face resources, authenticate separately:

```powershell
hf auth login
```

The Qwen3-VL baseline is intended for a CUDA-capable machine. `device_map: auto`
and `dtype: auto` are configured in `configs/baseline_qwen3_vl.yaml`; adjust
them there if the available hardware requires a different setup.

## Download datasets

The downloader uses `datasets.load_dataset` and exports every completed
configuration under `data/raw/` with visible image files:

```powershell
uv run python scripts/download_datasets.py --dataset gqa
uv run python scripts/download_datasets.py --dataset mmb
uv run python scripts/download_datasets.py --dataset mme
```

To download every configured dataset:

```powershell
uv run python scripts/download_datasets.py --dataset all
```

GQA is intentionally restricted to these balanced configurations:

```text
test_balanced_images
test_balanced_instructions
testdev_balanced_images
testdev_balanced_instructions
train_balanced_images
train_balanced_instructions
val_balanced_images
val_balanced_instructions
```

The `*_all_*` GQA configurations are not downloaded.

Each output directory contains:

```text
<configuration>/
├── images/          # JPEG/PNG/WebP files when the source has an image column
├── records.jsonl    # metadata; image values are relative paths into images/
└── manifest.json    # source repository, config, split, row and image counts
```

Older directories created by `save_to_disk` are detected and converted in
place, so rerunning the same command exports their embedded images without
downloading the dataset again. The existing Arrow files are left untouched.

To convert only datasets that are already present locally and avoid downloading
missing GQA configurations, run:

```powershell
uv run python scripts/download_datasets.py --dataset all --existing-only
```

## Run the full-token baseline

After the relevant dataset has been downloaded, run a small smoke subset first:

```powershell
uv run python scripts/run_baseline.py `
  --config configs/baseline_qwen3_vl.yaml `
  --dataset mme_test `
  --limit 10
```

Other configured sources are `mmb_test` and `gqa_val_balanced`. The GQA source
joins the balanced instructions and balanced images by the configured image
key. Results are written to `outputs/baseline/<dataset>.jsonl` and include the
prediction, reference answers, exact-match when available, latency, input and
output token counts, and peak VRAM when CUDA is available.

The baseline uses the standard Qwen3-VL Transformers path: an
`AutoProcessor`, `Qwen3VLForConditionalGeneration`, `apply_chat_template`, and
`generate`. It keeps all visual tokens and records `pruning_method: "none"`.

## Research extension path

The proposal suggests evaluating quality and efficiency together rather than
optimizing accuracy alone. The intended order is:

1. Full-token baseline as the quality reference.
2. Random-K and uniform spatial baselines.
3. Fixed-K question-aware ranking.
4. Adaptive budget prediction.
5. Diversity-aware selection and, if needed, spatial safeguards.
6. Ablations, cross-dataset evaluation, latency/VRAM profiling, and token
   visualizations.

`src/token_reduction_vl/pruning/base.py` defines the initial interface without
assuming which of these methods will become the thesis contribution.

## Current validation policy

The repository setup does not download datasets or model weights automatically.
Use the commands above explicitly when ready. At this stage, validation should
focus on configuration, import/syntax checks, and small smoke subsets before
any full benchmark run.
