# Adaptive Visual Token Pruning for Vision Language Models

This repository is the initial implementation scaffold for a thesis project on
adaptive visual-token pruning in vision-language models. The proposal is still
exploratory, so the code does not commit to a pruning method yet. The first
milestone is reproducible data preparation and a full-token baseline with
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

## Current scope

The current code prepares four things:

1. Downloading the selected Hugging Face datasets into separate local folders.
2. Normalizing GQA, MMBench, and MME into one versioned VQA JSONL schema.
3. Running ordinary full-token Qwen3-VL inference and benchmark-specific scoring.
4. Providing a method-agnostic pruning interface for future Random-K,
   question-aware, adaptive-budget, diversity-aware, or spatial methods.

No pruning algorithm, training loop, or model-weight download is executed by
the repository setup itself.

## Project structure

```text
.
├── architecture/                    # editable model diagrams and exports
│   └── model-goc-baseline/          # full-token Qwen3-VL baseline
├── configs/
│   ├── baseline_qwen3_vl.yaml       # model, data sources, and run defaults
│   └── data_v1.yaml                 # split roles and canonical paths
├── data/
│   ├── raw/                         # downloaded Hugging Face datasets
│   │   ├── GQA/
│   │   ├── MMB/
│   │   └── MME/
│   ├── processed/v1/                # canonical, versioned JSONL records
│   ├── features/                    # cached visual features
│   ├── predictions/                 # optional per-example predictions
│   └── metrics/                     # evaluation summaries
├── outputs/
│   ├── baseline/                    # shareable baseline prediction JSONL
│   ├── reports/                     # shareable plots, tables, and reports
│   └── checkpoints/                 # local model weights, ignored by Git
├── scripts/
│   ├── download_datasets.py         # dataset downloader
│   ├── preprocess_datasets.py       # canonical schema conversion
│   ├── evaluate_predictions.py      # benchmark-specific metrics
│   ├── visualize_benchmarks.py      # structured plots, tables, and report
│   └── run_baseline.py              # full-token baseline runner
├── src/token_reduction_vl/
│   ├── data/                        # canonical sample and source adapters
│   ├── evaluation/                  # JSONL runner and metrics
│   ├── models/                      # Qwen3-VL wrapper
│   └── pruning/                     # future pruning interfaces
└── .agents/                         # project-specific agent instructions
```

Downloaded data, model caches, virtual environments, temporary test files, and
training checkpoints are ignored by Git. Benchmark predictions and reports in
`outputs/` are intentionally versioned so team members can inspect and compare
the same results without rerunning inference.

## Model architecture diagrams

Editable network diagrams live under `architecture/`, with one directory per
baseline or pruning method. The current full-token model is documented in:

- `architecture/model-goc-baseline/qwen3-vl-2b-instruct-baseline.drawio`
- `architecture/model-goc-baseline/README.md`

The Draw.io source has separate pages for the end-to-end execution path and
module-level details. Export PNG, SVG, or PDF versions into the adjacent
`exports/` directory so each design remains self-contained.

## Environment setup

The project uses Python 3.12 and `uv`:

```powershell
uv sync
```

On Windows, the project pins the compatible `torch 2.14.1` and
`torchvision 0.29.1` CUDA 13.0 wheels through `uv`. After syncing, verify that
the NVIDIA GPU is visible before running the model:

```powershell
uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The current CUDA wheel choice requires an NVIDIA driver capable of CUDA 13.0.
Use a matching PyTorch index for a different machine rather than installing a
CPU-only `torch` build accidentally.

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

For MMBench, the downloader retrieves both `validation` and `test`.
`validation` contains labels and is used for local evaluation; `test` is kept
for inference/submission and should not be reported as locally scored accuracy.

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

## Preprocess into the shared schema

Preprocessing never downloads data and never duplicates image files. It writes
model-agnostic JSONL records whose `image` field points back to `data/raw`:

```powershell
# Cheap pipeline check: at most 10 rows from every available split
uv run python scripts/preprocess_datasets.py --dataset all --limit 10

# Replace smoke outputs with complete processed files
uv run python scripts/preprocess_datasets.py --dataset all --overwrite
```

MMBench validation must be downloaded before `--dataset all` can complete. A
single dataset or split can be processed independently, for example:

```powershell
uv run python scripts/preprocess_datasets.py --dataset gqa --split train
uv run python scripts/preprocess_datasets.py --dataset mme --split test
```

Outputs are organized as:

```text
data/processed/v1/
├── gqa/{train,val,testdev,test}.jsonl
├── mmbench/{validation,test}.jsonl
├── mme/test.jsonl
└── manifest.json
```

Each record includes `sample_id`, `image_id`, dataset/split/task information,
the question, optional hint and choices, labels when available, category, and
preserved source metadata. `sample_id` is always unique; `image_id` may repeat
where several questions belong to the same image, as required by MME pair
scoring. Runtime helpers build Qwen messages and supervised targets without
persisting model-specific special tokens.

The initial leakage-safe split policy is recorded in `configs/data_v1.yaml`:

- train on GQA balanced train;
- tune on GQA balanced val;
- evaluate on GQA balanced testdev, MMBench validation, and MME test;
- use GQA test and MMBench test for inference/submission only.

## Run the full-token baseline

After the relevant dataset has been downloaded, run a small smoke subset first:

```powershell
uv run python scripts/run_baseline.py `
  --config configs/baseline_qwen3_vl.yaml `
  --dataset mme_test `
  --limit 10
```

Other configured sources are `mmb_validation`, `mmb_test`,
`gqa_val_balanced`, and `gqa_testdev_balanced`. All sources consume the
processed schema. Results are written to `outputs/baseline/<dataset>.jsonl` and include the
prediction, reference answers, exact-match when available, latency, input and
output token counts, and peak VRAM when CUDA is available.

The baseline uses the standard Qwen3-VL Transformers path: an
`AutoProcessor`, `Qwen3VLForConditionalGeneration`, `apply_chat_template`, and
`generate`. It keeps all visual tokens and records `pruning_method: "none"`.

Score a completed prediction file separately from generation:

```powershell
uv run python scripts/evaluate_predictions.py `
  outputs/baseline/mme_test.jsonl `
  --output data/metrics/mme_test.json
```

GQA uses normalized short-answer accuracy, MMBench parses and scores A-D/E
labels, and MME reports question accuracy plus category-level pair accuracy
(`accuracy+`) and the conventional combined MME score. Unlabelled inference
files report no accuracy rather than treating missing labels as incorrect.

## Visualize the complete baseline

After all three full prediction files exist, generate the complete report:

```powershell
uv run python scripts/visualize_benchmarks.py
```

The report uses `configs/report_baseline.yaml` as the single source of truth
for input predictions, model identity, precision, pruning method, and output
location. Generated artifacts are deliberately separated from raw predictions
and canonical metric files:

```text
outputs/reports/baseline/qwen3-vl-2b-instruct-full-token-bf16/
├── report.md
├── summary.json
├── manifest.json
├── figures/
│   ├── 01_quality_accuracy.png
│   ├── 02_efficiency_overview.png
│   ├── 03_latency_distribution.png
│   ├── 04_tokens_vs_latency.png
│   └── 05_mme_categories.png
└── tables/
    ├── benchmark_summary.csv
    └── mme_categories.csv
```

The numbered figures keep a stable reading order. Tables remain machine
readable, while `manifest.json` records the source paths, sizes, row counts,
and modification times. Existing report directories are protected from
accidental replacement; regenerate the same report explicitly with:

```powershell
uv run python scripts/visualize_benchmarks.py --overwrite
```

Use a different `report.id` and `report.output_dir` for future pruning methods
instead of mixing them into this full-token baseline directory.

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

The repository setup does not download datasets, preprocess full corpora, or
download model weights automatically. Use the commands above explicitly when
ready. Run unit tests with `uv run pytest`, then use small preprocessing and
baseline smoke subsets before a complete benchmark.
