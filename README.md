# Adaptive Visual Token Pruning for Vision Language Models

This repository is the working research codebase for a thesis on adaptive
visual-token pruning in vision-language models. The project now has a complete,
reproducible data pipeline and a measured full-token baseline using
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). The
pruning method is intentionally still open: the current baseline is the control
against which future token-reduction methods will be compared.

## Current scope

The current code provides six connected components:

1. Downloading the selected Hugging Face datasets into separate local folders.
2. Normalizing GQA, MMBench, and MME into one versioned VQA JSONL schema.
3. Running ordinary full-token Qwen3-VL inference and benchmark-specific scoring.
4. Generating quality/efficiency figures, tables, manifests, and an interactive
   incorrect-answer gallery.
5. Documenting the verified checkpoint architecture in a five-page Draw.io file.
6. Providing a method-agnostic pruning interface for future Random-K,
   question-aware, adaptive-budget, diversity-aware, or spatial methods.

No pruning algorithm or fine-tuning/training loop has been implemented yet.
Model and dataset downloads remain explicit commands and are never started by
project setup or unit tests.

## Current project status

| Area | Status | Current artifact |
| --- | --- | --- |
| Python environment | Ready | Python 3.12, `uv.lock`, CUDA PyTorch/torchvision pins |
| Raw datasets | Ready locally | GQA balanced, MMBench, and MME under `data/raw/` |
| Canonical data | Ready | Versioned JSONL under `data/processed/v1/` |
| Full-token baseline | Completed | 19,281 predictions across three labelled benchmark splits |
| Benchmark scoring | Completed | GQA, MMBench, and MME-specific metrics |
| Visualization/reporting | Completed | Markdown report, JSON/CSV summaries, five figures, HTML failure gallery |
| Architecture documentation | Completed for baseline | Five-page editable Draw.io model diagram |
| Token-pruning method | Not selected | Interface exists; experiments are the next phase |
| Fine-tuning/training | Not implemented | To be designed after the pruning approach is selected |

The current baseline report is available at
[`outputs/reports/baseline/qwen3-vl-2b-instruct-full-token-bf16/report.md`](outputs/reports/baseline/qwen3-vl-2b-instruct-full-token-bf16/report.md),
and the visual error browser is
[`failure_cases.html`](outputs/reports/baseline/qwen3-vl-2b-instruct-full-token-bf16/failure_cases.html).

## Project structure

```text
.
├── architecture/                    # editable model diagrams and exports
│   └── model-goc-baseline/          # full-token Qwen3-VL baseline
├── configs/
│   ├── baseline_qwen3_vl.yaml       # model, data sources, and run defaults
│   ├── data_v1.yaml                 # split roles and canonical paths
│   └── report_baseline.yaml         # report identity, inputs, figures, failure gallery
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
│   ├── reports/                     # shareable plots, tables, Markdown, and HTML
│   └── checkpoints/                 # local model weights, ignored by Git
├── scripts/
│   ├── download_datasets.py         # dataset downloader
│   ├── preprocess_datasets.py       # canonical schema conversion
│   ├── evaluate_predictions.py      # benchmark-specific metrics
│   ├── visualize_benchmarks.py      # structured plots, tables, and report
│   ├── generate_failure_gallery.py  # standalone HTML error browser
│   └── run_baseline.py              # full-token baseline runner
├── src/token_reduction_vl/
│   ├── data/                        # canonical sample and source adapters
│   ├── evaluation/                  # JSONL runner and metrics
│   ├── models/                      # Qwen3-VL wrapper
│   ├── pruning/                     # future pruning interfaces
│   └── reporting/                   # figures, tables, Markdown, and HTML reports
├── docs/adr/                        # research/data architecture decisions
├── tests/                           # fast dataset, preprocessing, metric, report tests
├── .agents/                         # project-specific agent skills and instructions
├── AGENT.md                         # top-level working guide
├── pyproject.toml                   # package metadata and dependencies
└── uv.lock                          # locked Python environment
```

Downloaded data, Hugging Face/model caches, virtual environments, temporary
test files, and heavyweight training checkpoints are ignored by Git. Benchmark
predictions and lightweight reports in `outputs/` are intentionally versioned
so team members can inspect and compare the same results without rerunning
inference. The canonical schema decision is documented in
[`docs/adr/0001-canonical-vqa-data-v1.md`](docs/adr/0001-canonical-vqa-data-v1.md).

## Model architecture diagrams

Editable network diagrams live under `architecture/`, with one directory per
baseline or pruning method. The current full-token model is documented in:

- `architecture/model-goc-baseline/qwen3-vl-2b-instruct-baseline.drawio`
- `architecture/model-goc-baseline/README.md`

The Draw.io source has five pages:

1. Clean end-to-end overview.
2. Detailed 24-layer vision encoder, PatchMerger, and fixed DeepStack taps.
3. Multimodal fusion and 28-layer decoder overview.
4. Text/image token flow, placeholder replacement, token accounting, and
   three-axis M-RoPE positions.
5. Detailed decoder block, grouped-query attention, SwiGLU, residual paths,
   DeepStack injection, KV cache, and autoregressive generation.

The diagram is checkpoint-specific, not copied from a generic larger Qwen3-VL
diagram. For the inspected 2B checkpoint, the ViT hidden size is 1024, the
language hidden size is 2048, and the fixed zero-based DeepStack visual indexes
are `[5, 11, 17]` (after ViT blocks 6, 12, and 18). DeepStack is documented as
multi-level feature reuse—not token pruning—and the full-token baseline keeps
every visual token emitted by the built-in 2×2 PatchMerger.

Export PNG, SVG, or PDF versions into the adjacent `exports/` directory so
each architecture version remains self-contained.

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

| Dataset | Hugging Face source | Local role |
| --- | --- | --- |
| GQA | [`lmms-lab-encoder/GQA`](https://huggingface.co/datasets/lmms-lab-encoder/GQA) | Balanced train/validation plus balanced test/testdev |
| MMBench | [`HuggingFaceM4/MMBench`](https://huggingface.co/datasets/HuggingFaceM4/MMBench) | Labelled validation evaluation and unlabelled test inference |
| MME | [`lmms-lab-encoder/MME`](https://huggingface.co/datasets/lmms-lab-encoder/MME) | Labelled paired yes/no evaluation |

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
processed schema. Run all three labelled benchmark splits with:

```powershell
uv run python scripts/run_baseline.py --dataset gqa_testdev_balanced
uv run python scripts/run_baseline.py --dataset mmb_validation
uv run python scripts/run_baseline.py --dataset mme_test
```

Results are written to `outputs/baseline/<dataset>.jsonl`. Every record contains
the sample/dataset identity, question, reference answers, model prediction,
benchmark metadata, latency, input/output token counts, peak VRAM when CUDA is
available, and `pruning_method: "none"`. `exact_match` is populated directly
for GQA; MMBench and MME are scored later with their benchmark-aware parsers.

The baseline uses the standard Qwen3-VL Transformers path: an
`AutoProcessor`, `Qwen3VLForConditionalGeneration`, `apply_chat_template`, and
`generate`. The measured report uses the full BF16 checkpoint, greedy decoding,
KV cache, and at most 32 newly generated tokens. It keeps all visual tokens and
records `pruning_method: "none"`.

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

## Completed baseline results

The current generated report was produced from complete runs on the three
labelled evaluation splits. These values describe this repository's recorded
run and hardware; latency and VRAM should not be treated as universal model
specifications.

| Benchmark | Rows | Accuracy | MME score | Mean latency | P95 latency | Mean input tokens | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GQA testdev balanced | 12,578 | 59.18% | — | 0.1549 s | 0.2101 s | 294.1 | 4.09 GiB |
| MMBench validation | 4,329 | 83.90% | — | 0.2371 s | 0.4220 s | 228.3 | 4.09 GiB |
| MME test | 2,374 | 82.06% | 2045.13 | 1.6062 s | 12.6834 s | 1,790.4 | 7.56 GiB |

The three accuracy values must not be averaged: GQA is open-ended VQA,
MMBench is multiple choice, and MME combines yes/no question accuracy with
paired consistency. MME's total score is a sum of category scores and is not a
percentage. Definitions and rationale for every quality and efficiency metric
are recorded in the generated
[`report.md`](outputs/reports/baseline/qwen3-vl-2b-instruct-full-token-bf16/report.md).

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
├── failure_cases.html
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
modification times, and generated artifact list. `report.md` explains the
metric definitions, why they are used, and how to interpret each figure.
Existing report directories are protected from accidental replacement;
regenerate the same report explicitly with:

```powershell
uv run python scripts/visualize_benchmarks.py --overwrite
```

Use a different `report.id` and `report.output_dir` for future pruning methods
instead of mixing them into this full-token baseline directory.

### Inspect incorrect answers

`failure_cases.html` is a self-contained, interactive error-analysis report.
It displays a deterministic, category-diverse sample of incorrect predictions
with:

- the original image;
- dataset, split, category, sample ID, and image ID;
- question, context/hint, and multiple-choice options when applicable;
- ground-truth answer beside the model answer;
- input/output token counts and generation latency;
- dataset/category filters, text search, and image zoom.

The current gallery contains 12 selected errors per benchmark. The complete
prediction files contain 5,134 GQA errors, 697 MMBench errors, and 426 MME
errors. Selection is round-robin across categories so the gallery is not
dominated by the first category in source order. Images are resized and
embedded as data URIs, making the approximately 2 MiB HTML file portable and
viewable offline without `data/raw/`.

Generate only the HTML gallery with:

```powershell
uv run python scripts/generate_failure_gallery.py
```

Adjust `failure_gallery.max_cases_per_dataset`, image width, JPEG quality, or
canonical source paths in `configs/report_baseline.yaml`. Running
`visualize_benchmarks.py --overwrite` regenerates the gallery together with the
rest of the report and lists it in `manifest.json`.

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

Every future experiment should identify the token budget, pruning location,
scoring signal, and any spatial safeguard. Report answer quality separately
from retained visual-token ratio, latency, throughput, VRAM, and an explicit
compute/FLOPs proxy where measurable. Random-K, uniform-K, and question-aware
fixed-K should be retained as controls rather than comparing only against the
full-token model.

## Tests and validation policy

Run the complete CPU-only unit-test suite with:

```powershell
uv run pytest -q
```

The current suite contains 18 tests covering dataset configuration and files,
canonical preprocessing, benchmark scoring, structured report generation, and
the self-contained failure gallery. Tests use small fixtures and do not load
Qwen model weights or download benchmark datasets.

For expensive work, use this progression:

1. Run unit tests.
2. Preprocess with `--limit 10`.
3. Run one baseline dataset with `--limit 10`.
4. Inspect the JSONL and score output.
5. Run complete inference only after the smoke path is correct.
6. Generate the structured report and inspect both aggregate metrics and
   individual failure cases.

The repository setup does not download datasets, preprocess full corpora,
download model weights, or start GPU inference automatically. Use the commands
above explicitly when ready. Before publishing final thesis results, also pin
the Hugging Face dataset/model revisions and record GPU, driver, Transformers,
Git revision, prompts, and generation configuration.

## Working with project agents

Repository-specific guidance lives in [`AGENT.md`](AGENT.md) and `.agents/`.
The local skills cover uv/Python 3.12, Hugging Face datasets and CLI use,
evaluation, memory planning, testing, Git workflow, VLM research, and research
decision records. These instructions keep generated experiments reproducible
while the final pruning method remains undecided.
