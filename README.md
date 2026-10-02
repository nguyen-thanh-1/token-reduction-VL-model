# Token Reduction VL Model

## Download datasets

The downloader uses the Hugging Face `datasets` library and stores each dataset
separately under `data/`:

```text
data/
├── GQA/
│   ├── train_balanced_images/
│   ├── train_balanced_instructions/
│   ├── val_balanced_image/
│   ├── val_balanced_instructions/
│   ├── test_balanced_images/
│   ├── test_balanced_instructions/
│   ├── testdev_balanced_images/
│   └── testdev_balanced_instructions/
├── MMB/
│   └── test/
└── MME/
    └── test/
```

Install the project dependencies with `uv`:

```powershell
uv sync
```

Download one dataset or all datasets:

```powershell
uv run python scripts/download_datasets.py --dataset gqa
uv run python scripts/download_datasets.py --dataset mmb
uv run python scripts/download_datasets.py --dataset mme
uv run python scripts/download_datasets.py --dataset all
```

By default, the downloaded datasets are saved in `data/`. Use
`--output-dir` to select another location. The GQA downloader intentionally
downloads only the balanced configurations listed above; it does not download
the `*_all_*` configurations.

For private or gated datasets, authenticate with Hugging Face before running:

```powershell
hf auth login
```
