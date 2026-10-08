# Qwen3-VL-2B-Instruct Full-Token BF16 Baseline

## Run identity

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Precision: `bf16`
- Pruning method: `none`
- Report schema: `1.0`

## Quality

| Benchmark | Labelled rows | Accuracy | MME score |
| --- | ---: | ---: | ---: |
| GQA testdev balanced | 12,578 | 59.18% | — |
| MMBench validation | 4,329 | 83.90% | — |
| MME test | 2,374 | 82.06% | 2045.13 |

The three accuracy values follow different benchmark protocols and are not
averaged into a single score.

![Quality overview](figures/01_quality_accuracy.png)

## Efficiency

| Benchmark | Mean latency (s) | P95 latency (s) | Mean input tokens | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: |
| GQA testdev balanced | 0.1549 | 0.2101 | 294.1 | 4.09 |
| MMBench validation | 0.2371 | 0.4220 | 228.3 | 4.09 |
| MME test | 1.6062 | 12.6834 | 1790.4 | 7.56 |

![Efficiency overview](figures/02_efficiency_overview.png)

![Latency distribution](figures/03_latency_distribution.png)

![Tokens versus latency](figures/04_tokens_vs_latency.png)

## MME category analysis

- Highest category score: `existence` (190.00).
- Lowest category score: `code_reasoning` (107.50).
- Total MME score: 2045.13.

![MME categories](figures/05_mme_categories.png)

## Artifact layout

- `summary.json`: machine-readable metrics and efficiency statistics.
- `tables/benchmark_summary.csv`: one row per benchmark.
- `tables/mme_categories.csv`: MME category-level values.
- `manifest.json`: report identity, input provenance, and generated files.
- `figures/`: deterministic numbered figures in reading order.
