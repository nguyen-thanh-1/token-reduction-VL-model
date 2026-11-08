# ADR 0006: Additional evaluation benchmarks — POPE and ScienceQA-IMG

- Status: Accepted
- Date: 2026-11-04

## Context and decision drivers

The PP3 paper is evaluated on GQA testdev, MMBench dev and MME. Reviewers of
visual-token pruning papers expect two more axes that these three do not cover:

- **Hallucination.** Dropping tokens may make the model claim objects that are
  not there (or deny ones that are). Almost every recent pruning paper
  (FastV, SparseVLM, VisionZip, DivPrune, CDPruner) reports POPE.
- **Reasoning with long context.** ScienceQA-IMG has long textual hints and
  diagrams; it checks that a question-conditioned selector still works when
  most of the prompt is context, not the question itself.

Both must be added without changing the training data, the tuning protocol
(ADR 0002) or any reported result.

## Decision

1. **POPE** (Li et al., EMNLP 2023), Hugging Face `lmms-lab-encoder/POPE`,
   config `default`, split `test` (9000 yes/no questions on 500 COCO val2014
   images; `category` = adversarial / popular / random, 3000 each). Revision
   pinned at download time and recorded in the raw manifest. Canonical
   `dataset: pope`, `task_type: binary_vqa`, `image_id = image_source`; the
   downloader stores each COCO image once (keyed by `image_source`).
   Metrics per category and overall: accuracy, F1 with "yes" as the positive
   class, precision, recall and yes-ratio. The headline number is the mean F1
   over the three categories, as in the POPE paper. An answer that is neither
   yes nor no counts as wrong (and as a "no" for F1, so it cannot inflate
   precision).
2. **ScienceQA-IMG** (Lu et al., NeurIPS 2022), `lmms-lab-encoder/ScienceQA`,
   config `ScienceQA-IMG`, split `test` (2017 questions with an image).
   Canonical `dataset: scienceqa`, `task_type: multiple_choice`, choices
   labelled A–E in source order, `answer_label` from the integer `answer`,
   the non-empty `hint` as `Context:`. Metric: accuracy (option-letter parse,
   exact option text as fallback, same as MMBench).
3. **Split roles.** Both are evaluation-only; their `validation` splits are
   not used. Nothing is tuned on them: every method runs with the
   configuration frozen on GQA val (ADR 0002). They are never averaged with
   GQA, MMBench or MME.
4. **Statistics.** Paired Wald CI and McNemar against full tokens, as for GQA
   and MMBench; POPE F1 differences with an image-level bootstrap (1000
   resamples), as for MME.
5. **Scope of methods.** Full, random, DivPrune, CDPruner, FastV-style and PP3
   at L1–L4; DSFold-D and PP3-fc only if time allows.

`configs/data_v1.yaml` lists `pope_test` and `sqa_img_test` under
`evaluation`; runs are named `<prefix>_<name>_pope/pope_test.jsonl` and
`<prefix>_<name>_sqa/sqa_img_test.jsonl`.

## Alternatives considered

- **TextVQA val** (OCR-heavy). Informative about the known weakness on text-rich
  images (MME cognition), but 5000 questions and a VQA-accuracy metric with ten
  references. Deferred; added as an appendix only if time allows.
- **MMStar / SEED-Bench.** Larger multiple-choice suites; overlap with MMBench
  in what they test.

## Consequences

- New download of about 0.5 GB of Parquet (POPE 255 MB, ScienceQA-IMG 263 MB
  for validation + test); only with the user's explicit OK. POPE keeps one
  file per COCO image (500 files) after export.
- `scripts/download_datasets.py --dataset all` keeps its old meaning (GQA,
  MMBench, MME); the new benchmarks are downloaded only with `--dataset pope`
  or `--dataset sqa`.
- GPU cost (measured, RTX 5060 Ti, 16 pruned configurations + full in one pass): POPE about 10 h (4.4 s per question; launch-bound, GPU about 20% busy), ScienceQA-IMG about 2 h. Two processes with disjoint `--only` lists roughly halve the wall time.
