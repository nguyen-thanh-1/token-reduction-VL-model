# ADR 0002: PP3 tuning split and non-inferiority protocol

- Status: Proposed
- Date: 2026-10-31

## Context and decision drivers

PP3 (DSFold + learned score head 3a) was compared with the full-token baseline
on `dev_1k_*`. Those ids are drawn from GQA testdev, MMBench validation and
MME test, which `configs/data_v1.yaml` reserves for evaluation. The 3a/3b/3c
choice was already made on them, so further tuning there would bias the final
numbers.

With about 1,000 paired samples the 95% interval of an accuracy difference is
about ±1.6 points, too wide to show a 1-point margin. The thesis targets are:

- minimum: accuracy no worse than the full-token baseline by more than 1 point
  (absolute) on each benchmark;
- stretch: accuracy above the baseline by removing distracting tokens.

## Decision

1. **Tuning split.** Every remaining PP3 choice (selection knobs, fold, label
   type, checkpoint) is made on `configs/splits/gqa_val_tune_2k.txt`
   (GQA val balanced), compared with a full-token run on the same ids.
2. **Development peek.** `dev_1k_*` results are kept as already seen. They may be
   used for sanity checks on MMBench/MME but not for choosing parameters.
3. **Final evaluation.** The chosen configuration is run once on the full GQA
   testdev-balanced, MMBench validation and MME test. Benchmarks are reported
   separately; MME score is reported next to accuracy.
4. **Non-inferiority test.** For each benchmark, the paired difference
   `acc(method) − acc(full)` is non-inferior when the one-sided 95% lower bound
   is above −1 point. Superiority additionally needs an exact McNemar
   p < 0.05 with more gained than lost answers.
5. **Diagnostics before new training** (stage 1, on the tuning split):
   random selection at the same K, random drop of 10%/25% (noise floor),
   drop of the 10%/25% lowest-score tokens (denoising test) and a one-token
   near-blind run (language prior).
6. Quality and efficiency are reported separately; latency claims wait for a
   faster selector.

## Consequences, risks, and validation

The final numbers are not influenced by tuning, at the cost of one extra
full-token run on the tuning split. GQA-only tuning may not transfer to MMBench
and MME; that is reported, not tuned away. Labels computed from the model's own
answer with absolute saliency cannot express harmful tokens, so the stretch goal
likely needs signed, ground-truth-answer labels (a later ADR if stage 1 supports
it).
