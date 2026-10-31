# PP3 — DSFold + learned score head

This directory shows the implemented PP3 method (`pp3-dsfold-scorer/`) on the
`Qwen/Qwen3-VL-2B-Instruct` model.

- Editable source: `Qwen3-VL-2B-Instruct-PP3-scorer.drawio`
- Derived from: `../dsfold-proposal/Qwen3-VL-2B-Instruct-DSFold-proposal.drawio`
  (itself a copy of `../model-goc-baseline/Qwen3-VL-2B-Instruct-architecture.drawio`)
- Method write-up: `pp3-dsfold-scorer/method.md`; shared DSFold steps: `pp1-dsfold/method.md`
- Code: `src/token_reduction_vl/pruning/pp3_scorer/`
- Image exports: `exports/`

Colour code: blue = frozen Qwen3-VL, red = DSFold/PP1 steps kept unchanged
(parameter set D-1.0), purple = new in PP3.

Changes relative to the DSFold copy:

- Section 2½ now reflects the implemented pipeline: S1 is replaced by the
  score head (`q̂ = softplus(s) + 1e-3`), S2–S4 keep set D-1.0, and the budget
  box lists the seven fixed levels L1–L7.
- Panel **D** step 2 notes the PP3 quality term.
- Panel **E**: score head 3a architecture (fixed PCA → TokenEncoder → 979-d
  token/question interaction → per-token MLP) and the inference flow into the
  DPP, fold, and frozen LLM, with tensor shapes, cost, and the 3b/3c variants.
- Panel **F**: route-B training flow (GQA train 50k → gradient×input saliency
  → PCA-compressed shards → ListNet → early stopping on held-out images).
- Panel **G**: GQA dev (1,020 samples) results at L3–L6 against PP1 set D and
  the full-token baseline. Update it when new results replace these.

Keep the baseline and DSFold diagrams unchanged; edit this copy when PP3 changes.
