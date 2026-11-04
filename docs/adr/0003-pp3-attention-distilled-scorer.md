# ADR 0003: PP3 score head distilled from the LLM's text-to-image attention

- Status: Proposed
- Date: 2026-11-01

## Context and decision drivers

ADR 0002's stage-3 pilot (tune2k, n = 2000) showed that the route-B gradient
labels limit the score head, not its architecture:

- 3a (MLP) and 3c (cross-attention to question tokens) reach the same
  held-out recall@K at L3 on |gradient × input| labels (0.536 vs 0.541);
- signed ground-truth labels are hard to predict (recall 0.341 against 0.279
  for the hand-crafted quality) and do not remove harmful tokens at
  inference (`3a_signed_lowdrop25` −0.30 points, oracle +3.2);
- more labels help: abs 10k → 50k improves L3 by about 1.45 points.

After 24 ViT layers a visual token also carries information from other image
regions, so "where the object is" and "which tokens the LLM reads" differ.
For accuracy the second is the right target. The full-token LLM states it
directly in its attention from text positions to visual positions.

## Decision

1. **Teacher.** For each training sample, run the full-token model once
   (teacher-forced on its own greedy answer, eager attention) and record, for
   LLM layers 8–19 (of 28), the attention from
   - *answer rows*: the last prompt position and the answer positions, and
   - *question rows*: the question-text positions,
   to the visual positions. Each head's row is renormalised over visual
   tokens, then heads, rows and layers are averaged. This gives two
   distributions over the `N` visual tokens, stored side by side in the
   existing shard format (`saliency` = answer rows, `saliency_q` = question
   rows). No backward pass is needed.
2. **Student 3d.** A text-guided head: a one-layer text encoder contextualises
   the question tokens, visual tokens get one self-attention and one
   cross-attention block (as 3c), and an explicit relevance term
   `logsumexp_w ⟨k_i, q_w⟩/√d` is added to the score, so a token can be ranked
   high because a specific word points to it.
3. **Loss.** Cross-entropy to the teacher distribution, optionally
   sharpened/flattened with `P ∝ p^(1/T)`. Target `mix` = 0.5 answer +
   0.5 question by default; `primary` and `question` are kept for ablation.
4. **Pilot gate G4** (tune2k, protocol of ADR 0002): train 3d on the first 10k
   GQA-train samples. Pass if `3d_attn_L3` beats `3a_L3` (the 50k abs head)
   head-to-head in point estimate and `3d_attn_L4` is not worse than `3a_L4`.
   If it passes, extend the labels to 50k and move to stage 2 (selection
   knobs) on the new head.

## Consequences

- The labels describe the model's own reading behaviour, including attention
  sinks (a few visual tokens with large attention regardless of the
  question). The training temperature and the mixed target are the first
  mitigations; the label script reports the top-1 mass to monitor sinks.
- Patch selection before the ViT is not pursued here: raw patches have no
  semantics and the ViT would see an out-of-distribution subset. It stays a
  later efficiency extension.
- The deployed selection (DPP + fold, ADR 0002 knobs) is unchanged; only the
  quality `q̂` comes from the new head.
