# ADR 0004: Focus + context selection and region metrics for PP3

- Status: Proposed
- Date: 2026-11-01

## Context and decision drivers

The thesis has two goals for a pruned model:

1. fewer visual tokens with accuracy at or above the full-token baseline;
2. a kept set that makes sense for the question: the region the question is
   about (or its surroundings), compact rather than scattered, and general
   across question types and benchmarks.

ADR 0003's attention-distilled head 3d meets goal 1 within noise on tune2k
(L4 −0.20 points, one-sided LB −0.99) but not goal 2. New region metrics
(`src/token_reduction_vl/reporting/region_metrics.py`,
`scripts/pp3/region_report.py`) on tune2k show:

- every head's kept set is as fragmented as random selection (L3: 3.3–3.8
  connected groups per 10 kept tokens vs 3.76 for random), because DPP spreads
  tokens on purpose;
- 3d_attn50k keeps 65% (L3) / 72% (L4) of the same tokens for different
  questions on the same image (random: 13% / 20%): it follows attention sinks
  (sky, background), not the question.

For an object that is absent from the image there is no target region; then
broad, systematic coverage is the right behaviour, but scattering that is the
same for every question is not.

## Decision

1. **Region metrics are reported with accuracy** for every PP3 candidate:
   groups per 10 kept tokens, kept-neighbour share, coverage of 3×3-token
   cells, and question overlap on the same image; split by yes/no vs other
   answers. Grounding against GQA object boxes is added once the scene graphs
   are available (separate download, needs approval).
2. **Contrastive focus target.** From the existing attention labels, for a
   sample on an image with other training questions:
   `t ∝ max(0, p − mean of p over the image's other questions)`. Sinks shared
   by all questions cancel. A 3d head (`3d_contrast`) is trained on `t`; 98.6%
   of the 50k labels have sibling questions.
3. **Focus + context selector** (`pp3_scorer/focus_context.py`) replaces DPP:
   - focus: greedy region growing on the focus score with an adjacency bonus
     `beta` (in standard deviations);
   - context: one token per coarse cell by the global head (3d_attn50k),
     skipping cells already covered by focus;
   - the focus share of K moves between `share_min` and `share_max` with how
     peaked the focus distribution is, so a flat map (nothing matches) falls
     back to coverage;
   - dropped tokens are folded as before.
4. **Selection** follows ADR 0002 (tune2k, then confirmation on a second val
   split), choosing on the accuracy / region-metric Pareto front, not on one
   number. Knobs: `beta`, `share_min`, `share_max`.

## Consequences

- A selector that is compact by construction may lose accuracy on questions
  that need several distant regions; the multi-seed growing and the context
  part are the mitigations, and the paired report shows the cost.
- The focus head inherits the teacher's biases minus the shared part; if the
  contrast is too sparse to learn, the fallback is GQA object boxes as focus
  supervision.
