# ADR 0005: External baselines for the PP3 comparison

- Status: Proposed
- Date: 2026-11-03

## Context and decision drivers

Every PP3 result so far is compared with full tokens, random selection and
PP3's own variants. A paper also needs published selection methods run on the
same model, splits and budgets. They must fit the shared pruner interface
(selection between the ViT and the LLM) so that only the selection differs.

## Decision

Two baselines, both at the shared budget levels and without folding (as
published; `gamma = 0`), with a folded variant only as an ablation:

1. **DivPrune** (Alvar et al., CVPR 2025): greedy max–min diversity on the
   cosine distance of the visual tokens fed to the LLM (`v`), no quality term.
   Implemented with DSFold's `weighted_maxmin` with unit quality; our choice
   for the first token is the one with the largest mean distance to the
   others (check against the reference code before the final runs).
2. **FastV-style** (Chen et al., ECCV 2024): rank visual tokens by the mean
   attention they receive from all later positions, averaged over heads, in
   LLM layer 2 (index 1) of the full-token prefill; keep the top K. FastV
   prunes inside the LLM after layer 2; here the kept set is fed to the LLM
   from the start, so the first two layers also see only K tokens. The paper
   reports this as "FastV-style selection at the input".

3. **CDPruner** (Zhang et al., NeurIPS 2025, arXiv:2506.10967), added
   2026-11-04 after review: the closest prior work, a training-free
   instruction-conditioned DPP. For models without a paired text encoder it
   uses the projector output `v` and the mean LLM embedding of the
   instruction: `r_i = cos(v_i, mean_w e_w)`, min-max normalised to `r̃`, and
   greedy MAP on `diag(r̃)·S·diag(r̃)` with `S_ij = cos(v_i, v_j)` (their
   Eq. 5–8, no balance factor θ). The instruction is the question plus option
   texts (`question_text_for_scoring`). It shares PP3's greedy DPP, including
   the weighted max–min fallback above `n_cap = 4096` tokens, and no fold, so
   PP3 vs CDPruner isolates the learned quality term. Check it
   against the reference code before the camera-ready version.

Runs are named `divprune_L*`, `fastv_L*` and `cdpruner_L*` under `outputs/pp3_scorer/runs/`.

## Consequences

- The FastV variant costs one extra partial prefill with eager attention;
  its selector time is reported but is not FastV's real cost.
- Further baselines (VisionZip, SparseVLM) are added only if time allows;
  VisionZip needs a [CLS]-like ViT attention that Qwen3-VL does not have.
