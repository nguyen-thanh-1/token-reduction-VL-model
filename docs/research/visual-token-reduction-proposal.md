# Visual-token reduction for Qwen3-VL-2B: literature review and proposal

- Status: Proposed (input for a future ADR; nothing here is locked)
- Date: 2026-10-21
- Scope: thesis method selection, with a publishable contribution as a stretch goal

## 1. Goal

Feed fewer visual tokens to the language model of `Qwen/Qwen3-VL-2B-Instruct`
while matching, or improving on, the full-token baseline's answer quality on
GQA testdev, MMBench validation, and MME test. Quality and efficiency are
reported separately (accuracy vs. retained tokens, prefill FLOPs, latency,
peak VRAM).

## 2. What the baseline already tells us

| Benchmark | Mean input tokens | Mean latency | P95 latency | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| GQA testdev | 294.1 | 0.155 s | 0.210 s | 4.09 GiB |
| MMBench val | 228.3 | 0.237 s | 0.422 s | 4.09 GiB |
| MME test | 1,790.4 | 1.606 s | 12.683 s | 7.56 GiB |

1. **The token count is set by image resolution, not by the model.** With
   `patch_size=16`, `merge_size=2`, and `max_pixels=16,777,216`, a high-
   resolution MME image yields several thousand visual tokens while a GQA image
   yields a few hundred. Reduction matters most on high-resolution inputs. The
   cheapest reducer of all is lowering `max_pixels`, so **every pruning result
   must be compared against a resolution-matched baseline at the same
   `N_visual`**. Almost no paper reports this control.
2. **The ViT costs about as much as the LLM per visual token.** Counting only
   linear layers, the 24-layer ViT costs about 1.21 GMACs per merged visual
   token (4 patches × 24 layers × width 1024). The 28-layer LLM costs about
   1.41 GMACs per token in prefill. Pruning *after* the ViT therefore removes at
   most about 54% of the visual-token prefill compute on this 2B model, and
   none of the ViT time. Reports must separate ViT time, LLM prefill time, and
   decode time. Claims such as "x% fewer tokens" should not be presented as
   end-to-end speedups.
3. The MME P95 latency (12.7 s) means a few very large images dominate the
   cost. Per-image adaptive budgets would help most there.

## 3. Where tokens can be reduced in Qwen3-VL (mapped to the diagram)

The block numbers refer to
`architecture/model-goc-baseline/Qwen3-VL-2B-Instruct-architecture.drawio`.
Line numbers refer to the installed
`.venv/Lib/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
(Transformers 5.x).

| Insertion point | Diagram block | Saves | Qwen3-VL-specific constraints |
| --- | --- | --- | --- |
| P0. Processor (resize, `max_pixels`) | 1. Input + processor | ViT + LLM | None; trivial, but it is the control to beat |
| P1. Inside the ViT (token merging/dropping between blocks) | 2. Vision encoder | ViT + LLM | 2D vision RoPE, the 2×2 merge needs complete 2×2 groups, DeepStack taps at ViT blocks 5/11/17 |
| P2. After the PatchMerger, before fusion | 2 → 3 (`get_image_features`, L1037; `masked_scatter`, L1189–1194) | LLM only | Must shrink `<|image_pad|>` count, the **three DeepStack streams** (L1225–1226), and **M-RoPE positions** (`get_rope_index`, L929) consistently |
| P3. Inside the LLM after layer k | 4. Language decoder | LLM layers ≥ k | DeepStack is added after decoder layers 0, 1, 2 (L837–858). Pruning at k ≥ 3 avoids DeepStack entirely; pruning at k < 3 must also slice the DeepStack features |

Your intuition is right: blocks 1–3 matter most. P2 is the natural place for a
method, and it is also where Qwen3-VL breaks the assumptions of methods
designed for LLaVA:

- **No [CLS] token.** VisionZip, PruMerge, and FasterVLM-style scores need
  [CLS] attention. Qwen's ViT has none. Its last layers also use SDPA/flash
  attention, which does not return attention maps without an eager fallback.
- **DeepStack.** Each kept visual position receives three extra residual
  features from ViT blocks 5, 11, and 17 at decoder layers 0–2. A pre-LLM
  pruner must decide what happens to the DeepStack features of pruned tokens.
- **M-RoPE.** Positions are 3-D (t, h, w) coordinates on the merged grid. Kept
  tokens should keep their *original* grid coordinates. Re-indexing them as a
  smaller dense grid silently changes the spatial layout. `rope_deltas` must
  then be recomputed for decoding.

## 4. Literature map

I read the abstracts below through the arXiv API. For the Qwen3-VL papers I
also searched the full HTML text for DeepStack and M-RoPE handling. I have not
read these papers in full; check the method sections before citing details.

### 4.1 Foundational methods (all are standard comparison baselines)

| Family | Method | arXiv | Where | Signal |
| --- | --- | --- | --- | --- |
| Merging | ToMe | 2210.09461 | ViT | bipartite similarity merging |
| In-LLM | FastV | 2403.06764 | LLM layer 2 | text→image attention |
| In-LLM | SparseVLM | 2410.04417 | LLM, progressive | text-guided, token recycling |
| In-LLM | PyramidDrop | 2410.17247 | LLM, staged | attention, pyramid schedule |
| Pre-LLM | LLaVA-PruMerge | 2403.15388 | after ViT | [CLS] attention + IQR, merge |
| Pre-LLM | VisionZip | 2412.04467 | after ViT | dominant + contextual merge |
| Pre-LLM | VisPruner | 2412.01818 | after ViT | visual cues + diversity |
| Pre-LLM | DivPrune | 2503.02175 | after ViT | max–min diversity, attention-free |
| Pre-LLM | DART | 2502.11494 | after ViT / LLM | duplication with pivots |
| Pre-LLM | CDPruner | 2506.10967 | after ViT | instruction-conditioned DPP |
| Learned | Matryoshka MM | 2405.17430 | projector | nested pooled scales |
| Learned | LLaVA-Mini | 2501.03895 | projector + prefusion | 1 visual token |
| Resolution | VisionThink | 2507.13348 | input | RL decides to request high-res |
| Architecture | DeepStack | 2406.04334 | LLM layers | multi-level residual injection |

### 4.2 Critiques that should shape our protocol

- **2502.11501 – "Are we solving the right problem?"** Many attention-based
  pruners lose to random selection or pooling; attention scores carry
  positional bias; duplication matters more than "importance"; current
  evaluation protocols are biased.
- **2412.13180 – FEATHER.** FastV-style pruning keeps tokens near the bottom of
  the image because of attention position bias, which hurts localization.
- **2512.07580 – "When token pruning is worse than random".** Beyond an
  "information horizon" in the LLM, visual tokens carry almost no information.
  The horizon is deeper for OCR and for stronger models (Qwen2.5-VL), and
  DivPrune + random pruning works well.
- **2608.00077 – Spatial provenance audit (Qwen3-VL-8B).** Accuracy at a fixed
  budget hides whether the kept tokens actually cover the evidence. They
  recommend paired confidence intervals, matched random/grid controls, and
  reporting real (not nominal) cost.
- **2610.23974 – "How much can we prune?" (Qwen3-VL-4B).** Offline greedy
  search finds subsets far better than current pruners, so there is
  significant headroom for better selectors.

### 4.3 Recent work on Qwen3-VL (2026; the area is crowded)

| arXiv | Method | Type | Takeaway for us |
| --- | --- | --- | --- |
| 2605.29657 | OccamToken | training-free, adaptive budget | register-anchored relative tests; attention sinks distort rankings |
| 2608.01979 | ET-Prune | training-free, adaptive budget | evidence-aware token floor; text-rich focus |
| 2608.07088 | RoRA | training-free, fixed budget | regional core/context/detail allocation |
| 2608.04132 | RUTA | trained gates + anchors | learned per-pair budget; 4.2% tokens, 94.4% perf. on Qwen3-VL-8B |
| 2606.00543 | ETC | trained distillation | Qwen3-VL-2B, down to 1 token |
| 2609.34942 | TCRR | trained resolution router | resolution as a per-query decision; −40.9% visual FLOPs on Qwen3-VL-8B |
| 2606.31903 | Operator-level skipping | training-free | skip attention/FFN for visual tokens instead of deleting them |
| 2610.12060 | ViMoD | light training | coarse context plus on-demand fine tokens during reasoning |
| 2609.34977 | SPIDER | training-free | intermediate ViT-layer tokens help select tokens |
| 2609.37581 | TReVS | training-free | adding query guidance to stage-1 pre-LLM pruning helps |

### 4.4 The gap

- **DeepStack is unaddressed.** None of the nine Qwen3-VL papers whose full text
  I could retrieve mentions DeepStack (I checked all except 2610.23974, which
  has no HTML version yet). How a pre-LLM pruner should treat the three
  DeepStack streams (slice, drop, or reuse them) has not been studied.
- **Free multi-level features are unused.** SPIDER and STD (2609.25635) show
  that intermediate ViT layers carry useful selection signals. Qwen3-VL already
  computes and projects three intermediate levels (DeepStack) into the LLM
  width, so they can be used for scoring at no extra cost.
- **Resolution-matched controls are almost never reported.** TCRR treats
  resolution as the method itself, but pruning papers rarely compare against
  "resize the image to the same token count".
- **Evidence from small models is thin.** Most results are on 7–8B models. At
  2B, the ViT share of compute is large (§2), which changes which reductions
  pay off.

## 5. Proposed method: DeepStack-aware Select-and-Fold (working name "DSFold")

The method is training-free and plugs in at P2. It keeps the
`VisualTokenPruner` interface in `src/token_reduction_vl/pruning/base.py`.

### 5.1 Select: which tokens to keep (budget K)

Each component is scored separately so that ablations can isolate it:

1. **Coverage/diversity term.** Max–min diversity (DivPrune) or a DPP kernel
   (CDPruner) over the final PatchMerger outputs `[N_visual, 2048]`. This is
   the robust core according to 2502.11501.
2. **Query relevance term.** Cosine similarity between visual tokens and the
   question's text-token embeddings in the LLM embedding space. These are
   available before the LLM runs. A stronger but more expensive alternative is
   a partial first-layer QK pass, as in ET-Prune. Report both, with their cost.
3. **Multi-level saliency term (new, Qwen3-VL-specific).** Use how much each
   token's DeepStack features change across the three levels
   (‖f₁₇ − f₁₁‖ + ‖f₁₁ − f₅‖) together with its final-level norm. The
   hypothesis is that object and text regions keep evolving across depth while
   flat background stabilizes early. This costs nothing extra, because the
   features already exist.

Combine the terms into a conditional DPP / greedy objective
(quality = relevance × saliency, similarity = cosine on final features). Keep
original M-RoPE coordinates for every kept token.

### 5.2 Fold: keep information from pruned tokens without adding tokens (new)

DeepStack gives each kept position three additive residual "slots" at decoder
layers 0–2. Instead of discarding pruned tokens:

- Assign each pruned token to its nearest kept token (feature cosine plus
  spatial distance on the merged grid).
- **Fold the pruned tokens' DeepStack features** into the kept token's DeepStack
  residuals using a similarity-weighted average, normalized so the residual
  scale matches the original.
- Optionally merge the main-stream features as well (as in VisionZip/ToMe).
  Run this as an ablation: main-stream merge, DeepStack-only fold, both, or
  neither.

Why it might work: DeepStack was trained to inject fine-grained, lower-level
detail at visual positions. Using that channel to carry evidence from pruned
neighbours adds detail without lengthening the sequence. This is the part most
likely to be publishable, and it is cheap to test.

### 5.3 Budget

The main experiments use fixed budgets so that comparisons are fair. Adaptive
budgeting is an extension: choose K per image from the spread of the scores
(for example, the K that covers a fixed fraction of the DPP log-determinant
gain), similar in spirit to OccamToken/ET-Prune. Report the realized average
retained tokens.

### 5.4 Hypotheses to test

- **H1.** Slicing DeepStack features consistently with the kept indices beats
  dropping DeepStack and beats feeding unsliced or misaligned features.
- **H2.** Keeping original M-RoPE coordinates beats re-indexing as a dense grid,
  especially on spatial GQA questions.
- **H3.** Folding pruned tokens into DeepStack improves quality over pure
  selection at aggressive budgets (≤ 22.2% retained).
- **H4.** The multi-level saliency term adds value on top of diversity alone.
- **H5.** At equal `N_visual`, pruning beats resolution reduction. This is not
  guaranteed, and it must be measured. If it fails, that negative result is
  itself a reportable finding for small models.

## 6. Experimental protocol

**Controls (required):** full token; resolution-matched (set `max_pixels` to
reach the same `N_visual`); random-K (3 seeds); uniform spatial grid-K; 2×2
average pooling.

**Published baselines to reimplement on Qwen3-VL-2B:** FastV (P3, k = 2 and
k = 3), DivPrune, CDPruner, DART, and VisionZip adapted without [CLS] (state
the adaptation).

**Budgets:** retention ratios {100%, 50%, 33.3%, 22.2%, 11.1%, 5.6%} of `N_visual` per image (33.3–5.6% match LLaVA 192/128/64/32 of 576),
plus absolute caps {64, 128, 256} tokens for comparison with LLaVA-style
papers.

**Metrics:** benchmark accuracy as today (GQA accuracy, MMBench accuracy, MME
score + Accuracy+). Also retained visual tokens (realized), analytical prefill
FLOPs (ViT and LLM separately), measured ViT / prefill / decode latency, and
peak VRAM. Report paired bootstrap 95% CIs against full token, with the image
as the resampling unit.

**Benchmarks:** keep the current three. For a paper, add at least one
text-rich benchmark (TextVQA or OCRBench) and POPE (hallucination), because
those are where pruning fails and reviewers expect them. This needs new
adapters plus an ADR update to the data protocol.

**Analyses:** retention maps on failure cases (extend the existing failure
gallery), a per-category breakdown on MME, and a check of position bias in
which tokens get kept (compare with FEATHER).

## 7. Implementation plan in this repository

1. `pruning/` – implement pruners against `VisualTokenPruner` (`RandomK`,
   `GridK`, `Pool2x2`, `DivPrune`, `DSFold`, ...). Extend `PruningOutput` with
   per-token DeepStack outputs and kept grid coordinates.
2. `models/qwen3_vl_pruned.py` – a wrapper that:
   - calls `model.model.get_image_features(...)` to get main and DeepStack
     features;
   - calls the pruner and rebuilds `input_ids` with K `<|image_pad|>` tokens;
   - computes full-sequence M-RoPE `position_ids` from the *original* grid,
     then gathers the kept columns;
   - slices or folds the DeepStack lists, builds `inputs_embeds`, and runs
     prefill through `model.model.language_model`;
   - runs its own greedy decode loop (≤ 32 tokens, with KV cache and
     `rope_deltas` recomputed from the pruned sequence).
   A hand-written loop avoids fighting `generate()`'s position-id logic and is
   easy to verify. It must reproduce the baseline outputs exactly when
   `NoOpPruner` is used. This is the first test to write.
3. Instrumentation – CUDA-event timing of the ViT, pruner, prefill, and decode
   stages; analytical FLOPs counter.
4. Configs – `configs/prune_<method>.yaml`. Each method/budget gets its own
   `report.id` and output directory. The baseline report is never overwritten.
5. Tests – NoOp equivalence (CPU, tiny random-init config if feasible),
   index/position bookkeeping, and fold weight normalization. No model weights
   in unit tests.
6. ADR 0002 – record the chosen method, insertion point, and protocol once the
   pilot results (milestone M2) are in.

## 8. Milestones

| # | Deliverable | Exit criterion |
| --- | --- | --- |
| M1 | Pruned-inference wrapper + NoOp equivalence + stage timing | identical predictions to baseline on 100 samples |
| M2 | Controls (resolution, random, grid, pool) at 4 budgets | first quality–cost Pareto plot |
| M3 | FastV, DivPrune, CDPruner on Qwen3-VL-2B | reproduced trends from papers |
| M4 | DeepStack study (H1, H2) | answer to "how to prune with DeepStack" — paper core #1 |
| M5 | DSFold select + fold (H3, H4) | beats best baseline at ≤ 22.2% retained, CI excludes 0 |
| M6 | Text-rich + POPE benchmarks, adaptive budget | generality |
| M7 | Write-up | thesis chapters; workshop/conference submission |

## 9. Paper strategy and risks

- **Realistic contribution:** (1) the first systematic study of visual-token
  pruning under DeepStack multi-level injection, with clear guidance on
  slicing, positions, and resolution-matched controls; (2) the Fold mechanism,
  which reuses DeepStack as a free side channel for pruned-token evidence.
  Item (1) has value even if (2) shows only modest gains. Efficiency workshops
  (CVPR/ICCV/NeurIPS) are the safe target; a main conference needs (2) to win
  clearly and to generalize to Qwen3-VL-4B/8B.
- **Risk: crowded field.** Do not pitch "another scoring heuristic". Pitch the
  DeepStack question and rigorous controls.
- **Risk: small speedups at 2B.** The ViT is about 46% of visual compute.
  Mitigation: report stage-wise timing honestly; combine with mild resolution
  reduction (P0) for end-to-end gains; add P1 (in-ViT merging) as a later
  extension.
- **Risk: field is moving fast.** Re-run the arXiv DeepStack search before each
  milestone; the gap can close.
- **Alternative if Fold fails:** a training-free coarse-to-fine cascade. Answer
  at low resolution first, and re-run with high resolution plus pruning only
  when the answer token's probability is low. This relates to VisionThink and
  TCRR but needs no training.

## 10. References (arXiv IDs)

2210.09461 ToMe · 2403.06764 FastV · 2403.15388 LLaVA-PruMerge ·
2405.17430 Matryoshka Multimodal Models · 2406.04334 DeepStack ·
2410.04417 SparseVLM · 2410.17247 PyramidDrop · 2412.01818 VisPruner ·
2412.04467 VisionZip · 2412.13180 FEATHER · 2501.03895 LLaVA-Mini ·
2502.11494 DART · 2502.11501 Are We Solving the Right Problem? ·
2503.02175 DivPrune · 2506.10967 CDPruner · 2507.13348 VisionThink ·
2511.21631 Qwen3-VL Technical Report · 2512.07580 Worse than Random ·
2603.27960, 2604.05546 surveys · 2605.29657 OccamToken · 2606.00543 ETC ·
2606.31903 Operator-level skipping · 2608.00077 Spatial provenance audit ·
2608.01979 ET-Prune · 2608.04132 RUTA · 2608.07088 RoRA ·
2609.25635 STD · 2609.34942 TCRR · 2609.34977 SPIDER · 2609.37581 TReVS ·
2610.12060 ViMoD · 2610.23974 How Much Can We Prune?
