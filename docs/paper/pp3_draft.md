# Attention-Distilled, Question-Aware Visual Token Selection for DeepStack Vision-Language Models

Draft v1 (2026-11-04). Version 1 of the paper: Qwen3-VL-2B-Instruct, GQA /
MMBench / MME. Section 5.1 holds the final test numbers (GQA testdev
balanced, MMBench validation, MME test; `outputs/reports/pp3_final/tables.md`);
Sections 5.2–5.5 are development results on GQA val and must not be reported
as test results. Timing in 5.6 is from a dedicated single-process run.

## Abstract

Vision-language models spend most of their input on visual tokens, many of
them redundant for a given question. We study visual-token selection between
the vision encoder and the language model of Qwen3-VL, whose DeepStack design
injects multi-level visual features into early decoder layers. We propose
**PP3**: (i) a light score head that predicts, from the visual tokens and the
question alone, how much the full-token language model attends to each token,
trained by distillation from its text-to-image attention; (ii) a
quality–diversity selection with a determinantal point process (DPP); and
(iii) a DeepStack fold that merges each dropped token's multi-level features
into its nearest kept token. Because the score head replaces the language
model's attention at inference time, selection needs no attention maps and
remains compatible with fused attention kernels. At equal token budgets PP3
outperforms random, DivPrune and FastV-style selection at every budget on
GQA, MMBench and MME perception. With 22% of the visual tokens it is
non-inferior to the full-token model on GQA testdev (−0.50 points, one-sided
bound −0.88 against a 1-point margin); with 33% it loses 0.10 points on GQA
and matches the full model's MME perception score (1517.6 vs 1511.6). The
margin over existing selectors grows as the budget shrinks: with 5.7% of the
tokens PP3 loses 4.0 points on GQA testdev, against 11.2 for DivPrune, 14.7
for random selection and 23.2 for a FastV-style selection. We further show that the
language model's attention is dominated by question-independent *attention
sinks*, introduce region metrics for the kept set (compactness and question
dependence), and propose a contrastive target that yields compact,
question-specific focus regions at no accuracy cost at 50% of the tokens.

## 1 Introduction

- Visual tokens dominate the input length of VLMs; the cost of the language
  model grows with it (prefill compute, KV cache).
- Existing selection methods either (a) use attention *inside* the language
  model (FastV and successors), which needs explicit attention weights and
  fixes the pruning layer, or (b) are question-agnostic (diversity- or
  encoder-attention-based, e.g. DivPrune, VisionZip).
- Qwen3-VL adds DeepStack: three intermediate ViT feature levels are added to
  the hidden states of decoder layers 0–2 at the visual positions. Dropping a
  token removes it from all four streams; no prior method handles this.
- Contributions:
  1. a light, question-aware score head that runs before the language model
     and predicts which visual tokens the language model will use, trained
     either from its text-to-image attention or from gradient saliency (the
     attention teacher is equally accurate with 50k labels and better with
     10k);
  2. DPP selection with a DeepStack fold for dropped tokens (the fold is
     neutral on GQA; reported as an ablation);
  3. an analysis of what the teacher attends to (attention sinks), region
     metrics for kept sets, and a contrastive focus target;
  4. a paired, non-inferiority evaluation protocol (McNemar, one-sided bound,
     1-point margin) on GQA, MMBench and MME.

## 2 Related work

- Token pruning inside the LLM: FastV, PyramidDrop, SparseVLM. `TODO(cite)`
- Pruning before the LLM / token merging: ToMe, VisionZip, DivPrune, LLaVA-PruMerge. `TODO(cite)`
- Learned token selection and distillation from attention. `TODO(cite)`
- Attention sinks / registers in transformers and VLMs. `TODO(cite)`

## 3 Method

### 3.1 Setting

The vision encoder and PatchMerger produce `N` visual tokens `v_i ∈ R^d`
on an `H'×W'` grid, and DeepStack features `f_i^(l)`, `l = 1..3`. Given a
keep ratio `r`, a selector keeps `K = ⌈rN⌉` tokens; the language model sees
the kept `v` at their original M-RoPE positions and the kept (folded)
DeepStack features. The shared levels are `r ∈ {1/18, 1/9, 2/9, 1/3, 1/2}`
(L1–L5).

### 3.2 Attention-distilled score head

**Teacher.** For each training question, the full-token model is run once,
teacher-forced on its own greedy answer, with eager attention in the language
model only. For decoder layers 8–19 (of 28) we read the attention from the
answer-predicting rows and from the question-text rows to the visual columns,
renormalise each head over visual tokens, and average over heads, rows and
layers. Target: `p = 0.5·p_answer + 0.5·p_question`.

**Label sources.** We compare this teacher with gradient × input saliency
labels (one backward pass per sample); Section 5.3 shows they are equivalent
at 50k labels and that the teacher is better at 10k.

**Student (3d).** Per-token features (`v`, the DeepStack levels, compressed
by a fitted PCA) pass through a token encoder, one self-attention layer over
visual tokens, and cross-attention to the question token embeddings
(contextualised by one transformer layer with word positions). The score is
`s_i = w·h_i + logsumexp_w ⟨W_k h_i, W_q q_w⟩/√d`. Loss: cross-entropy
between `softmax(s)` and `p` (temperature 1). Trained on 50k GQA train
questions, 20 epochs; 1,993,153 parameters (the language model has 1.72B). `TODO: training time.`

### 3.3 Quality–diversity selection and DeepStack fold

Quality `q̂_i = softplus(s_i) + 1e-3`. Greedy MAP inference for the DPP
kernel `L = diag(q̂) S diag(q̂)`, `S_ij = cos(v_i, v_j)`, picks `K` tokens.
Each dropped token `d` is assigned to the kept token
`a(d) = argmax_k cos(v_d, v_k) − λ·dist(d, k)/diag` (λ = 0.5), and for each
DeepStack level `f'_k = norm-match(f_k + γ Σ_{d∈A(k)} w_dk f_d)` with
`w_dk = softmax_d(cos(v_d, v_k)/τ)` (γ = 0.5, τ = 0.1); the main stream `v` is
not changed.

### 3.4 Focus + context (interpretable variant, PP3-fc)

**Attention sinks.** The teacher (and the head that imitates it) gives most
mass to background tokens (sky, walls) for every question on an image: kept
sets of different questions on the same image overlap with Jaccard 0.65 (L3)
/ 0.72 (L4), against 0.13 / 0.20 for random selection.

**Contrastive target.** For a training question on an image with other
training questions, `t ∝ max(0, p − mean_{other questions} p)` removes the
shared part; 98.6% of the 50k samples have such siblings. A second 3d head is
trained on `t`.

**Selection.** A share `ρ` of `K` (focus) is grown as connected regions on the
z-scored contrastive score with an 8-neighbour bonus `β`; the rest (context)
is chosen by the DPP over the remaining tokens with the global head's
quality. Default `ρ = 0.25`, `β = 1`.

## 4 Experimental setup

- Model: Qwen3-VL-2B-Instruct, bf16, greedy decoding, max 32 new tokens.
  One NVIDIA GeForce RTX 5060 Ti (16 GB), CUDA 13.0, PyTorch 2.14.1,
  Transformers 5.19.0, Python 3.12. `TODO: git SHA of the committed code`
  (runs record 398f742 with uncommitted changes).
- Data: GQA balanced (train for labels, val for development, testdev for
  test), MMBench validation, MME test. GQA test and MMBench test are not used.
- Protocol: all configurations decoded from the same ViT pass; paired
  comparison with the full-token model on the same questions; exact McNemar
  p, 95% CI, one-sided 95% lower bound against a 1-point non-inferiority
  margin. Benchmarks are never averaged; MME is reported as perception and
  cognition scores.
- Development: choices made on tune2k, confirmed on gqa_val_5k (ids disjoint
  from tune2k), then run once on the test sets.
- Baselines (same interface, no fold, as published): random (with fold),
  DivPrune (max–min diversity on `v`), FastV-style (top-K by attention
  received in LLM layer 2 of the full-token prefill, applied at the input).

## 5 Results

### 5.1 Main results

Generated by `scripts/pp3/final_report.py --prefix final`
(`outputs/reports/pp3_final/tables.md`). GQA testdev balanced (n = 12578),
MMBench validation (n = 4329), MME test (n = 2374); every configuration is
decoded from the same ViT pass as the full-token run. GQA and MMB: accuracy
difference to full in points with paired 95% CI; MME: perception (max 2000)
and cognition (max 800) scores. Benchmarks are not averaged.

| K/N | Method | GQA diff [95% CI] | MMB diff [95% CI] | MME perc. | MME cogn. |
|---|---|---|---|---|---|
| 1.00 | Full tokens | 59.18% | 83.90% | 1511.6 | 533.6 |
| 0.057 (L1) | Random | −14.71 [−15.57, −13.85] | −16.26 [−17.55, −14.97] | 1174.4 | 294.3 |
| | DivPrune | −11.18 [−12.00, −10.36] | −11.67 [−12.83, −10.50] | 1250.2 | 289.3 |
| | FastV-style | −23.19 [−24.13, −22.25] | −24.86 [−26.32, −23.39] | 984.3 | 304.3 |
| | **PP3** | **−4.04** [−4.67, −3.41] | **−9.84** [−10.98, −8.70] | **1385.3** | **323.2** |
| 0.11 (L2) | Random | −9.75 [−10.52, −8.97] | −9.75 [−10.87, −8.62] | 1330.3 | 337.1 |
| | DivPrune | −5.97 [−6.67, −5.27] | −6.01 [−6.99, −5.02] | 1400.5 | 324.6 |
| | FastV-style | −15.63 [−16.51, −14.76] | −10.63 [−11.77, −9.48] | 1290.5 | 331.1 |
| | **PP3** | **−1.88** [−2.43, −1.34] | **−4.53** [−5.43, −3.63] | **1469.8** | **358.9** |
| 0.22 (L3) | Random | −5.44 [−6.11, −4.77] | −4.46 [−5.33, −3.58] | 1495.0 | 333.9 |
| | DivPrune | −2.58 [−3.16, −2.00] | −3.03 [−3.83, −2.22] | 1462.2 | 389.3 |
| | FastV-style | −8.48 [−9.22, −7.75] | −4.97 [−5.84, −4.10] | 1415.0 | **426.8** |
| | **PP3** | **−0.50** [−0.96, −0.05] | **−1.71** [−2.40, −1.02] | **1502.4** | 397.1 |
| 0.33 (L4) | Random | −3.16 [−3.75, −2.57] | −2.49 [−3.25, −1.74] | 1471.5 | 440.0 |
| | DivPrune | −1.39 [−1.90, −0.88] | −1.41 [−2.08, −0.73] | 1467.4 | 445.4 |
| | FastV-style | −4.53 [−5.14, −3.92] | −3.16 [−3.89, −2.44] | 1489.4 | **454.6** |
| | **PP3** | **−0.10** [−0.48, +0.29] | **−0.97** [−1.58, −0.36] | **1517.6** | 453.9 |
| 0.50 (L5) | PP3-fc (ρ = 0.25) | −0.15 [−0.46, +0.16] | −0.21 [−0.69, +0.28] | 1533.7 | 492.5 |

Findings:

- **PP3 is the best selector at every budget on GQA, MMB and MME
  perception.** The margin to the best baseline grows as the budget shrinks:
  on GQA +7.14 points at L1, +4.09 at L2, +2.08 at L3, +1.29 at L4 (all over
  DivPrune).
- **Non-inferiority (one-sided 95% bound, 1-point margin).** GQA: L3
  −0.50 (bound −0.88) and L4 −0.10 (bound −0.42) pass; L2 does not (bound
  −2.34). MMB: L4 −0.97 (bound −1.48) does not pass; PP3-fc at L5 does
  (−0.21, bound −0.62). These agree with the development and confirmation
  splits (5.2, 5.2b), where L3 / L4 also passed on GQA.
- **MME perception** at L4 (1517.6) and PP3-fc at L5 (1533.7) match or
  exceed the full-token score with one third / one half of the tokens.
- **MME cognition** drops for every selector (full 533.6; best at L4 454.6).
  The loss sits in text-heavy subtasks (full → PP3 L3: code reasoning
  108 → 60, text translation 185 → 125, numerical calculation 118 → 95;
  OCR, a perception subtask, 125 → 118). Each cognition subtask has 40
  questions on 20 images, so one image moves a subtask score by 10–20 points.
  At L3 FastV-style is ahead on cognition (426.8 vs 397.1).
- The score head is trained on GQA only and still leads on MMBench and MME.

### 5.2 Development results (tune2k, n = 2000, full = 68.90)

Budget curve (difference to full in points; one-sided lower bound):

| K/N | 3d_attn50k | PP3-fc (ρ = 0.25) |
|---|---|---|
| 0.11 (L2) | −1.20 (−2.30) | −2.15 (−3.25) |
| 0.22 (L3) | −0.20 (−1.15) | −0.80 (−1.71) |
| 0.33 (L4) | −0.20 (−0.99) | −0.35 (−1.14) |
| 0.50 (L5) | −0.05 (−0.68) | −0.05 (−0.71) |

All methods, difference to full in points (tune2k):

| K/N | Random | DivPrune | FastV-style | PP3 (3d_attn50k) |
|---|---|---|---|---|
| 0.057 (L1) | −20.10 | −16.55 | −30.40 | **−4.35** |
| 0.11 (L2) | −13.05 | −7.90 | −17.85 | **−1.20** |
| 0.22 (L3) | −6.95 | −3.35 | −7.70 | **−0.20** |
| 0.33 (L4) | −4.00 | −1.45 | −4.35 | **−0.20** |

Head-to-head (McNemar, same questions): PP3 vs DivPrune +6.70 [+4.92, +8.48]
at L2 and +3.15 [+1.81, +4.49] at L3; vs FastV-style +16.65 at L2 and +7.50
at L3 (all p < 0.001). The FastV-style selection is below random at every
budget; its layer-2 attention concentrates on sink tokens (`TODO: verify
with a FastV implementation that prunes inside the LLM`).

### 5.2b Confirmation split (gqa_val_confirm, n = 4890, disjoint from tune2k, full = 69.55)

The configuration was fixed before this split was scored.

| K/N | Random | DivPrune | FastV-style | PP3 | PP3 one-sided LB |
|---|---|---|---|---|---|
| 0.056 (L1) | −18.69 | −14.46 | −29.08 | **−4.62** | −5.45 |
| 0.11 (L2) | | | | **−1.62** | −2.30 |
| 0.22 (L3) | −5.81 | −3.11 | −7.03 | **−0.18** | **−0.73** |
| 0.33 (L4) | | | | **+0.04** | **−0.43** |

PP3 vs DivPrune: +9.84 [+8.55, +11.12] at L1, +2.92 [+2.09, +3.76] at L3; vs
FastV-style +6.85 [+5.76, +7.94] at L3 (all p < 0.001). At L3 and L4 the
one-sided lower bound clears the 1-point non-inferiority margin.

### 5.3 Ablation (tune2k)

| Step | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Random (fold) | −20.10 | −13.05 | −6.95 | −4.00 |
| DSFold-D, hand-crafted quality | −11.05 | −6.05 | −3.25 | −0.95 |
| learned head 3a, gradient labels (50k) | −3.15 | −1.60 | −0.50 | −0.40 |
| cross-attention head 3c, gradient labels | | | −1.00 | −0.70 |
| 3d, attention teacher (10k) | | | −0.25 | −0.20 |
| 3d, attention teacher (50k) | −4.35 | −1.20 | −0.20 | −0.20 |
| 3d (50k), no fold (γ = 0) | | −1.25 | −0.25 | +0.15 |

- A learned quality is the largest step: DSFold-D → 3a gains +7.90 (L1),
  +4.45 (L2), +2.75 (L3), +0.55 (L4) points.
- Even the training-free DSFold-D beats DivPrune at every budget
  (L1 −11.05 vs −16.55; L3 −3.25 vs −3.35; L4 −0.95 vs −1.45).
- Label source: with 10k labels the attention teacher beats gradient labels
  by +1.70 at L3 (p = 0.003). With 50k labels both are equivalent: 3d
  (attention) vs 3a (gradient) −1.20 at L1 (p = 0.15), +0.40 at L2
  (p = 0.60), +0.30 at L3, +0.20 at L4. The teacher is more label-efficient
  and needs no backward pass, but it is not more accurate at 50k.
- The DeepStack fold makes no measurable difference on GQA.

### 5.4 Region analysis (tune2k)

| Run | groups / 10 kept | question overlap |
|---|---|---|
| random L3 | 3.76 | 0.13 |
| 3d_attn50k L3 | ~3.8 | 0.65 |
| PP3-fc focus part, L4 (ρ = 0.25) | 1.76 | 0.15 |

Trade-off at L4: focus share 0 / 25 / 40 / 55% → −0.20 / −0.35 / −0.60 /
−1.05 points. At L5 the focus costs nothing (−0.05 for both).

### 5.5 Yes/no questions

Yes/no questions (566 of 2000, detected from the question text) stay 0.7–1.8
points below full at every budget up to 50%, with or without the fold; the losses are almost all
questions whose answer is "no" and which the pruned model answers "yes"
(e.g. L4: 14 lost, 1 gained on "no"; 3 lost, 6 gained on "yes"), the
opposite of random selection, which loses "yes" answers (24 lost). A
per-question budget by question form gives no gain beyond noise.

### 5.6 Efficiency

Dedicated timing run, one process alone on the GPU, 300 questions each:
GQA val (tune2k, short visual sequences) and the first 300 MME test
questions (code reasoning and artwork, ~900 input tokens). Mean ms per
sample, ViT excluded (identical for all methods); "total" = selector +
prefill + decode. Accuracy in this run is not used (n = 300).
`outputs/{none,pp3_scorer}/runs/timing_*`.

| Run | GQA tokens | selector | prefill | decode | total | MME tokens | selector | prefill | decode | total |
|---|---|---|---|---|---|---|---|---|---|---|
| Full | 249 | 0.1 | 51.0 | 51.5 | 102.6 | 912 | 0.1 | 88.0 | 31.3 | 119.4 |
| Random L3 | 75 | 2.5 | 27.5 | 30.0 | 60.0 | 230 | 2.7 | 31.6 | 25.4 | 59.7 |
| DivPrune L1 | 38 | 3.4 | 27.7 | 31.9 | 63.0 | 84 | 8.1 | 28.1 | 25.2 | 61.4 |
| DivPrune L3 | 75 | 7.6 | 27.6 | 32.1 | 67.3 | 230 | 25.2 | 31.5 | 25.2 | 81.9 |
| FastV-style L1* | 38 | 1.6 | 27.6 | 29.7 | 58.9 | 84 | 1.7 | 28.1 | 25.1 | 54.9 |
| FastV-style L3* | 75 | 1.7 | 27.5 | 30.7 | 59.9 | 230 | 1.8 | 31.4 | 25.0 | 58.2 |
| PP3 L1 | 38 | 9.9 | 27.8 | 31.8 | 69.5 | 84 | 21.0 | 28.2 | 25.2 | 74.4 |
| PP3 L3 | 75 | 29.4 | 40.6 | 34.6 | 104.6 | 230 | 65.0 | 33.9 | 26.8 | 125.7 |
| PP3 L4 | 100 | 26.9 | 28.0 | 30.4 | 85.3 | 327 | 90.3 | 38.6 | 26.1 | 155.0 |

*The FastV-style selector time excludes its extra full-token pass through
the first two decoder layers (`prepare_sample`), so its cost is understated.

- **Pruning itself pays.** With any selector the language model's prefill
  drops from 51 to ~28 ms on GQA and from 88 to 28–39 ms on MME, and decode
  drops from 51.5 to ~30 ms on GQA (shorter KV cache). Peak VRAM changes by
  < 0.3 GB because the 2B model's weights dominate.
- **The PP3 selector costs 10–90 ms**, growing with `K` and `N` (score head
  plus greedy DPP). End to end, PP3 is faster than full at L1 (−32% on GQA,
  −38% on MME), about equal at L3/L4 on GQA (85–105 vs 103 ms), and slower at
  L3/L4 on MME (126–155 vs 119 ms).
- On the full MME test (mean 1790 input tokens, shared GPU) prefill drops
  from 396 to 97 ms at L3, while the selector grows to ~300 ms; the selector,
  not the language model, is the bottleneck at high resolution.
- PP3 is therefore presented as a quality-at-budget method; its token
  reduction (71–76% fewer LLM input tokens at L3) is real, but turning it into
  wall-clock gain needs a faster selector (future work).

## 6 Limitations

- One model size (2B), one seed per configuration; 4B and other families in
  future work.
- No wall-clock gain: the selector is not optimised and its greedy DPP grows
  quadratically with `N` (5.6).
- The teacher imitates the full-token model, so parity with full is the
  expected ceiling; PP3 does not exceed the full model on GQA or MMB.
- Text-heavy inputs (MME cognition, OCR) lose 80–140 MME points for every
  selector; the head is trained on GQA only.
- MMBench at 33% of the tokens is 0.97 points below full and misses the
  non-inferiority margin.
- Teacher attention inherits the language model's biases (sinks).
- Yes/no questions with "no" answers remain below full tokens.
- The FastV-style baseline prunes at the LLM input with layer-2 attention; it
  is not the original in-LLM FastV.

## 7 Conclusion

A 2M-parameter question-aware score head, distilled from the language
model's own text-to-image attention and combined with DPP selection, is a
strong visual-token selector for the DeepStack model Qwen3-VL-2B. At equal
budgets it beats random, DivPrune and FastV-style selection on GQA, MMBench
and MME perception, by the widest margin at the smallest budgets, and with
22–33% of the tokens it is non-inferior to the full model on GQA and matches
its MME perception score. Open problems are text-heavy images, a selector
whose cost does not grow with the square of the token count, and training
targets that can exceed rather than imitate the full model.
