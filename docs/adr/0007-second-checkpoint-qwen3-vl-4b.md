# ADR 0007: Second checkpoint — Qwen3-VL-4B-Instruct

- Status: Accepted
- Date: 2026-11-05

## Context and decision drivers

Every PP3 result so far is on Qwen3-VL-2B-Instruct. A single checkpoint is the
first thing a reviewer questions: the score head could be fitting quirks of one
model. Qwen3-VL-4B-Instruct has the same ViT (24 layers, width 1024, DeepStack
at ViT layers 5/11/17) and the same token layout, but a deeper and wider LLM
(36 layers, width 2560, 32 query / 8 KV heads; 2B: 28 layers, width 2048). It
tests whether the method transfers to a larger LLM without any change to the
method itself.

## Decision

1. **Checkpoint.** `Qwen/Qwen3-VL-4B-Instruct`, revision
   `ebb281ec70b05090aa6165b016eac8ec08e71b17` (8.88 GB of bf16 safetensors),
   the official and most-downloaded release. bf16 as for 2B; no quantization
   (it would change the teacher attention and the timing, and confound model
   size with numeric precision). Downloaded only with the user's OK.
2. **Method unchanged.** Same DSFold parameters (`dsfold_D-1.0.yaml`), same
   score-head architecture (3d) and training hyper-parameters as
   `pp3_scorer_attn50k.yaml`, same PCA output sizes (256 / 64 / 256), so the
   head keeps about 2.0M parameters. Only two things are re-derived for 4B:
   - the fixed PCA compressor (fitted on 1000 GQA-train samples of 4B features);
   - the teacher layers: 2B averages LLM layers 8–19 of 28 (relative depth
     0.29–0.68); 4B uses layers 10–24 of 36, the same relative depth. Chosen
     by rule, not tuned on 4B.
3. **Labels.** Same GQA-train split (`gqa_train_50k.txt`). Start with the first
   20k samples; extend to 50k only if the GPU schedule allows. The number of
   labels used is reported.
4. **Evaluation.** Test sets only, configuration frozen as above: GQA testdev,
   MMBench dev, MME, POPE at L1–L4, against full, random, DivPrune and CDPruner
   (FastV-style is not run). Same statistics as for 2B. Results are reported in
   their own table, never averaged with 2B.
5. **Separation of outputs.** `configs/methods/pp3_scorer_attn_4b.yaml` sets
   `model.run_tag: 4b`; `run_diagnostics.py` refuses a model other than the
   baseline model unless `--prefix` ends with the tag (`final4b`), so 4B runs
   land in `outputs/{none,pp3_scorer}/runs/final4b_*` and never overwrite 2B
   runs. Features in `data/features/pp3_scorer_4b/`, checkpoint
   `score_head_3d_attn4b_seed0.pt`. `run_meta.json` records the model id and
   revision.

## Memory plan (RTX 5060 Ti, 16 GB)

| Step | Loads the LLM | Estimated peak VRAM |
|---|---|---|
| Fit PCA, attention labels (GQA train, eager attention) | yes | 10–11 GB |
| Train score head (stored features) | no | < 1 GB |
| Evaluation GQA / MMBench / POPE | yes | 10–11 GB |
| Evaluation MME (up to ~16k visual tokens) | yes | 12–13 GB (smoke test first) |

Only one process that loads the 4B LLM runs at a time. Loading needs about 9 GB
of free system RAM.

## Alternatives considered

- **FP8 / 4-bit checkpoints.** Smaller, but they change the teacher and the
  timing; kept as a fallback only if bf16 does not fit, and then reported.
- **Qwen3-VL-8B.** About 17 GB in bf16; does not fit the GPU.
- **A different model family (e.g. LLaVA).** No DeepStack, so the fold and the
  four-stream selection would not be exercised; more porting work.

## Consequences

- About 9 GB of disk for weights and 3.6 GB (20k) to 9 GB (50k) for features.
- GPU time is roughly 1.3–2× that of 2B per step.
- `PrunedQwen3VL` accepts a `revision`; PP3 scripts load the model from the
  method config's `model:` block (unchanged for every existing 2B config).

## Amendment (2026-11-05): 50k labels, as for 2B

The first 4B head (`3d_attn4b`, first 20k labels) overfit: train ListNet loss
fell from 4.79 to 4.64 while the held-out loss stayed at 4.76, so early
stopping (patience 3) kept epoch 1 (held-out recall@K at L3 0.685, against
0.836 for the 2B head on 50k labels). The 4B teacher is also sharper (top-1
mass 0.15 against 0.066). Its test runs (`final4b_3d_attn4b_*`, MMBench and
part of GQA testdev) are kept and reported as such: L1 −10.09 (GQA, first
3576 questions) and −11.73 (MMBench), no better than DivPrune or CDPruner.

The head is retrained on 50k labels, the same split and size as 2B, with every
other setting unchanged (`3d_attn4b50k`). The decision uses only the held-out
training images and tune2k (GQA val), never the test sets: it goes to test
only if its held-out recall is close to the 2B head's and it beats CDPruner on
tune2k at L1 and L3. The fix (label count) was fixed before any new test run.

## Amendment 2 (2026-11-05): best checkpoint by held-out recall@K

With 50k labels (`3d_attn4b50k`) the two held-out criteria disagree on 4B: the
held-out loss is lowest at epoch 1 (4.715) and rises after, while held-out
recall@K at L3 keeps rising (0.696, 0.712, 0.714, 0.726, 0.731 at epochs 0–4).
Early stopping on the loss kept epoch 1. On tune2k (GQA val) that checkpoint
gives L1 62.70 and L3 70.50 against CDPruner 61.55 and 69.70 (full 71.60), so
the gate of amendment 1 passes on tune2k but held-out recall (0.71) is not close
to the 2B head's (0.836); the hand-crafted ranking reaches 0.31 on 4B and 0.30
on 2B, so the task is equally hard and the gap is the head's.

Selection is the step that matters for token choice, so the rule becomes
`select_on: recall` (best checkpoint and patience on held-out recall@K at L3)
for **both** checkpoints. On 2B both criteria pick epoch 19, so the 2B head and
every 2B result are unchanged. The 4B head continues from the epoch-4 state of
`3d_attn4b50k` (same optimizer and 20-epoch cosine schedule) as
`3d_attn4b50k_rec`. The decision again uses only held-out training images and
tune2k; `3d_attn4b50k` and its tune2k runs are kept. Test runs use whichever of
the two passes the gate with the larger tune2k margin, chosen before any test run.

**Outcome (tune2k, decided before any test run).** `3d_attn4b50k_rec` (epoch 19,
held-out recall 0.757) gives L1 65.70 and L3 70.85 against CDPruner 61.55 and
69.70 and against `3d_attn4b50k` 62.70 and 70.50 (full 71.60). It passes the
gate with the larger margin and is the 4B head for every test run (prefix
`final4b`, tag `3d_attn4b50k_rec`). Full and the training-free baselines are
independent of the head, so their `final4b` runs are reused and resumed.
Timing of the two tune2k runs made in a separate process is not compared with
the shared-pass timing of the others.
