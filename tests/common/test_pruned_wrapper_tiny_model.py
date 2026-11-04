"""The pruning wrapper reproduces ``model.generate()`` on a tiny random Qwen3-VL (CPU).

Only the processor/tokenizer files of the cached checkpoint are read; no
weights are loaded and nothing is downloaded. Skipped when the cache is absent.
"""

from pathlib import Path

import pytest
import torch

SNAPSHOTS = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
SNAPSHOT = next(iter(sorted(SNAPSHOTS.glob("*"))), None) if SNAPSHOTS.exists() else None

pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="Qwen3-VL processor files are not cached locally")


@pytest.fixture(scope="module")
def tiny_runner():
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=151936, rope_theta=5_000_000,
            rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16,
            spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0, 1, 2],
            num_position_embeddings=2304,
        ),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = [151645, 151643]
    runner = PrunedQwen3VL.__new__(PrunedQwen3VL)
    runner.model = model.requires_grad_(False)
    runner.processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    runner.max_new_tokens = 8
    runner.image_token_id = 151655
    runner.eos_token_ids = {151645, 151643}
    return runner


@pytest.fixture(scope="module")
def sample():
    from PIL import Image

    image = Image.new("RGB", (320, 224), color=(120, 30, 200))
    return image, "What colour is the box?\n\nAnswer with a short phrase."


def test_noop_wrapper_matches_generate(tiny_runner, sample) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import question_text_for_scoring
    from token_reduction_vl.pruning import NoOpPruner

    image, prompt = sample
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = tiny_runner.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    with torch.no_grad():
        reference = tiny_runner.model.generate(**inputs, max_new_tokens=8, do_sample=False)
    expected = reference[0, inputs["input_ids"].shape[1] :].tolist()
    result = tiny_runner.predict(image, prompt, question_text_for_scoring(prompt), NoOpPruner(), 1.0)
    assert result.output_ids == expected
    assert result.input_tokens == inputs["input_ids"].shape[1]


def test_pp3_pruning_and_saliency_run_end_to_end(tiny_runner, sample) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import question_text_for_scoring
    from token_reduction_vl.pruning.pp3_scorer import ScoredDSFoldPruner, build_score_head
    from token_reduction_vl.pruning.pp3_scorer.saliency import gradient_x_input_saliency, signed_gradient_x_input

    image, prompt = sample
    text = question_text_for_scoring(prompt)
    pruned = tiny_runner.predict(image, prompt, text, ScoredDSFoldPruner(build_score_head("3c", (64, 64, 64))), 2 / 9)
    assert pruned.num_kept == 16 and pruned.num_visual == 70
    assert pruned.input_tokens == 92 - 70 + 16

    encoded = tiny_runner.encode(image, prompt, text)
    saliency = gradient_x_input_saliency(tiny_runner, encoded, torch.tensor([100, 200]))
    assert saliency.shape == (70,)
    assert bool(torch.isfinite(saliency).all()) and bool((saliency > 0).any())

    signed = signed_gradient_x_input(tiny_runner, encoded, torch.tensor([100, 200]))
    assert signed.shape == (70,)
    assert bool((signed.abs() <= saliency + 1e-5).all())


def test_attention_teacher_reads_visual_attention(tiny_runner, sample) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import question_text_for_scoring
    from token_reduction_vl.pruning.pp3_scorer.attention_teacher import attention_teacher, question_positions

    image, prompt = sample
    encoded = tiny_runner.encode(image, prompt, question_text_for_scoring(prompt))
    rows = question_positions(encoded.input_ids, encoded.image_positions, encoded.question_ids)
    assert rows.numel() > 0
    assert encoded.input_ids[0, rows].tolist() == encoded.question_ids.reshape(-1)[: rows.numel()].tolist()

    previous = tiny_runner.language_model.config._attn_implementation
    teacher = attention_teacher(tiny_runner, encoded, torch.tensor([100, 200]), layers=(1, 2))
    assert tiny_runner.language_model.config._attn_implementation == previous
    for target in (teacher.answer, teacher.question):
        assert target.shape == (70,)
        assert float(target.sum()) == pytest.approx(1.0, abs=1e-4)
        assert bool((target >= 0).all())
    assert teacher.question_rows == rows.numel()


def test_fastv_pruner_ranks_by_layer_attention(tiny_runner, sample) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import question_text_for_scoring
    from token_reduction_vl.pruning.pp3_scorer.baselines import FastVPruner

    image, prompt = sample
    scoring_text = question_text_for_scoring(prompt)
    encoded = tiny_runner.encode(image, prompt, scoring_text)
    previous = tiny_runner.language_model.config._attn_implementation
    pruner = FastVPruner()
    pruner.prepare_sample(tiny_runner, encoded)
    assert tiny_runner.language_model.config._attn_implementation == previous
    assert pruner.attention.shape == (70,) and bool((pruner.attention >= 0).all())
    result = tiny_runner.predict(image, prompt, scoring_text, pruner, 1 / 3, encoded=encoded)
    expected = torch.sort(torch.topk(pruner.attention, result.num_kept).indices).values.tolist()
    assert result.keep_indices == expected


def test_received_attention_chunks_match_eager_weights(tiny_runner, sample) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import question_text_for_scoring
    from token_reduction_vl.pruning.base import PrunerOutput
    from token_reduction_vl.pruning.pp3_scorer.baselines import FASTV_LAYER, received_attention

    image, prompt = sample
    encoded = tiny_runner.encode(image, prompt, question_text_for_scoring(prompt))
    n = encoded.num_visual
    prepared = tiny_runner.prepare(
        encoded,
        PrunerOutput(torch.arange(n), encoded.v, list(encoded.deepstack), {"K": n, "N": n}),
    )
    cols = encoded.image_positions
    start = int(cols.min())
    captured = []
    lm = tiny_runner.language_model
    previous = lm.config._attn_implementation
    handle = lm.layers[FASTV_LAYER].self_attn.register_forward_hook(
        lambda _m, _a, out: captured.append(out[1][0, :, start:, :].float().mean(0))
    )
    try:
        lm.config._attn_implementation = "eager"
        with torch.no_grad():
            lm(
                inputs_embeds=prepared.inputs_embeds,
                position_ids=prepared.position_ids,
                attention_mask=torch.ones(prepared.inputs_embeds.shape[:2], dtype=torch.long),
                visual_pos_masks=prepared.visual_pos_masks,
                deepstack_visual_embeds=prepared.deepstack,
                use_cache=False,
            )
    finally:
        lm.config._attn_implementation = previous
        handle.remove()
    rows = captured[0]
    later = (torch.arange(start, start + rows.shape[0])[:, None] >= cols[None, :]).sum(0)
    reference = rows[:, cols].sum(0) / later.clamp_min(1)
    for chunk in (7, 512):
        assert torch.allclose(received_attention(tiny_runner, encoded, chunk=chunk), reference, atol=1e-5)
