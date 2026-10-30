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
    from token_reduction_vl.pruning.pp3_scorer.saliency import gradient_x_input_saliency

    image, prompt = sample
    text = question_text_for_scoring(prompt)
    pruned = tiny_runner.predict(image, prompt, text, ScoredDSFoldPruner(build_score_head("3c", (64, 64, 64))), 2 / 9)
    assert pruned.num_kept == 16 and pruned.num_visual == 70
    assert pruned.input_tokens == 92 - 70 + 16

    encoded = tiny_runner.encode(image, prompt, text)
    saliency = gradient_x_input_saliency(tiny_runner, encoded, torch.tensor([100, 200]))
    assert saliency.shape == (70,)
    assert bool(torch.isfinite(saliency).all()) and bool((saliency > 0).any())
