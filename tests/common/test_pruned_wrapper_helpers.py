import torch

from token_reduction_vl.data.messages import build_question_prompt
from token_reduction_vl.data.schema import CanonicalSample, Choice
from token_reduction_vl.models.qwen3_vl_pruned import (
    build_position_ids,
    decode_position_ids,
    question_text_for_scoring,
    sequence_keep_indices,
)


def test_question_text_drops_hint_options_header_and_instruction() -> None:
    sample = CanonicalSample(
        sample_id="mmb:1", dataset="mmbench", split="validation", task_type="multiple_choice",
        image="x.jpg", image_id="1", question="Which animal is shown?", hint="A farm photo.",
        choices=(Choice("A", "horse"), Choice("B", "cow")), answer_label="A", has_label=True,
    )
    prompt = build_question_prompt(sample)
    text = question_text_for_scoring(prompt, [c.to_dict() for c in sample.choices])
    assert text == "Which animal is shown? horse cow"


def test_question_text_for_open_and_binary_prompts() -> None:
    assert question_text_for_scoring("Is it overcast?\n\nAnswer with a short phrase.") == "Is it overcast?"
    assert question_text_for_scoring("Is there a dog?\n\nAnswer yes or no only.") == "Is there a dog?"


def test_sequence_keep_indices_keeps_text_and_selected_image_tokens() -> None:
    # 2 text tokens, 4 image tokens at positions 2..5, 2 trailing text tokens.
    image_positions = torch.tensor([2, 3, 4, 5])
    keep = sequence_keep_indices(8, image_positions, torch.tensor([1, 3]))
    assert keep.tolist() == [0, 1, 3, 5, 6, 7]


def test_position_ids_keep_original_grid_coordinates() -> None:
    # Text at 0..1; a 2x2 image starting at s = 2 → (t, h, w) = (2, 2+row, 2+col); text resumes at 2 + 2 = 4.
    mrope = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5],
            [0, 1, 2, 2, 3, 3, 4, 5],
            [0, 1, 2, 3, 2, 3, 4, 5],
        ]
    ).view(3, 1, 8)
    keep_seq = sequence_keep_indices(8, torch.tensor([2, 3, 4, 5]), torch.tensor([0, 3]))
    position_ids, max_position = build_position_ids(mrope, keep_seq)
    assert position_ids.shape == (4, 1, 6)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3, 4, 5]  # sequence index row
    assert position_ids[1:, 0, 2].tolist() == [2, 2, 2]  # token (row 0, col 0)
    assert position_ids[1:, 0, 3].tolist() == [2, 3, 3]  # token (row 1, col 1) keeps its coordinates
    assert max_position == 5


def test_decode_positions_continue_after_the_last_text_token() -> None:
    position = decode_position_ids(prefill_len=6, max_position=5, step=0, device=torch.device("cpu"))
    assert position.view(-1).tolist() == [6, 6, 6, 6]
    position = decode_position_ids(prefill_len=6, max_position=5, step=3, device=torch.device("cpu"))
    assert position.view(-1).tolist() == [9, 9, 9, 9]
