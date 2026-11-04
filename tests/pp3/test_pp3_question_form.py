import pytest

from token_reduction_vl.pruning.pp3_scorer.question_form import question_form


@pytest.mark.parametrize(
    ("question", "form"),
    [
        ("Does the belt look tan?\n\nAnswer with a short phrase.", "yesno"),
        ("Are there both cars and trucks in the picture?", "yesno"),
        ("Is the shirt red or blue?", "choice"),
        ("Which side is the cup on, left or right?", "choice"),
        ("What device is not on?\n\nAnswer with a short phrase.", "open"),
        ("Who is wearing the hat?", "open"),
        ("Island or beach scene?", "choice"),
        ("Isolated tree?", "open"),
    ],
)
def test_question_form_from_text(question: str, form: str) -> None:
    assert question_form(question) == form


def test_options_make_a_choice_question() -> None:
    assert question_form("Is it a cat?", choices=[{"label": "A", "text": "cat"}]) == "choice"
