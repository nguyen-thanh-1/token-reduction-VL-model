"""Question form from the text alone, for a per-question budget rule.

Uses only what is available at inference (no dataset annotations):

- ``yesno``: starts with an auxiliary verb (Is/Are/Does/Can/...) and offers no
  alternatives, so the answer is yes or no;
- ``choice``: names alternatives (`` or ``) or comes with answer options;
- ``open``: everything else (What/Which/Who/Where/How ... questions).
"""

from __future__ import annotations

import re

FORMS = ("yesno", "choice", "open")
AUXILIARIES = frozenset(
    "is are am was were do does did can could will would should shall may might must has have had".split()
)


def question_form(question: str, choices: list | tuple | None = None) -> str:
    """``yesno``, ``choice`` or ``open`` for the question (prompt suffix lines are ignored)."""

    first_line = question.strip().split("\n", 1)[0].strip().lower()
    if choices:
        return "choice"
    if re.search(r"\bor\b", first_line):
        return "choice"
    words = re.findall(r"[a-z']+", first_line)
    return "yesno" if words and words[0] in AUXILIARIES else "open"
