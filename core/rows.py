"""Shared row helpers for the Phase 1 judge files."""

from __future__ import annotations

import random
import re

TEAM_OPTIONS = ["billing", "tech", "sales"]
YES_NO = ["no", "yes"]
NLI_OPTIONS = ["entailment", "neutral", "contradiction"]
FORBIDDEN_SOURCES = ("typed-decisions", "jev-distill")


def one_hot(index: int, size: int) -> list[float]:
    if index < 0 or index >= size:
        raise ValueError(f"index {index} outside 0..{size - 1}")
    return [1.0 if i == index else 0.0 for i in range(size)]


def make_row(
    row_id: str,
    source: str,
    question_type: str,
    state: str,
    question: str,
    options: list[str],
    target: list[float],
) -> dict:
    row = {
        "id": row_id,
        "source": source,
        "type": question_type,
        "state": state.strip(),
        "question": question.strip(),
        "options": list(options),
        "target": list(target),
        "shuffled": False,
    }
    validate_row(row)
    return row


def validate_row(row: dict) -> None:
    if any(token in row["source"] for token in FORBIDDEN_SOURCES):
        raise ValueError(f"forbidden source {row['source']}")
    if not row["state"] or not row["question"]:
        raise ValueError(f"empty text in {row['id']}")
    if len(row["options"]) < 2:
        raise ValueError(f"{row['id']} needs at least two options")
    if len(row["options"]) != len(row["target"]):
        raise ValueError(f"{row['id']} options and target differ in length")
    if len(set(row["options"])) != len(row["options"]):
        raise ValueError(f"{row['id']} has duplicate options")
    if abs(sum(row["target"]) - 1.0) > 1e-6:
        raise ValueError(f"{row['id']} target does not sum to 1")
    if sum(1 for value in row["target"] if value == 1.0) != 1:
        raise ValueError(f"{row['id']} target is not one-hot")


def shuffle_options(row: dict, rng: random.Random) -> dict:
    order = list(range(len(row["options"])))
    while True:
        rng.shuffle(order)
        if order != list(range(len(order))):
            break
    updated = dict(row)
    updated["options"] = [row["options"][index] for index in order]
    updated["target"] = [row["target"][index] for index in order]
    updated["shuffled"] = True
    validate_row(updated)
    return updated


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None
