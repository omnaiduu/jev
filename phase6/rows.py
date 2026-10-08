"""Judge rows for Phase 6.

Phase 1 rejects a source that contains typed-decisions. This module does not
call that guard. typed-decisions is a legal source here, and only here.
"""

from __future__ import annotations

import random

ALLOWED_SOURCES = ("boolq", "multinli", "banking77", "typed-decisions", "snli")


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
    option_ids: list[str],
    target: list[float],
    label_id: str,
    target_kind: str,
    group_id: str,
    case_id: str = "",
    question_name: str = "",
    workflow: str = "",
) -> dict:
    if label_id not in option_ids:
        raise ValueError(f"{row_id} label {label_id!r} is not an option id")
    row = {
        "id": row_id,
        "source": source,
        "type": question_type,
        "state": state.strip(),
        "question": question.strip(),
        "options": list(options),
        "option_ids": list(option_ids),
        "target": [float(value) for value in target],
        "target_kind": target_kind,
        "label_id": label_id,
        "label_index": option_ids.index(label_id),
        "group_id": group_id,
        "case_id": case_id,
        "question_name": question_name,
        "workflow": workflow,
        "shuffled": False,
    }
    validate_row(row)
    return row


def validate_row(row: dict) -> None:
    if row["source"] not in ALLOWED_SOURCES:
        raise ValueError(f"unexpected source {row['source']}")
    if not row["state"] or not row["question"]:
        raise ValueError(f"empty text in {row['id']}")
    if len(row["options"]) < 2:
        raise ValueError(f"{row['id']} needs at least two options")
    width = len(row["options"])
    if len(row["option_ids"]) != width or len(row["target"]) != width:
        raise ValueError(f"{row['id']} options, option ids, and target differ in length")
    if len(set(row["options"])) != width:
        raise ValueError(f"{row['id']} has duplicate option text")
    if len(set(row["option_ids"])) != width:
        raise ValueError(f"{row['id']} has duplicate option ids")
    if abs(sum(row["target"]) - 1.0) > 1e-6:
        raise ValueError(f"{row['id']} target does not sum to 1")
    if any(value < 0 for value in row["target"]):
        raise ValueError(f"{row['id']} target has a negative weight")
    if row["target_kind"] == "one_hot":
        if sum(1 for value in row["target"] if value == 1.0) != 1:
            raise ValueError(f"{row['id']} target is not one-hot")
    elif row["target_kind"] != "soft":
        raise ValueError(f"{row['id']} target_kind {row['target_kind']}")
    if row["option_ids"][row["label_index"]] != row["label_id"]:
        raise ValueError(f"{row['id']} label index does not match the label id")


def _permute(row: dict, order: list[int], shuffled: bool) -> dict:
    updated = dict(row)
    for field in ("options", "option_ids", "target"):
        updated[field] = [row[field][index] for index in order]
    updated["label_index"] = updated["option_ids"].index(row["label_id"])
    updated["shuffled"] = shuffled
    validate_row(updated)
    return updated


def shuffle_options(row: dict, rng: random.Random) -> dict:
    order = list(range(len(row["options"])))
    while True:
        rng.shuffle(order)
        if order != list(range(len(order))):
            break
    return _permute(row, order, True)


def reverse_options(row: dict) -> dict:
    """Reverse the option list and rebind A, B, C in that new order.

    The label id stays. The letter index moves. This is the position-bias
    copy, not a training row.
    """
    order = list(range(len(row["options"]) - 1, -1, -1))
    flipped = _permute(row, order, False)
    if flipped["label_id"] != row["label_id"]:
        raise RuntimeError("reversed options lost the label id")
    return flipped


def example_key(row: dict) -> str:
    """Identity of the underlying example, ignoring option order and the letter."""
    return "\n".join(
        [
            row["source"],
            row.get("case_id") or "",
            row.get("question_name") or "",
            row["state"],
            row["question"],
        ]
    )
