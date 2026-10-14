"""Turn one typed-decisions row into five judge questions.

Option keys such as "human_review" are more than one Gemma token, so a single
softmax has no slot for them. Each option is bound to one letter, A, B, C, ...,
and the letter is what the readout scores. The option id and its description
stay in the prompt.
"""

from __future__ import annotations

import json
from typing import Any


LETTERS = [chr(ord("A") + i) for i in range(26)]


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def option_pairs(question: dict) -> list[tuple[str, str]]:
    criteria = question.get("criteria")
    if criteria is None:
        # Some noul items only state a claim. The gold label is still false or true.
        if question.get("type") != "noul":
            raise ValueError(f"question has no criteria: {question!r}")
        criteria = {
            "false": "The statement is not true.",
            "true": "The statement is true.",
        }
    if isinstance(criteria, list):
        return [(str(index), text) for index, text in enumerate(criteria)]
    return [(str(key), "" if text is None else str(text)) for key, text in criteria.items()]


def iter_questions(row: dict) -> list[dict]:
    questions = _parse(row["questions"])
    gold = _parse(row["gold"])
    state = row["state"]
    items = []
    for name, question in questions.items():
        pairs = option_pairs(question)
        if len(pairs) > len(LETTERS):
            raise ValueError(f"{name} has {len(pairs)} options, more than {len(LETTERS)} letters")
        label = str(gold[name]["label"])
        ids = [option_id for option_id, _ in pairs]
        if label not in ids:
            raise ValueError(f"{name} gold label {label!r} is not in {ids}")
        items.append(
            {
                "case_id": row["id"],
                "workflow": row["workflow"],
                "name": name,
                "type": question["type"],
                "state": state,
                "instructions": question["instructions"],
                "options": [
                    {"letter": LETTERS[index], "id": option_id, "text": text}
                    for index, (option_id, text) in enumerate(pairs)
                ],
                "label": label,
                "label_index": ids.index(label),
            }
        )
    return items


def render_user_prompt(item: dict) -> str:
    lines = [
        "State:",
        item["state"],
        "",
        f"Question: {item['instructions']}",
        "Options:",
    ]
    for option in item["options"]:
        lines.append(f"{option['letter']}. {option['id']}: {option['text']}")
    lines.append("")
    lines.append("Reply with the single letter of the best option.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a judge. The next token must be one of the option letters. Do not explain."
)
