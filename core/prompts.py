"""Judge prompt for a Phase 1 row. The readout scores the letter, not the option word."""

from __future__ import annotations

LETTERS = [chr(ord("A") + i) for i in range(26)]

SYSTEM_PROMPT = (
    "You are a judge. The next token must be one of the option letters. Do not explain."
)


def render_user_prompt(row: dict) -> str:
    if len(row["options"]) > len(LETTERS):
        raise ValueError(f"{row['id']} has too many options")
    lines = [
        "State:",
        row["state"],
        "",
        f"Question: {row['question']}",
        "Options:",
    ]
    for index, option in enumerate(row["options"]):
        lines.append(f"{LETTERS[index]}. {option}")
    lines.append("")
    lines.append("Reply with the single letter of the best option.")
    return "\n".join(lines)


def render_prompt(tokenizer, row: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(row)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
