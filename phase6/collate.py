"""Right-padded batches for the letter readout.

The mask is built here. It is not the tokenizer's padding, because Gemma
left-pads by default. Every row is checked before it can be scored or trained.
"""

from __future__ import annotations

from phase2.prompts import LETTERS, render_prompt
from phase6.padding import assert_right_pad


def collate(tokenizer, rows: list[dict], letter_ids: dict[str, int], device, torch, max_length: int):
    encoded_rows = []
    truncated = 0
    for row in rows:
        ids = tokenizer(render_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            ids = ids[-max_length:]
            truncated += 1
        if not ids:
            raise RuntimeError(f"{row['id']} encoded to an empty prompt")
        encoded_rows.append(ids)
    width = max(len(ids) for ids in encoded_rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad token")
    input_ids = []
    masks = []
    indexes = []
    for ids in encoded_rows:
        pad = width - len(ids)
        mask = [1] * len(ids) + [0] * pad
        input_ids.append(ids + [pad_id] * pad)
        masks.append(mask)
        indexes.append(assert_right_pad(mask))
    selected = []
    targets = []
    for row in rows:
        selected.append(
            torch.tensor(
                [letter_ids[LETTERS[index]] for index in range(len(row["options"]))],
                dtype=torch.long,
                device=device,
            )
        )
        targets.append(torch.tensor(row["target"], dtype=torch.float32, device=device))
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(masks, device=device),
        "selected": selected,
        "targets": targets,
        "indexes": torch.tensor(indexes, dtype=torch.long, device=device),
        "truncated": truncated,
    }
