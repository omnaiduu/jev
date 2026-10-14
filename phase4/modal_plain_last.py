"""Score plain E4B at the last content token.

Phase 0 read attention_mask.sum() - 1. With left padding that is the first
content token on a short row. This run keeps that index and also reads the
last content token. No LoRA. No training. The container is single-use.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase0.items import SYSTEM_PROMPT, iter_questions, render_user_prompt
from phase4.score import last_content_index, phase0_index, readout, summarize_records

APP_NAME = "phase4-plain-last-token"
MODEL_ID = "google/gemma-4-E4B-it"
EXAM = "LocalLLaMA/typed-decisions"
EXAM_CONFIG = "all"
EXAM_SPLIT = "test"
BATCH_SIZE = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.18.0", "datasets", "accelerate", "safetensors")
    .add_local_python_source("phase0", "phase4")
    .add_local_file("results/phase0/baseline.json", remote_path="/baseline.json")
)
cache = modal.Volume.from_name("phase0-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def _letter_id(tokenizer, letter: str) -> int:
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{letter!r} encoded as {ids}, not one token")
    return ids[0]


def _prompt(tokenizer, item: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(item)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/cache": cache},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def score() -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(0)
    cache_dir = "/cache/huggingface"
    baseline = json.loads(Path("/baseline.json").read_text())
    dataset = load_dataset(EXAM, EXAM_CONFIG, split=EXAM_SPLIT, cache_dir=cache_dir)
    items = []
    for row in dataset:
        items.extend(iter_questions(row))
    if len(items) != 2000:
        raise RuntimeError(f"expected 2000 exam questions, got {len(items)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"padding_side={tokenizer.padding_side}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    letter_ids = {letter: _letter_id(tokenizer, letter) for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    prompts = [_prompt(tokenizer, item) for item in items]
    if baseline["example_prompt"] not in prompts:
        raise RuntimeError("Phase 0 exam prompt was not reproduced")

    shortcut_rows = []
    last_rows = []
    disagree = 0
    for start in range(0, len(items), BATCH_SIZE):
        batch_items = items[start : start + BATCH_SIZE]
        encoded = tokenizer(
            prompts[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        mask = encoded["attention_mask"].tolist()
        for index, item in enumerate(batch_items):
            option_ids = [option["id"] for option in item["options"]]
            letters = [option["letter"] for option in item["options"]]
            shortcut = phase0_index(mask[index])
            last = last_content_index(mask[index])
            shortcut_logits = [float(logits[index, shortcut][letter_ids[letter]]) for letter in letters]
            last_logits = [float(logits[index, last][letter_ids[letter]]) for letter in letters]
            shortcut_row = readout(shortcut_logits, item["label_index"], option_ids, 1.0)
            last_row = readout(last_logits, item["label_index"], option_ids, 1.0)
            shared = {
                "case_id": item["case_id"],
                "workflow": item["workflow"],
                "name": item["name"],
                "type": item["type"],
                "label": item["label"],
            }
            shortcut_row.update(shared)
            last_row.update(shared)
            shortcut_rows.append(shortcut_row)
            last_rows.append(last_row)
            if shortcut_row["prediction"] != last_row["prediction"]:
                disagree += 1
        done = min(start + BATCH_SIZE, len(items))
        if start == 0 or done == len(items) or done % 400 == 0:
            print(f"scored {done}/{len(items)}", flush=True)

    cache.commit()
    shortcut_summary = summarize_records(shortcut_rows, ("type", "workflow"))
    last_summary = summarize_records(last_rows, ("type", "workflow"))
    phase0_correct = round(baseline["accuracy"] * baseline["n_questions"])
    return {
        "model": MODEL_ID,
        "lora": False,
        "temperature": 1.0,
        "padding_side": tokenizer.padding_side,
        "prompt_matches_phase0": True,
        "index_disagrees": disagree,
        "phase0_recorded_correct": phase0_correct,
        "shortcut_correct": sum(row["correct"] for row in shortcut_rows),
        "last_content_correct": sum(row["correct"] for row in last_rows),
        "shortcut": shortcut_summary,
        "last_content": last_summary,
        "shortcut_matches_phase0": sum(row["correct"] for row in shortcut_rows) == phase0_correct,
        "records": [
            {
                "case_id": shortcut_rows[index]["case_id"],
                "name": shortcut_rows[index]["name"],
                "type": shortcut_rows[index]["type"],
                "shortcut_correct": shortcut_rows[index]["correct"],
                "last_correct": last_rows[index]["correct"],
                "shortcut_prediction": shortcut_rows[index]["prediction"],
                "last_prediction": last_rows[index]["prediction"],
            }
            for index in range(len(items))
        ],
    }


@app.local_entrypoint()
def main():
    payload = score.remote()
    records = payload.pop("records")
    out_dir = Path("results/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plain_last.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / "plain_last.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "shortcut_correct": payload["shortcut_correct"],
                "last_content_correct": payload["last_content_correct"],
                "phase0_recorded_correct": payload["phase0_recorded_correct"],
                "shortcut_matches_phase0": payload["shortcut_matches_phase0"],
                "index_disagrees": payload["index_disagrees"],
                "padding_side": payload["padding_side"],
            },
            indent=2,
        )
    )
