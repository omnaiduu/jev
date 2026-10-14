"""Phase 0: score Gemma 4 E4B on typed-decisions with a letter-logit readout.

No LoRA and no training. One forward pass per question. Softmax is taken only
over the letter tokens of that question's options.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase0.items import SYSTEM_PROMPT, iter_questions, render_user_prompt
from phase0.metrics import accuracy, brier_score, expected_calibration_error, softmax

APP_NAME = "phase0-gemma-e4b-baseline"
MODEL_ID = "google/gemma-4-E4B-it"
EXAM = "LocalLLaMA/typed-decisions"
EXAM_CONFIG = "all"
EXAM_SPLIT = "test"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.18.0", "datasets", "accelerate", "safetensors")
    .add_local_python_source("phase0")
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


def _summarize(records: list[dict]) -> dict:
    correct = [record["correct"] for record in records]
    confidences = [record["confidence"] for record in records]
    probabilities = [record["probabilities"] for record in records]
    label_indexes = [record["label_index"] for record in records]
    summary = {
        "n_questions": len(records),
        "n_cases": len({record["case_id"] for record in records}),
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier_score(probabilities, label_indexes),
    }
    for key in ("type", "workflow"):
        summary[key] = {}
        values = sorted({record[key] for record in records})
        for value in values:
            group = [record for record in records if record[key] == value]
            summary[key][value] = {
                "n": len(group),
                "accuracy": accuracy([record["correct"] for record in group]),
                "ece": expected_calibration_error(
                    [record["confidence"] for record in group],
                    [record["correct"] for record in group],
                ),
                "brier": brier_score(
                    [record["probabilities"] for record in group],
                    [record["label_index"] for record in group],
                ),
            }
    return summary


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/cache": cache},
)
def score_exam(batch_size: int = 8) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(0)
    cache_dir = "/cache/huggingface"
    dataset = load_dataset(EXAM, EXAM_CONFIG, split=EXAM_SPLIT, cache_dir=cache_dir)
    items = []
    for row in dataset:
        items.extend(iter_questions(row))
    if len(items) != 2000:
        raise RuntimeError(f"expected 2000 exam questions, got {len(items)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    letter_ids = {letter: _letter_id(tokenizer, letter) for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    prompts = [_prompt(tokenizer, item) for item in items]

    records = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        for index, item in enumerate(batch_items):
            last = logits[index, lengths[index]]
            option_logits = [float(last[letter_ids[option["letter"]]]) for option in item["options"]]
            probabilities = softmax(option_logits)
            pick = max(range(len(probabilities)), key=probabilities.__getitem__)
            records.append(
                {
                    "case_id": item["case_id"],
                    "workflow": item["workflow"],
                    "name": item["name"],
                    "type": item["type"],
                    "label": item["label"],
                    "label_index": item["label_index"],
                    "prediction": item["options"][pick]["id"],
                    "correct": pick == item["label_index"],
                    "confidence": probabilities[pick],
                    "probabilities": probabilities,
                    "option_ids": [option["id"] for option in item["options"]],
                }
            )
        print(f"scored {min(start + batch_size, len(items))}/{len(items)}", flush=True)

    cache.commit()
    summary = _summarize(records)
    summary.update(
        {
            "model": MODEL_ID,
            "exam": EXAM,
            "exam_config": EXAM_CONFIG,
            "exam_split": EXAM_SPLIT,
            "method": (
                "One forward pass per question. Softmax is restricted to the "
                "single-token letters A, B, C, ... bound to the options in dataset order. "
                "No LoRA, no temperature, thinking off."
            ),
            "example_prompt": prompts[0],
        }
    )
    return {"summary": summary, "records": records}


@app.local_entrypoint()
def main():
    payload = score_exam.remote()
    out_dir = Path("results/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "baseline.json"
    records_path = out_dir / "predictions.jsonl"
    summary_path.write_text(json.dumps(payload["summary"], indent=2) + "\n")
    with records_path.open("w") as handle:
        for record in payload["records"]:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(payload["summary"], indent=2))
