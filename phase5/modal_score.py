"""Score the pass-2 LoRA on the exam at both readout indexes.

The shortcut is mask.sum()-1, which Phase 0 and Phase 4 published.
The last content token is the last mask bit that is 1.
Temperatures are the Phase 3 values. The adapter file is not updated.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase0.items import SYSTEM_PROMPT, iter_questions, render_user_prompt
from phase2.loss import text_only_names
from phase4.score import (
    last_content_index,
    phase0_index,
    readout,
    summarize_records,
    temperature_for,
    temperatures_from_phase3,
)

APP_NAME = "phase5-exam"
MODEL_ID = "unsloth/gemma-4-E4B-it"
PHASE0_MODEL_ID = "google/gemma-4-E4B-it"
ADAPTER_DIR = "/lora/adapter-pass2"
EXAM = "LocalLLaMA/typed-decisions"
EXAM_CONFIG = "all"
EXAM_SPLIT = "test"
BATCH_SIZE = 8
TOKEN_MAX = 4096
MODEL_MAX = 2048

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("unsloth")
    .pip_install(
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        extra_options="--force-reinstall --no-deps",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("datasets")
    .add_local_python_source("phase0", "phase2", "phase4", "phase5")
    .add_local_file("results/phase0/baseline.json", remote_path="/baseline.json")
    .add_local_file("results/phase3/temperature.json", remote_path="/temperature.json")
)
cache = modal.Volume.from_name("phase2-hf-cache", create_if_missing=True)
phase0_cache = modal.Volume.from_name("phase0-hf-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("phase2-lora", create_if_missing=True)
app = modal.App(APP_NAME)


def _has_lora(model) -> bool:
    return any("lora_" in name for name, _param in model.named_parameters())


def _from_pretrained(loader, torch, model_name: str):
    import inspect

    signature = inspect.signature(loader.from_pretrained)
    names = set(signature.parameters)
    has_var = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    options = {
        "load_in_4bit": False,
        "load_in_16bit": True,
        "dtype": torch.bfloat16,
        "max_seq_length": MODEL_MAX,
        "full_finetuning": False,
    }
    if not has_var:
        options = {key: value for key, value in options.items() if key in names}

    def call(selected):
        if "model_name" in names:
            return loader.from_pretrained(model_name=model_name, **selected)
        return loader.from_pretrained(model_name, **selected)

    try:
        return call(options)
    except TypeError as error:
        if "load_in_16bit" not in str(error):
            raise
        options.pop("load_in_16bit", None)
        return call(options)


def _load(FastVisionModel, torch):
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model, processor = _from_pretrained(FastVisionModel, torch, ADAPTER_DIR)
        if _has_lora(model):
            print("loaded pass2 adapter", flush=True)
            return model, processor
        del model
        torch.cuda.empty_cache()
    except Exception as error:
        print(f"pass2 adapter load failed: {type(error).__name__}: {error}", flush=True)
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
    raise RuntimeError("pass2 adapter did not load")


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
    volumes={"/cache": cache, "/lora": lora_volume, "/phase0cache": phase0_cache},
    env={"HF_HOME": "/cache/huggingface"},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def score() -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from unsloth import FastVisionModel

    temperatures = temperatures_from_phase3(json.loads(Path("/temperature.json").read_text()))
    baseline = json.loads(Path("/baseline.json").read_text())
    dataset = load_dataset(EXAM, EXAM_CONFIG, split=EXAM_SPLIT, cache_dir="/phase0cache/huggingface")
    items = []
    for row in dataset:
        items.extend(iter_questions(row))
    if len(items) != 2000:
        raise RuntimeError(f"expected 2000 exam questions, got {len(items)}")
    tokenizer = AutoTokenizer.from_pretrained(PHASE0_MODEL_ID, cache_dir="/phase0cache/huggingface")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter} encoded as {ids}")
        letter_ids[letter] = ids[0]
    prompts = [_prompt(tokenizer, item) for item in items]
    if baseline["example_prompt"] not in prompts:
        raise RuntimeError("Phase 0 exam prompt was not reproduced")

    torch.set_grad_enabled(False)
    model, _processor = _load(FastVisionModel, torch)
    if hasattr(FastVisionModel, "for_inference"):
        FastVisionModel.for_inference(model)
    model.eval()
    model.config.use_cache = False
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if trainable:
        text_only_names(trainable)
    device = model.device if getattr(getattr(model, "device", None), "type", None) not in (None, "meta") else torch.device("cuda")

    shortcut_rows = []
    last_rows = []
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        encoded = tokenizer(
            prompts[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TOKEN_MAX,
        ).to(device)
        output = model(**{key: value for key, value in encoded.items()}, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        mask = encoded["attention_mask"].tolist()
        for index, item in enumerate(batch):
            option_ids = [option["id"] for option in item["options"]]
            letters = [option["letter"] for option in item["options"]]
            temperature = temperature_for(item["type"], temperatures)
            for bucket, position in (
                (shortcut_rows, phase0_index(mask[index])),
                (last_rows, last_content_index(mask[index])),
            ):
                selected = [float(logits[index, position][letter_ids[letter]]) for letter in letters]
                row = readout(selected, item["label_index"], option_ids, temperature)
                row.update(
                    {
                        "case_id": item["case_id"],
                        "workflow": item["workflow"],
                        "name": item["name"],
                        "type": item["type"],
                        "label": item["label"],
                    }
                )
                bucket.append(row)
        done = min(start + BATCH_SIZE, len(items))
        if start == 0 or done == len(items) or done % 400 == 0:
            print(f"scored {done}/{len(items)}", flush=True)

    return {
        "adapter": "adapter-pass2",
        "merged": False,
        "weights_updated": False,
        "temperatures": temperatures,
        "padding_side": tokenizer.padding_side,
        "shortcut_correct": sum(row["correct"] for row in shortcut_rows),
        "last_content_correct": sum(row["correct"] for row in last_rows),
        "shortcut": summarize_records(shortcut_rows, ("type",)),
        "last_content": summarize_records(last_rows, ("type",)),
    }


@app.local_entrypoint()
def main():
    payload = score.remote()
    out = Path("results/phase5")
    out.mkdir(parents=True, exist_ok=True)
    (out / "exam.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shortcut_correct": payload["shortcut_correct"],
                "last_content_correct": payload["last_content_correct"],
                "shortcut_by_type": {
                    name: item["n"] and int(round(item["accuracy"] * item["n"]))
                    for name, item in payload["shortcut"]["type"].items()
                },
                "last_by_type": {
                    name: int(round(item["accuracy"] * item["n"]))
                    for name, item in payload["last_content"]["type"].items()
                },
            },
            indent=2,
        )
    )
