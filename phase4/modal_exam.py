"""Score the frozen LoRA on the exam, a flipped option order, and held-out SNLI.

The exam prompt and the tokenizer padding call match Phase 0. Temperature comes
from the Phase 3 file. The LoRA is not updated. The container is single-use.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase0.items import LETTERS, SYSTEM_PROMPT, iter_questions, render_user_prompt
from phase2.loss import text_only_names
from phase2.prompts import render_prompt
from phase4.score import (
    flip_item,
    flip_report,
    gate,
    readout,
    slot_histogram,
    summarize_records,
    temperature_for,
    temperatures_from_phase3,
)

APP_NAME = "phase4-exam"
MODEL_ID = "unsloth/gemma-4-E4B-it"
PHASE0_MODEL_ID = "google/gemma-4-E4B-it"
ADAPTER_DIR = "/lora/adapter"
EXAM = "LocalLLaMA/typed-decisions"
EXAM_CONFIG = "all"
EXAM_SPLIT = "test"
TOKEN_MAX = 4096
MODEL_MAX = 2048
BATCH_SIZE = 8

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
    .add_local_python_source("phase0", "phase2", "phase4")
    .add_local_file("results/phase0/baseline.json", remote_path="/baseline.json")
    .add_local_file("results/phase3/temperature.json", remote_path="/temperature.json")
    .add_local_file("data/phase1/held_out.jsonl", remote_path="/data/held_out.jsonl")
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


def _attach_checkpoint(FastVisionModel, torch, model):
    import inspect

    from peft import set_peft_model_state_dict

    peft_kwargs = {
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "random_state": 0,
    }
    signature = inspect.signature(FastVisionModel.get_peft_model)
    if "finetune_audio_layers" in signature.parameters:
        peft_kwargs["finetune_audio_layers"] = False
    model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    state_path = Path("/lora/adapter_state.pt")
    if not state_path.exists():
        raise RuntimeError("adapter_state.pt is missing")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state)
    if not _has_lora(model):
        raise RuntimeError("LoRA tensors missing after loading adapter_state.pt")
    return model


def _load(FastVisionModel, torch):
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model, processor = _from_pretrained(FastVisionModel, torch, ADAPTER_DIR)
        if _has_lora(model):
            print("loaded adapter directory", flush=True)
            return model, processor, "adapter_directory"
        print("adapter directory loaded without LoRA tensors", flush=True)
        del model
        torch.cuda.empty_cache()
    except Exception as error:
        print(f"adapter directory load failed: {type(error).__name__}: {error}", flush=True)
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)

    model, processor = _from_pretrained(FastVisionModel, torch, MODEL_ID)
    model = _attach_checkpoint(FastVisionModel, torch, model)
    print("loaded base model plus adapter_state.pt", flush=True)
    return model, processor, "adapter_state"


def _tokenizer_of(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _device(model, torch):
    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) in (None, "meta"):
        return torch.device("cuda")
    return device


def _letter_ids(tokenizer) -> dict[str, int]:
    mapping = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} encoded as {ids}, not one token")
        mapping[letter] = ids[0]
    return mapping


def _exam_prompt(tokenizer, item: dict) -> str:
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


def _indexes(attention_mask) -> tuple[list[int], list[int]]:
    """Phase 0 used mask.sum()-1. The last real token is the last 1 in the mask."""
    phase0 = []
    real = []
    for row in attention_mask.tolist():
        ones = [index for index, bit in enumerate(row) if bit]
        if not ones:
            raise RuntimeError("attention row is empty")
        phase0.append(sum(row) - 1)
        real.append(ones[-1])
    return phase0, real


def _option_logits(last, letter_ids, letters, torch) -> list[float]:
    selected = torch.tensor([letter_ids[letter] for letter in letters], device=last.device)
    return last.index_select(0, selected).float().tolist()


def _score_exam_batch(model, tokenizer, letter_ids, items, prompts, temperatures, device, torch):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TOKEN_MAX,
    ).to(device)
    kwargs = {key: value for key, value in encoded.items()}
    kwargs["use_cache"] = False
    output = model(**kwargs)
    logits = output.logits if hasattr(output, "logits") else output[0]
    phase0_index, real_index = _indexes(encoded["attention_mask"])
    official = []
    last_real = []
    for index, item in enumerate(items):
        letters = [option["letter"] for option in item["options"]]
        option_ids = [option["id"] for option in item["options"]]
        temperature = temperature_for(item["type"], temperatures)
        official_logits = _option_logits(logits[index, phase0_index[index]], letter_ids, letters, torch)
        real_logits = _option_logits(logits[index, real_index[index]], letter_ids, letters, torch)
        row = readout(official_logits, item["label_index"], option_ids, temperature)
        row.update(
            {
                "case_id": item["case_id"],
                "workflow": item["workflow"],
                "name": item["name"],
                "type": item["type"],
                "label": item["label"],
            }
        )
        official.append(row)
        real_row = readout(real_logits, item["label_index"], option_ids, temperature)
        real_row.update({"case_id": item["case_id"], "name": item["name"]})
        last_real.append(real_row)
    return official, last_real


def _score_snli_batch(model, tokenizer, letter_ids, rows, device, torch, temperature: float):
    encoded_rows = []
    for row in rows:
        ids = tokenizer(render_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
        if len(ids) > 2048:
            ids = ids[-2048:]
        encoded_rows.append(ids)
    width = max(len(ids) for ids in encoded_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    mask = []
    for ids in encoded_rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    input_tensor = torch.tensor(input_ids, device=device)
    mask_tensor = torch.tensor(mask, device=device)
    output = model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    lengths = mask_tensor.sum(dim=1) - 1
    records = []
    for index, row in enumerate(rows):
        letters = [LETTERS[option_index] for option_index in range(len(row["options"]))]
        label_index = row["target"].index(1.0)
        option_logits = _option_logits(logits[index, int(lengths[index])], letter_ids, letters, torch)
        record = readout(option_logits, label_index, row["options"], temperature)
        record.update(
            {
                "case_id": row["id"],
                "name": "relation",
                "type": row["type"],
                "source": row["source"],
                "label": row["options"][label_index],
            }
        )
        records.append(record)
    return records


def _run_batches(score_fn, items, prompts, label: str):
    records = []
    last_real = []
    total = len(items)
    for start in range(0, total, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, total)
        batch_prompts = prompts[start:stop] if prompts is not None else None
        official, real = score_fn(items[start:stop], batch_prompts)
        records.extend(official)
        last_real.extend(real)
        if start == 0 or stop == total or stop % 400 == 0:
            print(f"{label} {stop}/{total}", flush=True)
    return records, last_real


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
def exam() -> dict:
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    temperatures = temperatures_from_phase3(json.loads(Path("/temperature.json").read_text()))
    baseline = json.loads(Path("/baseline.json").read_text())
    if baseline.get("exam") != EXAM or baseline.get("n_questions") != 2000:
        raise RuntimeError("Phase 0 baseline file is not the typed-decisions test")
    if not Path(ADAPTER_DIR, "adapter_config.json").exists():
        raise RuntimeError("adapter_config.json is missing")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        EXAM,
        EXAM_CONFIG,
        split=EXAM_SPLIT,
        cache_dir="/phase0cache/huggingface",
    )
    items = []
    for row in dataset:
        items.extend(iter_questions(row))
    if len(items) != 2000:
        raise RuntimeError(f"expected 2000 exam questions, got {len(items)}")
    flipped_items = [flip_item(item) for item in items]

    tokenizer = AutoTokenizer.from_pretrained(PHASE0_MODEL_ID, cache_dir="/phase0cache/huggingface")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    padding_side = tokenizer.padding_side
    print(f"exam tokenizer padding_side={padding_side}", flush=True)
    letter_ids = _letter_ids(tokenizer)
    prompts = [_exam_prompt(tokenizer, item) for item in items]
    example = baseline["example_prompt"]
    if example not in prompts:
        raise RuntimeError("Phase 0 exam prompt was not reproduced: " + prompts[0][:240])
    flip_prompts = [_exam_prompt(tokenizer, item) for item in flipped_items]

    held_out = []
    with open("/data/held_out.jsonl") as handle:
        for line in handle:
            if line.strip():
                held_out.append(json.loads(line))
    if len(held_out) != 2000 or any(row.get("source") != "snli" for row in held_out):
        raise RuntimeError("held-out file is not the 2000 SNLI rows")

    torch.set_grad_enabled(False)
    model, processor, load_method = _load(FastVisionModel, torch)
    if hasattr(FastVisionModel, "for_inference"):
        FastVisionModel.for_inference(model)
    model.eval()
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if trainable:
        text_only_names(trainable)
    if not _has_lora(model):
        raise RuntimeError("scoring ran without LoRA tensors")
    model_tokenizer = _tokenizer_of(processor)
    model_letters = _letter_ids(model_tokenizer)
    if model_letters != letter_ids:
        raise RuntimeError("exam tokenizer letter ids differ from the LoRA tokenizer")
    device = _device(model, torch)

    def score_items(batch_items, batch_prompts):
        return _score_exam_batch(
            model, tokenizer, letter_ids, batch_items, batch_prompts, temperatures, device, torch
        )

    official, last_real = _run_batches(score_items, items, prompts, "exam")
    flipped, _flipped_real = _run_batches(score_items, flipped_items, flip_prompts, "flip")
    if model_tokenizer.pad_token_id is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    snli_temperature = temperature_for("choice", temperatures)
    snli = []
    for start in range(0, len(held_out), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(held_out))
        snli.extend(
            _score_snli_batch(
                model,
                model_tokenizer,
                letter_ids,
                held_out[start:stop],
                device,
                torch,
                snli_temperature,
            )
        )
        if start == 0 or stop == len(held_out) or stop % 400 == 0:
            print(f"snli {stop}/{len(held_out)}", flush=True)

    index_changed = sum(
        left["prediction"] != right["prediction"] for left, right in zip(official, last_real)
    )
    summary = summarize_records(official, ("type", "workflow"))
    real_summary = summarize_records(last_real, ())
    decision = gate(summary, baseline)
    report = flip_report(official, flipped)
    snli_summary = summarize_records(snli, ("type", "source"))
    return {
        "model": MODEL_ID,
        "phase0_tokenizer": PHASE0_MODEL_ID,
        "load_method": load_method,
        "merged": False,
        "weights_updated": False,
        "temperatures": temperatures,
        "padding_side": padding_side,
        "prompt_matches_phase0": True,
        "phase0_index_differs": index_changed,
        "exam": summary,
        "exam_last_real_token": {
            "n_questions": real_summary["n_questions"],
            "accuracy": real_summary["accuracy"],
            "ece": real_summary["ece"],
            "brier": real_summary["brier"],
        },
        "slots": slot_histogram(official),
        "gate": decision,
        "flip": report,
        "flip_accuracy": summarize_records(flipped, ())["accuracy"],
        "snli": snli_summary,
        "records": official,
        "flip_records": [
            {
                "case_id": row["case_id"],
                "name": row["name"],
                "prediction": row["prediction"],
                "correct": row["correct"],
            }
            for row in flipped
        ],
        "snli_records": snli,
    }


@app.local_entrypoint()
def main():
    payload = exam.remote()
    out_dir = Path("results/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = payload.pop("records")
    flip_records = payload.pop("flip_records")
    snli_records = payload.pop("snli_records")
    (out_dir / "exam.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, rows in (
        ("predictions.jsonl", records),
        ("flip.jsonl", flip_records),
        ("snli.jsonl", snli_records),
    ):
        with (out_dir / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    brief = {
        "gate": payload["gate"],
        "flip": payload["flip"],
        "snli_accuracy": payload["snli"]["accuracy"],
        "snli_ece": payload["snli"]["ece"],
        "padding_side": payload["padding_side"],
        "phase0_index_differs": payload["phase0_index_differs"],
        "load_method": payload["load_method"],
    }
    print(json.dumps(brief, indent=2))
