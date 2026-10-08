"""Fit one temperature per question type on the calibration pile.

The LoRA is loaded and frozen. This call does not update it.
The container is single-use and scales down two seconds after this call.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase2.loss import text_only_names
from phase2.prompts import LETTERS, render_prompt
from phase3.fit import GRID, assert_calibration_row, fit_temperature

APP_NAME = "phase3-temperature"
MODEL_ID = "unsloth/gemma-4-E4B-it"
ADAPTER_DIR = "/lora/adapter"
MAX_LENGTH = 2048
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
    .add_local_python_source("phase0", "phase2", "phase3")
    .add_local_file("data/phase1/calibration.jsonl", remote_path="/data/calibration.jsonl")
)
cache = modal.Volume.from_name("phase2-hf-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("phase2-lora", create_if_missing=True)
app = modal.App(APP_NAME)


def _tokenizer_of(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _has_lora(model) -> bool:
    return any("lora_" in name for name, _param in model.named_parameters())


def _from_pretrained(loader, torch, model_name: str):
    """16-bit load. Drop load_in_16bit only if this Unsloth build rejects it."""
    import inspect

    signature = inspect.signature(loader.from_pretrained)
    names = set(signature.parameters)
    has_var = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    options = {
        "load_in_4bit": False,
        "load_in_16bit": True,
        "dtype": torch.bfloat16,
        "max_seq_length": MAX_LENGTH,
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
    """Same LoRA attachment the trainer used, then the saved tensors."""
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
    """Prefer the saved adapter folder. Fall back to the training checkpoint."""
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


def _batch(tokenizer, rows, letter_ids, device):
    import torch

    encoded_rows = []
    for row in rows:
        ids = tokenizer(render_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_LENGTH:
            ids = ids[-MAX_LENGTH:]
        encoded_rows.append(ids)
    width = max(len(ids) for ids in encoded_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    mask = []
    for ids in encoded_rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    selected = []
    labels = []
    for row in rows:
        selected.append(
            torch.tensor(
                [letter_ids[LETTERS[index]] for index in range(len(row["options"]))],
                dtype=torch.long,
                device=device,
            )
        )
        labels.append(row["target"].index(1.0))
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(mask, device=device),
        selected,
        labels,
    )


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/cache": cache, "/lora": lora_volume},
    env={"HF_HOME": "/cache/huggingface"},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def fit() -> dict:
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    rows = []
    with open("/data/calibration.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 4000:
        raise RuntimeError(f"calibration pile has {len(rows)} rows, expected 4000")
    for row in rows:
        assert_calibration_row(row)
    if not Path(ADAPTER_DIR, "adapter_config.json").exists():
        raise RuntimeError("adapter_config.json is missing; the LoRA was not saved")

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
    print(f"trainable tensors left on {len(trainable)}", flush=True)

    tokenizer = _tokenizer_of(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = _letter_ids(tokenizer)
    device = _device(model, torch)

    records = []
    steps = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for step in range(steps):
        batch_rows = rows[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        input_ids, mask, selected, labels = _batch(tokenizer, batch_rows, letter_ids, device)
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        lengths = mask.sum(dim=1) - 1
        for index, row in enumerate(batch_rows):
            last = logits[index, int(lengths[index])]
            letter_logits = last.index_select(0, selected[index]).float().tolist()
            records.append(
                {
                    "type": row["type"],
                    "logits": letter_logits,
                    "label_index": labels[index],
                }
            )
        if step % 50 == 0 or step + 1 == steps:
            print(f"scored {min((step + 1) * BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)

    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
    fitted = {name: fit_temperature(group) for name, group in sorted(by_type.items())}
    payload = {
        "fit_on": "data/phase1/calibration.jsonl",
        "not_fit_on": ["train", "typed-decisions", "snli"],
        "rows": len(rows),
        "grid": GRID,
        "merged": False,
        "weights_updated": False,
        "load_method": load_method,
        "types": fitted,
        "missing_types": ["score"] if "score" not in fitted else [],
        "missing_reason": (
            "The calibration pile has noul and choice only. "
            "score was not given a temperature."
            if "score" not in fitted
            else ""
        ),
    }
    Path("/lora/temperature.json").write_text(json.dumps(payload) + "\n")
    lora_volume.commit()
    return payload


@app.local_entrypoint()
def main():
    payload = fit.remote()
    out_dir = Path("results/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "temperature.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    brief = {
        name: {
            "n": item["n"],
            "T": item["T"],
            "nll_at_1": item["nll_at_1"],
            "nll": item["nll"],
            "accuracy": item["accuracy"],
            "ece_at_1": item["ece_at_1"],
            "ece_at_T": item["ece_at_T"],
        }
        for name, item in payload["types"].items()
    }
    print(json.dumps({"types": brief, "missing_types": payload["missing_types"]}, indent=2))
