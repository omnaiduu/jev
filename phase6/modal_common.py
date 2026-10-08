"""Shared Unsloth load for the Phase 6 train and score containers."""

from __future__ import annotations

from pathlib import Path

from phase6.meta import DATASETS, MAX_LENGTH, MODEL_ID, RANK, RETIRED_ADAPTER_DIRS


def adapter_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}")
    path = Path(f"/lora/v2-{dataset}")
    retired = {item.rstrip("/") for item in RETIRED_ADAPTER_DIRS}
    if path.as_posix() in retired or path.name in {"adapter", "adapter-pass2"}:
        raise RuntimeError(f"refusing to write retired adapter path {path}")
    return path


def from_pretrained(loader, torch, model_name: str, checkpointing: bool):
    import inspect

    signature = inspect.signature(loader.from_pretrained)
    names = set(signature.parameters)
    has_var = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    options = {
        "load_in_4bit": False,
        "load_in_16bit": True,
        "dtype": torch.bfloat16,
        "max_seq_length": MAX_LENGTH,
        "full_finetuning": False,
    }
    if checkpointing:
        options["use_gradient_checkpointing"] = "unsloth"
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


def peft_kwargs() -> dict:
    return {
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "r": RANK,
        "lora_alpha": RANK,
        "lora_dropout": 0,
        "bias": "none",
        "random_state": 0,
    }


def attach_new_lora(FastVisionModel, model):
    import inspect

    kwargs = peft_kwargs()
    signature = inspect.signature(FastVisionModel.get_peft_model)
    if "finetune_audio_layers" in signature.parameters:
        kwargs["finetune_audio_layers"] = False
    return FastVisionModel.get_peft_model(model, **kwargs)


def tokenizer_of(processor):
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.padding_side != "right":
        raise RuntimeError("tokenizer refused right padding")
    return tokenizer


def device_of(model, torch):
    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) in (None, "meta"):
        return torch.device("cuda")
    return device


def letter_ids(tokenizer) -> dict[str, int]:
    from phase2.prompts import LETTERS

    mapping = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} encoded as {ids}, not one token")
        mapping[letter] = ids[0]
    return mapping


def disable_cache(model) -> None:
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
