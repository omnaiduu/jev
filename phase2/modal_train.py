"""One pass of letter cross-entropy on the Phase 1 train pile.

Unsloth loads Gemma 4 E4B and attaches the LoRA. The loss is a PyTorch loop.
The adapter is saved on its own. It is not merged into the base model.
The container is single-use and scales down two seconds after this call.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import modal

from phase2.loss import letter_cross_entropy, text_only_names
from phase2.prompts import LETTERS, render_prompt

APP_NAME = "phase2-lora-train"
MODEL_ID = "unsloth/gemma-4-E4B-it"
MAX_LENGTH = 2048
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
RANK = 16

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("unsloth")
    .add_local_python_source("phase2")
    .add_local_file("data/phase1/train.jsonl", remote_path="/data/train.jsonl")
)
cache = modal.Volume.from_name("phase2-hf-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("phase2-lora", create_if_missing=True)
app = modal.App(APP_NAME)


def _lr(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (0.1 + 0.9 * cosine)


def _tokenizer_of(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _from_pretrained(loader, torch):
    """16-bit load. Drop load_in_16bit only if this Unsloth build rejects it."""
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
        "use_gradient_checkpointing": "unsloth",
        "full_finetuning": False,
    }
    if not has_var:
        options = {key: value for key, value in options.items() if key in names}

    def call(selected):
        if "model_name" in names:
            return loader.from_pretrained(model_name=MODEL_ID, **selected)
        return loader.from_pretrained(MODEL_ID, **selected)

    try:
        return call(options)
    except TypeError as error:
        if "load_in_16bit" not in str(error):
            raise
        options.pop("load_in_16bit", None)
        return call(options)


def _device(model, torch):
    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) in (None, "meta"):
        return torch.device("cuda")
    return device


def _adapter_report(path: Path) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    names = sorted(item.name for item in files)
    if "adapter_config.json" not in names:
        raise RuntimeError("adapter_config.json missing after save: " + ", ".join(names[:12]))
    if not any(name.startswith("adapter_model") for name in names):
        raise RuntimeError("adapter weights missing after save: " + ", ".join(names[:12]))
    oversized = [
        f"{item.name}:{item.stat().st_size}"
        for item in files
        if item.stat().st_size > 200 * 1024 * 1024
    ]
    if oversized:
        raise RuntimeError("save wrote base weights: " + ", ".join(oversized))
    return {"adapter_bytes": sum(item.stat().st_size for item in files), "adapter_files": names}


def _letter_ids(tokenizer) -> dict[str, int]:
    mapping = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} encoded as {ids}, not one token")
        mapping[letter] = ids[0]
    return mapping


def _batch(tokenizer, rows: list[dict], letter_ids: dict[str, int], device):
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
    targets = []
    selected = []
    for row in rows:
        targets.append(torch.tensor(row["target"], dtype=torch.float32, device=device))
        selected.append(
            torch.tensor(
                [letter_ids[LETTERS[index]] for index in range(len(row["options"]))],
                dtype=torch.long,
                device=device,
            )
        )
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(mask, device=device),
        selected,
        targets,
    )


@app.function(
    image=image,
    gpu="L40S",
    timeout=6 * 60 * 60,
    volumes={"/cache": cache, "/lora": lora_volume},
    env={"HF_HOME": "/cache/huggingface"},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def train() -> dict:
    import inspect

    from unsloth import FastVisionModel
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
    torch.manual_seed(0)
    rows = []
    with open("/data/train.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 1000:
        raise RuntimeError(f"train pile is unexpectedly small: {len(rows)}")

    model, processor = _from_pretrained(FastVisionModel, torch)
    peft_kwargs = {
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
    signature = inspect.signature(FastVisionModel.get_peft_model)
    if "finetune_audio_layers" in signature.parameters:
        peft_kwargs["finetune_audio_layers"] = False
    model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    if hasattr(FastVisionModel, "for_training"):
        FastVisionModel.for_training(model)
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    text_only_names(trainable)
    print(f"trainable tensors {len(trainable)} sample {trainable[:4]}", flush=True)
    device = _device(model, torch)
    tokenizer = _tokenizer_of(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = _letter_ids(tokenizer)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
    )
    steps = math.ceil(len(rows) / BATCH_SIZE)
    losses = []
    model.train()
    for step in range(steps):
        batch_rows = rows[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        lr = _lr(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        input_ids, mask, selected, targets = _batch(
            tokenizer, batch_rows, letter_ids, device
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        lengths = mask.sum(dim=1) - 1
        loss = letter_cross_entropy(logits, lengths, selected, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (param for param in model.parameters() if param.requires_grad),
            1.0,
        )
        optimizer.step()
        value = float(loss.detach())
        losses.append(round(value, 6))
        if step % 25 == 0 or step + 1 == steps:
            print(f"step {step + 1}/{steps} loss {value:.4f} lr {lr:.6f}", flush=True)

    out_dir = Path("/lora/adapter")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    adapter = _adapter_report(out_dir)
    lora_volume.commit()

    early = sum(losses[:8]) / 8
    late_smoke = sum(losses[8:16]) / 8
    full_early = sum(losses[:100]) / 100
    full_late = sum(losses[-100:]) / 100
    return {
        "model": MODEL_ID,
        "rows": len(rows),
        "steps": steps,
        "batch_size": BATCH_SIZE,
        "rank": RANK,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "load_in_4bit": False,
        "merged": False,
        "adapter_bytes": adapter["adapter_bytes"],
        "adapter_files": adapter["adapter_files"],
        "trainable_tensors": len(trainable),
        "trainable_sample": trainable[:12],
        "smoke_first_8_mean": early,
        "smoke_next_8_mean": late_smoke,
        "smoke_loss_fell": late_smoke < early,
        "epoch_first_100_mean": full_early,
        "epoch_last_100_mean": full_late,
        "epoch_loss_fell": full_late < full_early,
        "losses_every_25": losses[::25],
        "first_16_losses": losses[:16],
    }


@app.local_entrypoint()
def main():
    payload = train.remote()
    out_dir = Path("results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "losses_every_25"}, indent=2))
