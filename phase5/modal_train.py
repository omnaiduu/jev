"""One more pass on the existing LoRA.

Starts from the Phase 2 adapter. Trains on data/phase5/pass.jsonl only.
Saves a new adapter at /lora/adapter-pass2. Does not overwrite adapter/.
Each call stops after about 8 minutes of updates so the client stays under
the Modal cancel window. The container is single-use.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

from phase2.loss import letter_cross_entropy, merge_every_25, text_only_names
from phase2.prompts import LETTERS, render_prompt

APP_NAME = "phase5-lora-pass"
MODEL_ID = "unsloth/gemma-4-E4B-it"
ADAPTER_DIR = "/lora/adapter"
OUT_DIR = "/lora/adapter-pass2"
PROGRESS = "/lora/pass2_progress.json"
STATE = "/lora/pass2_adapter_state.pt"
MAX_LENGTH = 2048
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_STEPS = 20
CHUNK_TRAIN_SECONDS = 8 * 60

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
    .add_local_python_source("phase2", "phase5")
    .add_local_file("data/phase5/pass.jsonl", remote_path="/data/pass.jsonl")
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


def _from_pretrained(loader, torch, model_name: str, checkpointing: bool):
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


def _device(model, torch):
    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) in (None, "meta"):
        return torch.device("cuda")
    return device


def _tokenizer_of(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _letter_ids(tokenizer) -> dict[str, int]:
    mapping = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} encoded as {ids}, not one token")
        mapping[letter] = ids[0]
    return mapping


def _load_progress() -> dict:
    path = Path(PROGRESS)
    if not path.exists():
        return {"step": 0, "losses_every_25": []}
    payload = json.loads(path.read_text())
    payload.setdefault("losses_every_25", [])
    payload["step"] = int(payload.get("step", 0))
    return payload


def _batch(tokenizer, rows, letter_ids, device, torch):
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


def _checkpoint(model, losses, completed, steps, start, torch) -> dict:
    from peft import get_peft_model_state_dict

    prior = _load_progress()
    payload = {
        "step": completed,
        "steps": steps,
        "last_loss": losses[-1],
        "losses_every_25": merge_every_25(prior["losses_every_25"], start, losses),
    }
    Path(PROGRESS).write_text(json.dumps(payload) + "\n")
    state = {name: tensor.detach().to("cpu") for name, tensor in get_peft_model_state_dict(model).items()}
    torch.save(state, STATE)
    lora_volume.commit()
    print(f"checkpoint {completed}/{steps}", flush=True)
    return payload


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
def train() -> dict:
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    rows = []
    with open("/data/pass.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 1000:
        raise RuntimeError(f"pass2 pile is unexpectedly small: {len(rows)}")
    if any(row["source"] in ("boolq", "typed-decisions") or "typed-decisions" in row["source"] for row in rows):
        raise RuntimeError("pass2 pile includes BoolQ or the exam")

    progress = _load_progress()
    start = progress["step"]
    try:
        model, processor = _from_pretrained(FastVisionModel, torch, ADAPTER_DIR, checkpointing=True)
    except Exception as error:
        print(f"adapter load with checkpointing failed: {type(error).__name__}: {error}", flush=True)
        model, processor = _from_pretrained(FastVisionModel, torch, ADAPTER_DIR, checkpointing=False)
    if hasattr(FastVisionModel, "for_training"):
        FastVisionModel.for_training(model)
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    if start:
        from peft import set_peft_model_state_dict

        if not Path(STATE).exists():
            raise RuntimeError(f"progress says step {start} but pass2 state is missing")
        set_peft_model_state_dict(model, torch.load(STATE, map_location="cpu", weights_only=True))
        print(f"resumed at step {start}", flush=True)
    else:
        print("loaded phase 2 adapter", flush=True)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if trainable:
        text_only_names(trainable)
    print(f"trainable tensors {len(trainable)}", flush=True)
    device = _device(model, torch)
    tokenizer = _tokenizer_of(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = _letter_ids(tokenizer)
    steps = math.ceil(len(rows) / BATCH_SIZE)
    if start > steps:
        raise RuntimeError(f"checkpoint step {start} is past {steps}")
    if start >= steps:
        print("pass already finished", flush=True)
        return {"status": "done", "step": start, "steps": steps, "rows": len(rows)}

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=LEARNING_RATE)
    losses = []
    model.train()
    deadline = time.time() + CHUNK_TRAIN_SECONDS
    completed = start
    for step in range(start, steps):
        batch_rows = rows[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        lr = _lr(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        input_ids, mask, selected, targets = _batch(tokenizer, batch_rows, letter_ids, device, torch)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        lengths = mask.sum(dim=1) - 1
        loss = letter_cross_entropy(logits, lengths, selected, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(round(value, 6))
        completed = step + 1
        if step % 25 == 0 or completed == steps:
            print(f"step {completed}/{steps} loss {value:.4f} lr {lr:.6f}", flush=True)
        if completed == steps:
            break
        if time.time() > deadline:
            saved = _checkpoint(model, losses, completed, steps, start, torch)
            return {"status": "partial", "step": completed, "steps": steps, "last_loss": losses[-1], "rows": len(rows), "losses_every_25": saved["losses_every_25"]}

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    saved = _checkpoint(model, losses, completed, steps, start, torch)
    payload = {
        "status": "done",
        "rows": len(rows),
        "steps": steps,
        "resumed_from": start,
        "learning_rate": LEARNING_RATE,
        "merged": False,
        "overwrote_phase2_adapter": False,
        "trainable_tensors": len(trainable),
        "early_every_25_mean": sum(saved["losses_every_25"][:4]) / max(1, min(4, len(saved["losses_every_25"]))),
        "epoch_last_100_mean": sum(losses[-100:]) / max(1, min(100, len(losses))),
        "losses_every_25": saved["losses_every_25"],
    }
    Path("/lora/pass2_train.json").write_text(json.dumps(payload, indent=2) + "\n")
    lora_volume.commit()
    return payload


@app.local_entrypoint()
def main():
    payload = train.remote()
    print(f"STATUS {payload['status']} step {payload.get('step')} / {payload.get('steps')}", flush=True)
    out = Path("results/phase5")
    out.mkdir(parents=True, exist_ok=True)
    if payload.get("status") == "done":
        (out / "train.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "losses_every_25"}, indent=2))
