"""Train one LoRA from the frozen Gemma 4 E4B base.

One dataset per call. The adapter is saved under /lora/v2-<dataset>.
It is not merged, and it does not read or write /lora/adapter or
/lora/adapter-pass2. Each call trains for about eight minutes after the
weights load, then saves. The next call resumes that dataset's checkpoint
with a fresh AdamW. The learning rate uses the global step.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

from phase2.loss import letter_cross_entropy, merge_every_25, text_only_names
from phase6.collate import collate
from phase6.meta import (
    BATCH_SIZE,
    CHUNK_TRAIN_SECONDS,
    DATASETS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_ID,
    WARMUP_STEPS,
    prompt_fingerprint,
)
from phase6.modal_common import (
    adapter_dir,
    attach_new_lora,
    device_of,
    disable_cache,
    from_pretrained,
    letter_ids,
    tokenizer_of,
)

APP_NAME = "phase6-lora-train"

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
    .add_local_python_source("phase0", "phase1", "phase2", "phase6")
)
for _dataset in DATASETS:
    image = image.add_local_file(
        f"data/phase6/{_dataset}/train.jsonl",
        remote_path=f"/data/{_dataset}/train.jsonl",
    ).add_local_file(
        f"data/phase6/{_dataset}/manifest.json",
        remote_path=f"/data/{_dataset}/manifest.json",
    )

cache = modal.Volume.from_name("phase2-hf-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("phase6-lora", create_if_missing=True)
app = modal.App(APP_NAME)


def _lr(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (0.1 + 0.9 * cosine)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_progress(directory: Path) -> dict:
    path = directory / "progress.json"
    if not path.exists():
        return {"step": 0, "losses_every_25": []}
    payload = json.loads(path.read_text())
    payload.setdefault("losses_every_25", [])
    payload["step"] = int(payload.get("step", 0))
    return payload


def _checkpoint(
    model,
    directory: Path,
    losses: list[float],
    completed: int,
    steps: int,
    start: int,
    truncated: int,
    torch,
) -> dict:
    from peft import get_peft_model_state_dict

    if len(losses) != completed - start:
        raise RuntimeError(f"loss list {len(losses)} does not match steps {start}..{completed}")
    directory.mkdir(parents=True, exist_ok=True)
    prior = _load_progress(directory)
    payload = {
        "step": completed,
        "steps": steps,
        "last_loss": losses[-1],
        "truncated_rows": int(prior.get("truncated_rows", 0)) + truncated,
        "losses_every_25": merge_every_25(prior["losses_every_25"], start, losses),
    }
    state = {
        name: tensor.detach().to("cpu")
        for name, tensor in get_peft_model_state_dict(model).items()
    }
    torch.save(state, directory / "adapter_state.pt")
    (directory / "progress.json").write_text(json.dumps(payload) + "\n")
    lora_volume.commit()
    print(f"checkpoint {completed}/{steps} -> {directory}", flush=True)
    return payload


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
def train(dataset: str) -> dict:
    from unsloth import FastVisionModel
    import torch
    from peft import set_peft_model_state_dict

    if dataset not in DATASETS:
        raise RuntimeError(f"unknown dataset {dataset}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    directory = adapter_dir(dataset)
    manifest = json.loads(Path(f"/data/{dataset}/manifest.json").read_text())
    if manifest["dataset"] != dataset:
        raise RuntimeError("manifest dataset does not match the call")
    if manifest["padding_side"] != "right":
        raise RuntimeError("manifest is not right-padded")
    if manifest["prompt_sha256"] != prompt_fingerprint()["prompt_sha256"]:
        raise RuntimeError("manifest prompt hash does not match the code")
    if manifest["test_ids_in_train"] or manifest["test_ids_in_calibration"]:
        raise RuntimeError("manifest says the test leaked into train or calibration")
    rows = _read_jsonl(Path(f"/data/{dataset}/train.jsonl"))
    if len(rows) != manifest["train_rows"]:
        raise RuntimeError(f"train file has {len(rows)} rows, manifest says {manifest['train_rows']}")
    if any(row["source"] != dataset for row in rows):
        raise RuntimeError("train file contains another dataset")

    print(f"torch {torch.__version__} cuda {torch.version.cuda} dataset {dataset}", flush=True)
    torch.manual_seed(0)
    model, processor = from_pretrained(FastVisionModel, torch, MODEL_ID, checkpointing=True)
    model = attach_new_lora(FastVisionModel, model)
    if hasattr(FastVisionModel, "for_training"):
        FastVisionModel.for_training(model)
    disable_cache(model)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    text_only_names(trainable)
    print(f"trainable tensors {len(trainable)} sample {trainable[:4]}", flush=True)
    device = device_of(model, torch)
    tokenizer = tokenizer_of(processor)
    ids = letter_ids(tokenizer)

    steps = math.ceil(len(rows) / BATCH_SIZE)
    progress = _load_progress(directory)
    start = progress["step"]
    if start > steps:
        raise RuntimeError(f"checkpoint step {start} is past {steps}")
    state_path = directory / "adapter_state.pt"
    if start:
        if not state_path.exists():
            raise RuntimeError(f"progress says step {start} but adapter_state.pt is missing")
        set_peft_model_state_dict(model, torch.load(state_path, map_location="cpu", weights_only=True))
        print(f"resumed at step {start}/{steps}", flush=True)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
    )
    losses = []
    model.train()
    deadline = time.time() + CHUNK_TRAIN_SECONDS
    completed = start
    truncated = 0
    for step in range(start, steps):
        batch_rows = rows[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        lr = _lr(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = collate(tokenizer, batch_rows, ids, device, torch, MAX_LENGTH)
        truncated += batch["truncated"]
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        loss = letter_cross_entropy(logits, batch["indexes"], batch["selected"], batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (param for param in model.parameters() if param.requires_grad),
            1.0,
        )
        optimizer.step()
        value = float(loss.detach())
        losses.append(round(value, 6))
        completed = step + 1
        if step % 25 == 0 or completed == steps:
            print(f"step {completed}/{steps} loss {value:.4f} lr {lr:.6f}", flush=True)
        if completed == steps:
            break
        if time.time() > deadline:
            saved = _checkpoint(model, directory, losses, completed, steps, start, truncated, torch)
            return {
                "status": "partial",
                "dataset": dataset,
                "step": completed,
                "steps": steps,
                "rows": len(rows),
                "last_loss": losses[-1],
                "losses_every_25": saved["losses_every_25"],
                "adapter_dir": directory.as_posix(),
                "merged": False,
            }

    if losses:
        saved = _checkpoint(model, directory, losses, completed, steps, start, truncated, torch)
    else:
        saved = progress
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    adapter = _adapter_report(directory)
    lora_volume.commit()
    curve = saved["losses_every_25"]
    early = sum(curve[:4]) / max(1, min(4, len(curve)))
    late = sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else float(saved.get("last_loss", 0.0))
    return {
        "status": "done",
        "dataset": dataset,
        "model": MODEL_ID,
        "rows": len(rows),
        "steps": steps,
        "resumed_from": start,
        "batch_size": BATCH_SIZE,
        "rank": 16,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "load_in_4bit": False,
        "merged": False,
        "adapter_dir": directory.as_posix(),
        "adapter_bytes": adapter["adapter_bytes"],
        "adapter_files": adapter["adapter_files"],
        "trainable_tensors": len(trainable),
        "trainable_sample": trainable[:12],
        "truncated_rows": saved.get("truncated_rows", truncated),
        "early_every_25_mean": early,
        "epoch_last_100_mean": late,
        "epoch_loss_fell": late < early,
        "losses_every_25": curve,
        "chunk_note": "Each chunk starts a fresh AdamW. The learning rate follows the global step.",
    }


@app.local_entrypoint()
def main(dataset: str):
    if dataset not in DATASETS:
        raise SystemExit(f"dataset must be one of {DATASETS}")
    out = Path("results/phase6") / dataset
    out.mkdir(parents=True, exist_ok=True)
    while True:
        payload = train.remote(dataset)
        status = {key: payload[key] for key in payload if key != "losses_every_25"}
        print(json.dumps(status, indent=2), flush=True)
        (out / "train_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if payload.get("status") == "done":
            (out / "train.json").write_text(json.dumps(payload, indent=2) + "\n")
            return
