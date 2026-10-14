"""Score one dataset at the last real token, with right padding.

`weights=plain` loads the base model. `weights=lora` loads /lora/v2-<dataset>.
The same collate is used for both, and for the reversed-option pass.
MultiNLI also scores SNLI. SNLI does not enter the keep bit. Its LoRA
temperature is the MultiNLI temperature.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase2.loss import text_only_names
from phase6.collate import collate
from phase6.meta import BATCH_SIZE, DATASETS, MAX_LENGTH, MODEL_ID, prompt_fingerprint
from phase6.modal_common import (
    adapter_dir,
    attach_new_lora,
    device_of,
    disable_cache,
    from_pretrained,
    letter_ids,
    tokenizer_of,
)
from phase6.report import document_at_temperature, result_document
from phase6.rows import reverse_options

APP_NAME = "phase6-score"

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
    .add_local_python_source("phase0", "phase1", "phase2", "phase3", "phase4", "phase6")
)
for _dataset in DATASETS:
    image = (
        image.add_local_file(
            f"data/phase6/{_dataset}/test.jsonl",
            remote_path=f"/data/{_dataset}/test.jsonl",
        )
        .add_local_file(
            f"data/phase6/{_dataset}/calibration.jsonl",
            remote_path=f"/data/{_dataset}/calibration.jsonl",
        )
        .add_local_file(
            f"data/phase6/{_dataset}/manifest.json",
            remote_path=f"/data/{_dataset}/manifest.json",
        )
    )
image = image.add_local_file("data/phase6/snli/test.jsonl", remote_path="/data/snli/test.jsonl")

cache = modal.Volume.from_name("phase2-hf-cache", create_if_missing=True)
lora_volume = modal.Volume.from_name("phase6-lora", create_if_missing=True)
app = modal.App(APP_NAME)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _forward(model, batch):
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return output.logits if hasattr(output, "logits") else output[0]


def _letter_logits(logits, index: int, batch) -> list[float]:
    position = int(batch["indexes"][index])
    selected = batch["selected"][index]
    return logits[index, position].index_select(0, selected).float().tolist()


def _load(torch, dataset: str, use_lora: bool):
    from unsloth import FastVisionModel

    model, processor = from_pretrained(FastVisionModel, torch, MODEL_ID, checkpointing=False)
    if not use_lora:
        if any("lora_" in name for name, _param in model.named_parameters()):
            raise RuntimeError("plain load contains LoRA tensors")
        return model, processor, None
    directory = adapter_dir(dataset)
    state_path = directory / "adapter_state.pt"
    if not state_path.exists():
        raise RuntimeError(f"missing LoRA checkpoint {state_path}")
    model = attach_new_lora(FastVisionModel, model)
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, torch.load(state_path, map_location="cpu", weights_only=True))
    names = [name for name, _param in model.named_parameters() if "lora_" in name]
    if not names:
        raise RuntimeError("LoRA tensors missing after loading adapter_state.pt")
    text_only_names(names)
    return model, processor, directory.as_posix()


def _score_split(model, tokenizer, ids, rows: list[dict], device, torch, with_flip: bool) -> dict:
    import torch as torch_mod

    packed = []
    truncated = 0
    asserts = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        batch = collate(tokenizer, batch_rows, ids, device, torch_mod, MAX_LENGTH)
        truncated += batch["truncated"]
        asserts += len(batch_rows)
        with torch_mod.inference_mode():
            logits = _forward(model, batch)
        flip_logits = None
        flipped = None
        if with_flip:
            flipped = [reverse_options(row) for row in batch_rows]
            flip_batch = collate(tokenizer, flipped, ids, device, torch_mod, MAX_LENGTH)
            truncated += flip_batch["truncated"]
            asserts += len(batch_rows)
            with torch_mod.inference_mode():
                flip_logits = _forward(model, flip_batch)
        for index, row in enumerate(batch_rows):
            item = {
                "id": row["id"],
                "type": row["type"],
                "case_id": row.get("case_id") or "",
                "label_index": row["label_index"],
                "label_id": row["label_id"],
                "option_ids": row["option_ids"],
                "logits": _letter_logits(logits, index, batch),
            }
            if with_flip:
                flip_values = _letter_logits(flip_logits, index, flip_batch)
                pick = max(range(len(flip_values)), key=lambda slot: flip_values[slot])
                item["flip_prediction"] = flipped[index]["option_ids"][pick]
            packed.append(item)
        done = min(start + BATCH_SIZE, len(rows))
        if start == 0 or done == len(rows) or done % 400 == 0:
            print(f"scored {done}/{len(rows)}", flush=True)
    return {"rows": packed, "truncated": truncated, "padding_asserts": asserts}


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
def score(dataset: str, weights: str) -> dict:
    import torch

    if dataset not in DATASETS:
        raise RuntimeError(f"unknown dataset {dataset}")
    if weights not in ("plain", "lora"):
        raise RuntimeError("weights must be plain or lora")
    use_lora = weights == "lora"
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in torch {torch.__version__}")
    try:
        lora_volume.reload()
    except Exception:
        pass
    manifest = json.loads(Path(f"/data/{dataset}/manifest.json").read_text())
    if manifest["prompt_sha256"] != prompt_fingerprint()["prompt_sha256"]:
        raise RuntimeError("manifest prompt hash does not match the code")
    if manifest["padding_side"] != "right":
        raise RuntimeError("manifest is not right-padded")
    test_rows = _read_jsonl(Path(f"/data/{dataset}/test.jsonl"))
    if len(test_rows) != manifest["test_rows"]:
        raise RuntimeError(f"test file has {len(test_rows)} rows, manifest says {manifest['test_rows']}")
    if any(row["shuffled"] for row in test_rows):
        raise RuntimeError("test rows were shuffled")

    torch.manual_seed(0)
    model, processor, loaded_from = _load(torch, dataset, use_lora)
    from unsloth import FastVisionModel

    if use_lora and hasattr(FastVisionModel, "for_inference"):
        FastVisionModel.for_inference(model)
    elif hasattr(FastVisionModel, "for_inference"):
        FastVisionModel.for_inference(model)
    disable_cache(model)
    model.eval()
    device = device_of(model, torch)
    tokenizer = tokenizer_of(processor)
    ids = letter_ids(tokenizer)
    print(f"scoring {dataset} weights={weights} padding_side={tokenizer.padding_side}", flush=True)

    test = _score_split(model, tokenizer, ids, test_rows, device, torch, with_flip=True)
    calibration = None
    if use_lora:
        calibration_rows = _read_jsonl(Path(f"/data/{dataset}/calibration.jsonl"))
        if len(calibration_rows) != manifest["calibration_rows"]:
            raise RuntimeError("calibration file does not match the manifest")
        if any(row["shuffled"] for row in calibration_rows):
            raise RuntimeError("calibration rows were shuffled")
        calibration = _score_split(model, tokenizer, ids, calibration_rows, device, torch, with_flip=False)
    snli = None
    if dataset == "multinli":
        snli_rows = _read_jsonl(Path("/data/snli/test.jsonl"))
        if any(row["source"] != "snli" for row in snli_rows):
            raise RuntimeError("SNLI file contains another source")
        snli = _score_split(model, tokenizer, ids, snli_rows, device, torch, with_flip=True)
    cache.commit()
    return {
        "dataset": dataset,
        "model": MODEL_ID,
        "lora": use_lora,
        "adapter_dir": loaded_from,
        "prompt_sha256": prompt_fingerprint()["prompt_sha256"],
        "test": test,
        "calibration": calibration,
        "snli": snli,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _meta(payload: dict, part: dict) -> dict:
    return {
        "model": payload["model"],
        "lora": payload["lora"],
        "adapter_dir": payload["adapter_dir"],
        "padding_asserts": part["padding_asserts"],
        "truncated": part["truncated"],
        "prompt_sha256": payload["prompt_sha256"],
    }


@app.local_entrypoint()
def main(dataset: str, weights: str = "plain"):
    if dataset not in DATASETS:
        raise SystemExit(f"dataset must be one of {DATASETS}")
    if weights not in ("plain", "lora"):
        raise SystemExit("weights must be plain or lora")
    payload = score.remote(dataset, weights)
    out = Path("results/phase6") / dataset
    out.mkdir(parents=True, exist_ok=True)
    prefix = "lora" if weights == "lora" else "plain"
    _write_jsonl(out / f"{prefix}_test.jsonl", payload["test"]["rows"])
    calibration_rows = None
    if payload["calibration"] is not None:
        calibration_rows = payload["calibration"]["rows"]
        _write_jsonl(out / "lora_calibration.jsonl", calibration_rows)
    document = result_document(dataset, payload["test"]["rows"], _meta(payload, payload["test"]), calibration_rows)
    (out / f"{prefix}.json").write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dataset": dataset,
                "weights": weights,
                "correct": document["test_at_1"]["correct"],
                "n": document["test_at_1"]["n"],
                "ece_at_1": document["test_at_1"]["ece"],
                "ece_at_T": document["test_at_T"]["ece"],
                "T": document["temperature"]["T"],
            },
            indent=2,
        ),
        flush=True,
    )
    if payload["snli"] is not None:
        snli_out = Path("results/phase6/snli")
        snli_out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(snli_out / f"{prefix}_test.jsonl", payload["snli"]["rows"])
        if weights == "lora":
            snli_document = document_at_temperature(
                "snli",
                payload["snli"]["rows"],
                _meta(payload, payload["snli"]),
                float(document["temperature"]["T"]),
                "multinli calibration",
            )
        else:
            snli_document = result_document("snli", payload["snli"]["rows"], _meta(payload, payload["snli"]), None)
        (snli_out / f"{prefix}.json").write_text(json.dumps(snli_document, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "dataset": "snli",
                    "weights": weights,
                    "enters_keep": False,
                    "correct": snli_document["test_at_1"]["correct"],
                    "n": snli_document["test_at_1"]["n"],
                },
                indent=2,
            ),
            flush=True,
        )
