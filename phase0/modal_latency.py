"""Time one prefill against a real decode on the Phase 0 model.

Same checkpoint, same exam prompts, batch size 1. No training.
The container is single-use: Modal tears it down when this call returns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from phase0.items import SYSTEM_PROMPT, iter_questions, render_user_prompt
from phase0.modal_baseline import EXAM, EXAM_CONFIG, EXAM_SPLIT, MODEL_ID, image

APP_NAME = "phase0-latency-check"
cache = modal.Volume.from_name("phase0-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)

REPEATS = 3
FORCED_NEW_TOKENS = 32


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


@app.function(
    image=image,
    gpu="L40S",
    timeout=30 * 60,
    volumes={"/cache": cache},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def latency_check() -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cache_dir = "/cache/huggingface"
    dataset = load_dataset(EXAM, EXAM_CONFIG, split=EXAM_SPLIT, cache_dir=cache_dir)
    items = []
    for row in dataset:
        items.extend(iter_questions(row))

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

    prompts = []
    for item in items:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(item)},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= 4096:
            prompts.append((len(token_ids), text, item["type"], item["workflow"]))
    prompts.sort(key=lambda row: row[0])
    picks = {
        "short": prompts[0],
        "mid": prompts[len(prompts) // 2],
        "long": prompts[-1],
    }

    def encode(text: str):
        return tokenizer(text, return_tensors="pt").to(model.device)

    def elapsed_ms(fn):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000.0, value

    def forward_once(encoded):
        with torch.inference_mode():
            model(**encoded)

    def decode_once(encoded, max_new_tokens: int, force: bool):
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if force:
            kwargs["min_new_tokens"] = max_new_tokens
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        new_count = int(output.shape[-1] - encoded["input_ids"].shape[-1])
        snippet = tokenizer.decode(output[0, -new_count:], skip_special_tokens=True)
        return new_count, snippet[:160]

    warmup_text = picks["short"][1]
    warmup = encode(warmup_text)
    forward_once(warmup)
    decode_once(warmup, 8, force=False)

    cases = []
    for name, (token_count, text, question_type, workflow) in picks.items():
        encoded = encode(text)
        no_decode = []
        natural = []
        natural_tokens = []
        snippet = ""
        forced = []
        forced_tokens = []
        for _ in range(REPEATS):
            duration, _ = elapsed_ms(lambda: forward_once(encoded))
            no_decode.append(duration)
            duration, decoded = elapsed_ms(lambda: decode_once(encoded, 64, force=False))
            natural.append(duration)
            natural_tokens.append(decoded[0])
            snippet = decoded[1]
            duration, decoded = elapsed_ms(lambda: decode_once(encoded, FORCED_NEW_TOKENS, force=True))
            forced.append(duration)
            forced_tokens.append(decoded[0])
        no_decode_ms = _median(no_decode)
        natural_ms = _median(natural)
        forced_ms = _median(forced)
        cases.append(
            {
                "name": name,
                "type": question_type,
                "workflow": workflow,
                "input_tokens": token_count,
                "no_decode_ms": round(no_decode_ms, 1),
                "with_decode_ms": round(natural_ms, 1),
                "decoded_tokens": int(_median([float(value) for value in natural_tokens])),
                "decode_text": snippet,
                "forced_32_ms": round(forced_ms, 1),
                "forced_32_tokens": int(_median([float(value) for value in forced_tokens])),
                "repeats_no_decode_ms": [round(value, 1) for value in no_decode],
                "repeats_with_decode_ms": [round(value, 1) for value in natural],
                "repeats_forced_32_ms": [round(value, 1) for value in forced],
            }
        )
        print(
            f"{name} tokens={token_count} no_decode={no_decode_ms:.1f}ms "
            f"with_decode={natural_ms:.1f}ms forced_32={forced_ms:.1f}ms",
            flush=True,
        )

    return {
        "model": MODEL_ID,
        "gpu": "L40S",
        "exam": EXAM,
        "exam_split": EXAM_SPLIT,
        "batch_size": 1,
        "repeats": REPEATS,
        "thinking": False,
        "container": "single_use_containers, scaledown_window 2 seconds, min_containers 0",
        "no_decode": "One forward pass. The letter logits are the answer. No new tokens.",
        "with_decode": "model.generate on the same prompt, max 64 new tokens, stops early on EOS.",
        "forced_32": "Same prompt, exactly 32 new tokens, so a short paragraph has a clock.",
        "cases": cases,
    }


@app.local_entrypoint()
def main():
    payload = latency_check.remote()
    out_dir = Path("results/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latency.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
