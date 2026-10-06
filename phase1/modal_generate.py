"""Ask a larger Gemma to write the emails and notes. The script keeps the label.

The container is single-use and scales down two seconds after this call.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from phase1.synthetic import (
    accept_passage,
    accept_refund,
    passage_prompt,
    passage_specs,
    refund_prompt,
    refund_specs,
    row_from_passage,
    row_from_refund,
)

APP_NAME = "phase1-synthetic-writer"
WRITER = "google/gemma-4-12B-it"
REFUND_TARGET = 1600
PASSAGE_TARGET = 800

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.18.0", "accelerate", "safetensors")
    .add_local_python_source("phase1")
)
cache = modal.Volume.from_name("phase1-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def _chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = encoded["input_ids"].shape[-1]
    texts = []
    for sequence in output:
        texts.append(tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip())
    return texts


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/cache": cache},
    env={"HF_HOME": "/cache/huggingface"},
    min_containers=0,
    scaledown_window=2,
    single_use_containers=True,
)
def write_rows(refund_target: int = REFUND_TARGET, passage_target: int = PASSAGE_TARGET) -> list[dict]:
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    cache_dir = "/cache/huggingface"
    tokenizer = AutoTokenizer.from_pretrained(WRITER, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            WRITER,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            WRITER,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    model.eval()

    def collect(specs: list[dict], prompt_fn, accept_fn, row_fn, target: int, max_new_tokens: int) -> list[dict]:
        accepted = []
        batch_size = 4
        cursor = 0
        while len(accepted) < target and cursor < len(specs):
            batch = specs[cursor : cursor + batch_size]
            cursor += batch_size
            texts = _generate_batch(
                model,
                tokenizer,
                [_chat(tokenizer, prompt_fn(spec)) for spec in batch],
                max_new_tokens,
            )
            for spec, text in zip(batch, texts):
                if accept_fn(text, spec):
                    accepted.append(row_fn(spec, text))
            print(f"accepted {len(accepted)}/{target}", flush=True)
        if len(accepted) < target:
            raise RuntimeError(f"accepted {len(accepted)} of {target}")
        return accepted[:target]

    rng = random.Random(0)
    refund = collect(
        refund_specs(refund_target * 3),
        refund_prompt,
        accept_refund,
        row_from_refund,
        refund_target,
        180,
    )
    passages = collect(
        passage_specs(passage_target * 3, rng),
        passage_prompt,
        accept_passage,
        row_from_passage,
        passage_target,
        180,
    )
    cache.commit()
    return refund + passages


@app.local_entrypoint()
def main():
    rows = write_rows.remote()
    path = Path("data/phase1/pool/synthetic.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} {path}")
