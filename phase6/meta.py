"""Frozen constants for the per-dataset runs.

The prompt hash is the Phase 2 system line plus one rendered user message.
Train and test both call phase2.prompts.render_prompt, so this hash is the
check that those calls still build `A. {option text}`.
"""

from __future__ import annotations

import hashlib

from phase2.prompts import SYSTEM_PROMPT, render_user_prompt

DATASETS = ("boolq", "multinli", "banking77", "typed-decisions")
CALIBRATION_N = 2000
SHUFFLE_FRACTION = 0.30
SEED = 0
MODEL_ID = "unsloth/gemma-4-E4B-it"
MAX_LENGTH = 2048
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
RANK = 16
CHUNK_TRAIN_SECONDS = 8 * 60
TEACHER_CEILING_CORRECT = 1470
TEACHER_CEILING_N = 2000

# Retired adapter directories. Phase 6 must not write either path.
RETIRED_ADAPTER_DIRS = ("/lora/adapter", "/lora/adapter-pass2")

LABEL_SOURCE = {
    "boolq": "human yes/no",
    "multinli": "human entailment, neutral, or contradiction",
    "banking77": "one of 77 intents, shown as the true intent plus 19 distractors",
    "typed-decisions": (
        "teacher letter, three samples at temperature 0.7; "
        "the loss target is the stored probability vector; "
        "the correct count is agreement with the stored label"
    ),
    "snli": "human entailment, neutral, or contradiction; transfer check only",
}


def prompt_fingerprint() -> dict:
    rendered = render_user_prompt(
        {
            "id": "canon",
            "state": "STATE",
            "question": "QUESTION",
            "options": ["alpha", "beta"],
        }
    )
    blob = SYSTEM_PROMPT + "\n" + rendered
    return {
        "prompt_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "prompt_template": "A. {option text}",
        "padding_side": "right",
        "system_prompt": SYSTEM_PROMPT,
        "canonical_user": rendered,
    }
