"""Rewrite BoolQ, MultiNLI, and Banking77 into judge rows.

SNLI is converted with the same three labels and then left entirely out of
train and calibration. That is the source Phase 4 scores without having
trained on it.
"""

from __future__ import annotations

import random

from phase1.rows import NLI_OPTIONS, YES_NO, make_row, one_hot

BANKING_OPTIONS = 20
MULTINLI_TRAIN = 23000
SNLI_HELD_OUT = 2000


def boolq_rows(records: list[dict]) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        answer = bool(record["answer"])
        rows.append(
            make_row(
                f"boolq-{index:05d}",
                "boolq",
                "noul",
                record["passage"],
                record["question"],
                YES_NO,
                one_hot(1 if answer else 0, 2),
            )
        )
    return rows


def nli_rows(records: list[dict], source: str, label_names: list[str]) -> list[dict]:
    if list(label_names) != NLI_OPTIONS:
        raise ValueError(f"{source} labels are {label_names}, expected {NLI_OPTIONS}")
    rows = []
    for index, record in enumerate(records):
        label = int(record["label"])
        if label < 0 or label >= len(label_names):
            continue
        premise = str(record["premise"]).strip()
        hypothesis = str(record["hypothesis"]).strip()
        if not premise or not hypothesis:
            continue
        rows.append(
            make_row(
                f"{source}-{index:05d}",
                source,
                "choice",
                f"Premise: {premise}\nHypothesis: {hypothesis}",
                "What is the relationship of the hypothesis to the premise?",
                NLI_OPTIONS,
                one_hot(label, 3),
            )
        )
    return rows


def banking_rows(records: list[dict], label_names: list[str], rng: random.Random) -> list[dict]:
    if len(label_names) < BANKING_OPTIONS:
        raise ValueError(f"need at least {BANKING_OPTIONS} intents, got {len(label_names)}")
    rows = []
    for index, record in enumerate(records):
        text = str(record["text"]).strip()
        if not text:
            continue
        gold = label_names[int(record["label"])]
        others = [name for name in label_names if name != gold]
        distractors = rng.sample(others, BANKING_OPTIONS - 1)
        options = distractors + [gold]
        rng.shuffle(options)
        pretty = [name.replace("_", " ") for name in options]
        gold_pretty = gold.replace("_", " ")
        rows.append(
            make_row(
                f"banking77-{index:05d}",
                "banking77",
                "choice",
                text,
                "Which intent?",
                pretty,
                one_hot(pretty.index(gold_pretty), BANKING_OPTIONS),
            )
        )
    return rows

