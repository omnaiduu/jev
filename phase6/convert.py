"""Rewrite one downloaded split into judge rows.

Banking77 has no natural option order. Each row stores the true intent plus
19 other intents, drawn once by the caller. typed-decisions stores a teacher
probability vector and a single label. The vector is the loss target. The
label is what the correct count agrees with.
"""

from __future__ import annotations

import json
import random
from typing import Any

from phase0.items import option_pairs
from phase1.rows import NLI_OPTIONS, YES_NO
from phase6.rows import make_row, one_hot

BANKING_OPTIONS = 20


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def boolq_rows(records: list[dict], split: str) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        passage = str(record["passage"]).strip()
        question = str(record["question"]).strip()
        if not passage or not question:
            continue
        label_id = "yes" if bool(record["answer"]) else "no"
        row_id = f"boolq-{split}-{index:06d}"
        rows.append(
            make_row(
                row_id,
                "boolq",
                "noul",
                passage,
                question,
                YES_NO,
                list(YES_NO),
                one_hot(YES_NO.index(label_id), 2),
                label_id,
                "one_hot",
                row_id,
            )
        )
    return rows


def nli_rows(records: list[dict], source: str, split: str, label_names: list[str]) -> tuple[list[dict], int]:
    if list(label_names) != NLI_OPTIONS:
        raise ValueError(f"{source} labels are {label_names}, expected {NLI_OPTIONS}")
    rows = []
    dropped = 0
    for index, record in enumerate(records):
        label = int(record["label"])
        premise = str(record["premise"]).strip()
        hypothesis = str(record["hypothesis"]).strip()
        if label < 0 or label >= len(label_names) or not premise or not hypothesis:
            dropped += 1
            continue
        label_id = label_names[label]
        row_id = f"{source}-{split}-{index:06d}"
        rows.append(
            make_row(
                row_id,
                source,
                "choice",
                f"Premise: {premise}\nHypothesis: {hypothesis}",
                "What is the relationship of the hypothesis to the premise?",
                NLI_OPTIONS,
                list(NLI_OPTIONS),
                one_hot(label, 3),
                label_id,
                "one_hot",
                row_id,
            )
        )
    return rows, dropped


def banking_rows(records: list[dict], label_names: list[str], rng: random.Random, split: str) -> list[dict]:
    if len(label_names) < BANKING_OPTIONS:
        raise ValueError(f"need at least {BANKING_OPTIONS} intents, got {len(label_names)}")
    if len(set(label_names)) != len(label_names):
        raise ValueError("banking intent names are not unique")
    pretty_names = [name.replace("_", " ") for name in label_names]
    if len(set(pretty_names)) != len(pretty_names):
        raise ValueError("banking intent names collide after replacing underscores")
    rows = []
    for index, record in enumerate(records):
        text = str(record["text"]).strip()
        if not text:
            continue
        gold = label_names[int(record["label"])]
        others = [name for name in label_names if name != gold]
        distractors = rng.sample(others, BANKING_OPTIONS - 1)
        chosen = distractors + [gold]
        rng.shuffle(chosen)
        options = [name.replace("_", " ") for name in chosen]
        row_id = f"banking77-{split}-{index:06d}"
        rows.append(
            make_row(
                row_id,
                "banking77",
                "choice",
                text,
                "Which intent?",
                options,
                chosen,
                one_hot(chosen.index(gold), BANKING_OPTIONS),
                gold,
                "one_hot",
                row_id,
            )
        )
    return rows


def _soft_target(pairs: list[tuple[str, str]], probabilities: dict, label: str, stats: dict) -> tuple[list[str], list[str], list[float], str]:
    option_ids = []
    options = []
    raw = []
    for option_id, text in pairs:
        if option_id not in probabilities:
            raise ValueError(f"label {label!r} probabilities have no weight for {option_id!r}")
        shown = str(text).strip() if text is not None and str(text).strip() else option_id
        option_ids.append(option_id)
        options.append(shown)
        raw.append(float(probabilities[option_id]))
    if label not in option_ids:
        raise ValueError(f"stored label {label!r} is not in {option_ids}")
    total = sum(raw)
    if total <= 0 or abs(total - 1.0) > 1e-3:
        raise ValueError(f"teacher probabilities sum to {total}")
    if abs(total - 1.0) > 1e-8:
        stats["renormalized"] += 1
    target = [value / total for value in raw]
    peak = max(target)
    label_index = option_ids.index(label)
    if target[label_index] + 1e-8 < peak:
        stats["label_not_mode"] += 1
    if sum(1 for value in target if abs(value - peak) <= 1e-8) > 1:
        stats["ties"] += 1
    return options, option_ids, target, label


def typed_rows(cases: list[dict], split: str) -> tuple[list[dict], dict]:
    stats = {
        "cases": len(cases),
        "questions": 0,
        "renormalized": 0,
        "label_not_mode": 0,
        "ties": 0,
    }
    rows = []
    for case in cases:
        questions = _parse(case["questions"])
        gold = _parse(case["gold"])
        case_id = str(case["id"])
        state = str(case["state"])
        for name, question in questions.items():
            gold_item = gold[name]
            pairs = option_pairs(question)
            options, option_ids, target, label = _soft_target(
                pairs,
                gold_item["probabilities"],
                str(gold_item["label"]),
                stats,
            )
            row_id = f"typed-decisions-{split}-{case_id}-{name}"
            rows.append(
                make_row(
                    row_id,
                    "typed-decisions",
                    str(question["type"]),
                    state,
                    str(question["instructions"]),
                    options,
                    option_ids,
                    target,
                    label,
                    "soft",
                    case_id,
                    case_id=case_id,
                    question_name=str(name),
                    workflow=str(case.get("workflow") or ""),
                )
            )
            stats["questions"] += 1
    return rows, stats
