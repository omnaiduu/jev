"""Fit one temperature per question type. No model code, so the search runs on CPU.

Temperature divides the logits before softmax. The winning option stays put.
The search only moves how sure the percentages look.
"""

from __future__ import annotations

import math

from phase0.metrics import accuracy, brier_score, expected_calibration_error, softmax

# The plan's examples, plus the steps between them. T = 1 is the unscaled model.
GRID = [round(0.5 + index * 0.05, 2) for index in range(51)]
CALIBRATION_TYPES = ("noul", "choice")
FORBIDDEN_SOURCES = ("snli", "typed-decisions")


def nll_at_temperature(logits: list[float], label_index: int, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if label_index < 0 or label_index >= len(logits):
        raise ValueError("label index outside the option list")
    probabilities = softmax([value / temperature for value in logits])
    return -math.log(max(probabilities[label_index], 1e-12))


def winning_index(logits: list[float]) -> int:
    return max(range(len(logits)), key=lambda index: logits[index])


def summarize(records: list[dict], temperature: float) -> dict:
    if not records:
        raise ValueError("temperature summary needs at least one row")
    correct = []
    confidences = []
    probabilities = []
    labels = []
    nlls = []
    for record in records:
        scaled = [value / temperature for value in record["logits"]]
        probs = softmax(scaled)
        pick = winning_index(scaled)
        correct.append(pick == record["label_index"])
        confidences.append(probs[pick])
        probabilities.append(probs)
        labels.append(record["label_index"])
        nlls.append(nll_at_temperature(record["logits"], record["label_index"], temperature))
    return {
        "n": len(records),
        "temperature": temperature,
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier_score(probabilities, labels),
        "nll": sum(nlls) / len(nlls),
    }


def fit_temperature(records: list[dict], grid: list[float] | None = None) -> dict:
    """Pick the grid temperature with the lowest mean letter cross-entropy."""
    if not records:
        raise ValueError("cannot fit temperature on an empty group")
    choices = list(GRID if grid is None else grid)
    if not choices:
        raise ValueError("temperature grid is empty")
    table = []
    best = None
    for temperature in choices:
        score = summarize(records, temperature)
        entry = {"T": temperature, "nll": score["nll"]}
        table.append(entry)
        if best is None or entry["nll"] < best["nll"]:
            best = entry
    chosen = summarize(records, best["T"])
    baseline = summarize(records, 1.0)
    if chosen["accuracy"] != baseline["accuracy"]:
        raise RuntimeError("temperature changed the winning option")
    return {
        "T": best["T"],
        "nll": chosen["nll"],
        "nll_at_1": baseline["nll"],
        "accuracy": chosen["accuracy"],
        "ece_at_1": baseline["ece"],
        "ece_at_T": chosen["ece"],
        "brier_at_1": baseline["brier"],
        "brier_at_T": chosen["brier"],
        "n": chosen["n"],
        "grid": table,
    }


def assert_calibration_row(row: dict) -> None:
    if row.get("shuffled"):
        raise ValueError(f"{row.get('id')} was shuffled; calibration must stay in file order")
    source = row.get("source", "")
    if source in FORBIDDEN_SOURCES or "typed-decisions" in source:
        raise ValueError(f"{row.get('id')} source {source} is not calibration data")
    if row.get("type") not in CALIBRATION_TYPES and row.get("type") != "score":
        raise ValueError(f"{row.get('id')} has unexpected type {row.get('type')}")
    target = row["target"]
    if len(target) != len(row["options"]) or abs(sum(target) - 1.0) > 1e-6:
        raise ValueError(f"{row.get('id')} target does not match its options")
