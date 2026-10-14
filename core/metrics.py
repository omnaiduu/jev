"""Phase 0 scores. No model code lives here, so the formulas can be checked on CPU."""

from __future__ import annotations

import math


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("softmax needs at least one logit")
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def accuracy(correct: list[bool]) -> float:
    if not correct:
        raise ValueError("accuracy needs at least one question")
    return sum(correct) / len(correct)


def brier_score(probabilities: list[list[float]], label_indexes: list[int]) -> float:
    """Mean multiclass Brier against a one-hot label.

    For one question, sum over options of (p - y)^2. y is 1 on the labeled
    option and 0 elsewhere. Averaged over questions.
    """
    if len(probabilities) != len(label_indexes) or not probabilities:
        raise ValueError("probabilities and label indexes must be the same non-empty length")
    total = 0.0
    for probs, label_index in zip(probabilities, label_indexes):
        if label_index < 0 or label_index >= len(probs):
            raise ValueError("label index outside the option list")
        total += sum((p - (1.0 if i == label_index else 0.0)) ** 2 for i, p in enumerate(probs))
    return total / len(probabilities)


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """ECE of the top-option percentage against whether that option was right.

    Questions are placed in equal-width bins on [0, 1]. Each bin contributes
    |mean confidence - accuracy| weighted by how many questions fell in it.
    """
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct flags must be the same non-empty length")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    n = len(confidences)
    total = 0.0
    for bin_index in range(n_bins):
        lo = bin_index / n_bins
        hi = (bin_index + 1) / n_bins
        members = [
            (confidence, hit)
            for confidence, hit in zip(confidences, correct)
            if (lo <= confidence < hi) or (bin_index == n_bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(item[0] for item in members) / len(members)
        bin_accuracy = sum(1.0 if item[1] else 0.0 for item in members) / len(members)
        total += (len(members) / n) * abs(mean_confidence - bin_accuracy)
    return total
