"""Cross-entropy on the letter logits. The rest of the vocabulary is ignored."""

from __future__ import annotations

import math


def nll(logits: list[float], target: list[float]) -> float:
    """-sum(target * log softmax(logits)). Used to check the formula without a GPU."""
    if len(logits) != len(target):
        raise ValueError("target and logits differ in length")
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return -sum(weight * math.log(exp / total) for weight, exp in zip(target, exps))


def letter_cross_entropy(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    letter_ids: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    """Mean over the batch of -sum(target * log softmax(letter logits)).

    `lengths` is the index of the last real token in each row.
    """
    import torch

    if logits.ndim != 3:
        raise ValueError("logits must be [batch, time, vocab]")
    losses = []
    for index in range(logits.shape[0]):
        selected = logits[index, int(lengths[index])].index_select(0, letter_ids[index])
        log_probs = torch.log_softmax(selected.float(), dim=-1)
        target = targets[index].to(device=log_probs.device, dtype=log_probs.dtype)
        if target.shape != log_probs.shape:
            raise ValueError("target and letter logits differ in length")
        losses.append(-(target * log_probs).sum())
    return torch.stack(losses).mean()


def text_only_names(names: list[str]) -> list[str]:
    banned = ("vision", "audio", "image", "projector")
    bad = [name for name in names if any(token in name.lower() for token in banned)]
    if bad:
        raise ValueError("LoRA attached outside the text stack: " + ", ".join(bad[:8]))
    return names
