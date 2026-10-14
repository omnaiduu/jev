"""Phase 4 comparisons. No model code, so the checks run on CPU.

The exam score uses the temperatures fit in Phase 3. Score has no calibration
rows, so its temperature stays 1. Flipping an item reverses the option list
and rebinds the letters. The comparison is by option id.
"""

from __future__ import annotations

from phase0.items import LETTERS
from phase0.metrics import accuracy, brier_score, expected_calibration_error, softmax

# Recorded Phase 3 fit. score is absent from that file.
PHASE3_TEMPERATURES = {"noul": 1.65, "choice": 1.3}
SCORE_TEMPERATURE = 1.0


def temperatures_from_phase3(payload: dict) -> dict[str, float]:
    if payload.get("fit_on") != "data/phase1/calibration.jsonl":
        raise ValueError("temperature file was not fit on the calibration pile")
    excluded = set(payload.get("not_fit_on") or [])
    for name in ("train", "typed-decisions", "snli"):
        if name not in excluded:
            raise ValueError(f"temperature file does not exclude {name}")
    if payload.get("missing_types") != ["score"]:
        raise ValueError("score must stay at temperature 1")
    found = {
        "noul": payload["types"]["noul"]["T"],
        "choice": payload["types"]["choice"]["T"],
        "score": SCORE_TEMPERATURE,
    }
    if found["noul"] != PHASE3_TEMPERATURES["noul"] or found["choice"] != PHASE3_TEMPERATURES["choice"]:
        raise ValueError("temperature file does not match the recorded Phase 3 fit")
    return found


def phase0_index(mask: list[int]) -> int:
    """Index Phase 0 used: count of content tokens, minus one."""
    content = sum(mask)
    if content <= 0:
        raise ValueError("mask has no content token")
    return content - 1


def last_content_index(mask: list[int]) -> int:
    """Last index whose attention mask is 1. That is the last real token."""
    ones = [index for index, bit in enumerate(mask) if bit]
    if not ones:
        raise ValueError("mask has no content token")
    return ones[-1]


def temperature_for(question_type: str, temperatures: dict[str, float]) -> float:
    if question_type not in temperatures:
        raise ValueError(f"no temperature for question type {question_type}")
    value = temperatures[question_type]
    if value <= 0:
        raise ValueError("temperature must be positive")
    return value


def winning_index(logits: list[float]) -> int:
    if not logits:
        raise ValueError("winning index needs at least one logit")
    return max(range(len(logits)), key=lambda index: logits[index])


def readout(
    logits: list[float],
    label_index: int,
    option_ids: list[str],
    temperature: float,
) -> dict:
    """Softmax(logits / T). The winning option is the argmax of the logits."""
    if len(logits) != len(option_ids):
        raise ValueError("logits and option ids differ in length")
    if label_index < 0 or label_index >= len(option_ids):
        raise ValueError("label index outside the option list")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = softmax([value / temperature for value in logits])
    pick = winning_index(logits)
    if pick != winning_index([value / temperature for value in logits]):
        raise RuntimeError("temperature changed the winning option")
    return {
        "label_index": label_index,
        "prediction": option_ids[pick],
        "correct": pick == label_index,
        "confidence": probabilities[pick],
        "probabilities": probabilities,
        "option_ids": list(option_ids),
        "temperature": temperature,
    }


def summarize_records(records: list[dict], group_keys: tuple[str, ...] = ()) -> dict:
    if not records:
        raise ValueError("summary needs at least one row")
    summary = {
        "n_questions": len(records),
        "n_cases": len({record["case_id"] for record in records}),
        "accuracy": accuracy([record["correct"] for record in records]),
        "ece": expected_calibration_error(
            [record["confidence"] for record in records],
            [record["correct"] for record in records],
        ),
        "brier": brier_score(
            [record["probabilities"] for record in records],
            [record["label_index"] for record in records],
        ),
    }
    for key in group_keys:
        summary[key] = {}
        for value in sorted({record[key] for record in records}):
            group = [record for record in records if record[key] == value]
            summary[key][value] = {
                "n": len(group),
                "accuracy": accuracy([record["correct"] for record in group]),
                "ece": expected_calibration_error(
                    [record["confidence"] for record in group],
                    [record["correct"] for record in group],
                ),
                "brier": brier_score(
                    [record["probabilities"] for record in group],
                    [record["label_index"] for record in group],
                ),
            }
    return summary


def slot_histogram(records: list[dict]) -> dict:
    picked: dict[int, int] = {}
    gold: dict[int, int] = {}
    for record in records:
        slot = record["option_ids"].index(record["prediction"])
        picked[slot] = picked.get(slot, 0) + 1
        gold_slot = record["label_index"]
        gold[gold_slot] = gold.get(gold_slot, 0) + 1
    return {
        "picked": {str(slot): count for slot, count in sorted(picked.items())},
        "gold": {str(slot): count for slot, count in sorted(gold.items())},
    }


def flip_item(item: dict) -> dict:
    """Reverse the option list and rebind A, B, C in that new order."""
    options = list(reversed(item["options"]))
    if len(options) < 2:
        raise ValueError("flip needs at least two options")
    rebound = []
    for index, option in enumerate(options):
        rebound.append({**option, "letter": LETTERS[index]})
    ids = [option["id"] for option in rebound]
    if item["label"] not in ids:
        raise ValueError("flipped options lost the gold label")
    flipped = dict(item)
    flipped["options"] = rebound
    flipped["label_index"] = ids.index(item["label"])
    return flipped


def flip_report(original: list[dict], flipped: list[dict]) -> dict:
    if len(original) != len(flipped) or not original:
        raise ValueError("flip report needs the same non-empty row lists")
    changed = 0
    for left, right in zip(original, flipped):
        if left["case_id"] != right["case_id"] or left["name"] != right["name"]:
            raise ValueError("flip rows are not paired")
        if left["prediction"] != right["prediction"]:
            changed += 1
    count = len(original)
    return {
        "n": count,
        "changed": changed,
        "unchanged": count - changed,
        "change_rate": changed / count,
    }


def gate(candidate: dict, baseline: dict) -> dict:
    """Keep the LoRA when exam accuracy rose and exam ECE fell."""
    accuracy_up = candidate["accuracy"] > baseline["accuracy"]
    ece_down = candidate["ece"] < baseline["ece"]
    return {
        "phase0_accuracy": baseline["accuracy"],
        "phase0_ece": baseline["ece"],
        "phase0_brier": baseline["brier"],
        "accuracy": candidate["accuracy"],
        "ece": candidate["ece"],
        "brier": candidate["brier"],
        "accuracy_up": accuracy_up,
        "ece_down": ece_down,
        "keep_lora": accuracy_up and ece_down,
    }
