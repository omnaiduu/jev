"""Temperature fit and the four-row keep table. No model code.

One T per dataset, fit on that dataset's calibration logits. A question type
that is missing from calibration stays at T = 1. Accuracy at the chosen T
must equal accuracy at T = 1, because T does not change the winning letter.
The keep bit is the correct count up and ECE down on that dataset's test.
SNLI does not enter the bit.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase0.metrics import accuracy, brier_score, expected_calibration_error
from phase3.fit import fit_temperature
from phase4.score import readout
from phase6.meta import DATASETS, TEACHER_CEILING_CORRECT, TEACHER_CEILING_N


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("summary needs at least one row")
    correct = [bool(record["correct"]) for record in records]
    summary = {
        "n": len(records),
        "correct": sum(correct),
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(
            [record["confidence"] for record in records],
            correct,
        ),
        "brier": brier_score(
            [record["probabilities"] for record in records],
            [record["label_index"] for record in records],
        ),
        "by_type": {},
    }
    for question_type in sorted({record["type"] for record in records}):
        group = [record for record in records if record["type"] == question_type]
        summary["by_type"][question_type] = summarize_flat(group)
    return summary


def summarize_flat(records: list[dict]) -> dict:
    correct = [bool(record["correct"]) for record in records]
    return {
        "n": len(records),
        "correct": sum(correct),
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(
            [record["confidence"] for record in records],
            correct,
        ),
        "brier": brier_score(
            [record["probabilities"] for record in records],
            [record["label_index"] for record in records],
        ),
    }


def score_logits(records: list[dict], temperature_of) -> list[dict]:
    scored = []
    for record in records:
        temperature = float(temperature_of(record["type"]))
        row = readout(
            record["logits"],
            int(record["label_index"]),
            list(record["option_ids"]),
            temperature,
        )
        row["id"] = record["id"]
        row["type"] = record["type"]
        row["case_id"] = record.get("case_id") or record["id"]
        scored.append(row)
    return scored


def fit_dataset_temperature(calibration: list[dict], test: list[dict]) -> dict:
    """Fit one T on every calibration row. Missing test types stay at 1."""
    if not calibration:
        raise ValueError("temperature fit needs calibration rows")
    fitted = fit_temperature(calibration)
    present = {record["type"] for record in calibration}
    missing = sorted({record["type"] for record in test} - present)
    chosen = float(fitted["T"])

    def temperature_of(question_type: str) -> float:
        if question_type not in present:
            return 1.0
        return chosen

    at_one = summarize(score_logits(test, lambda _question_type: 1.0))
    at_t = summarize(score_logits(test, temperature_of))
    if at_one["correct"] != at_t["correct"] or at_one["accuracy"] != at_t["accuracy"]:
        raise RuntimeError("temperature changed the test's winning letter")
    calibration_at_one = summarize(score_logits(calibration, lambda _question_type: 1.0))
    calibration_at_t = summarize(score_logits(calibration, temperature_of))
    if calibration_at_one["correct"] != calibration_at_t["correct"]:
        raise RuntimeError("temperature changed the calibration winning letter")
    return {
        "T": chosen,
        "nll": fitted["nll"],
        "nll_at_1": fitted["nll_at_1"],
        "calibration_n": fitted["n"],
        "calibration_types": sorted(present),
        "missing_types": missing,
        "accuracy_at_1": calibration_at_one["accuracy"],
        "accuracy_at_T": calibration_at_t["accuracy"],
        "ece_at_1": calibration_at_one["ece"],
        "ece_at_T": calibration_at_t["ece"],
        "test_at_1": at_one,
        "test_at_T": at_t,
        "grid_best_nll": fitted["nll"],
    }


def flip_report(rows: list[dict]) -> dict:
    """rows carry prediction at T=1 and flip_prediction from the reversed options."""
    if not rows:
        raise ValueError("flip report needs rows")
    changed = 0
    for row in rows:
        if "flip_prediction" not in row or "prediction" not in row:
            raise ValueError(f"{row.get('id')} is missing a flip prediction")
        if row["prediction"] != row["flip_prediction"]:
            changed += 1
    count = len(rows)
    return {
        "n": count,
        "changed": changed,
        "unchanged": count - changed,
        "change_rate": changed / count,
    }


def result_document(
    dataset: str,
    test_rows: list[dict],
    meta: dict,
    calibration_rows: list[dict] | None = None,
) -> dict:
    """Build the JSON result from logits. Plain runs pass no calibration rows."""
    at_one = score_logits(test_rows, lambda _question_type: 1.0)
    flip_rows = []
    for raw, scored in zip(test_rows, at_one):
        flip_rows.append(
            {
                "id": raw["id"],
                "prediction": scored["prediction"],
                "flip_prediction": raw["flip_prediction"],
            }
        )
    document = {
        "dataset": dataset,
        "model": meta["model"],
        "lora": meta["lora"],
        "adapter_dir": meta.get("adapter_dir"),
        "padding_side": "right",
        "padding_asserts": meta["padding_asserts"],
        "truncated": meta["truncated"],
        "prompt_sha256": meta["prompt_sha256"],
        "index": "last_content",
        "test_at_1": summarize(at_one),
        "flip": flip_report(flip_rows),
    }
    if calibration_rows is None:
        document["temperature"] = {
            "T": 1.0,
            "fit": False,
            "missing_types": [],
            "calibration_types": [],
        }
        document["test_at_T"] = document["test_at_1"]
        return document
    fitted = fit_dataset_temperature(calibration_rows, test_rows)
    document["temperature"] = {
        "T": fitted["T"],
        "fit": True,
        "fit_on": "calibration",
        "nll": fitted["nll"],
        "nll_at_1": fitted["nll_at_1"],
        "calibration_n": fitted["calibration_n"],
        "calibration_types": fitted["calibration_types"],
        "missing_types": fitted["missing_types"],
        "calibration_accuracy_at_1": fitted["accuracy_at_1"],
        "calibration_accuracy_at_T": fitted["accuracy_at_T"],
        "calibration_ece_at_1": fitted["ece_at_1"],
        "calibration_ece_at_T": fitted["ece_at_T"],
    }
    document["test_at_T"] = fitted["test_at_T"]
    if document["test_at_1"]["correct"] != document["test_at_T"]["correct"]:
        raise RuntimeError("temperature changed the test letter")
    return document


def document_at_temperature(dataset: str, test_rows: list[dict], meta: dict, temperature: float, fit_on: str) -> dict:
    """Score with a T that was fit on another file. Used for the SNLI transfer check."""
    document = result_document(dataset, test_rows, meta, None)
    at_t = summarize(score_logits(test_rows, lambda _question_type: temperature))
    if at_t["correct"] != document["test_at_1"]["correct"]:
        raise RuntimeError("temperature changed the winning letter")
    document["temperature"] = {
        "T": temperature,
        "fit": False,
        "fit_on": fit_on,
        "applied_from": fit_on,
        "missing_types": [],
        "calibration_types": ["choice"],
    }
    document["test_at_T"] = at_t
    return document


def keep_decision(plain_correct: int, plain_ece: float, lora_correct: int, lora_ece: float) -> dict:
    accuracy_up = lora_correct > plain_correct
    ece_down = lora_ece < plain_ece
    return {
        "plain_correct": plain_correct,
        "lora_correct": lora_correct,
        "plain_ece": plain_ece,
        "lora_ece": lora_ece,
        "accuracy_up": accuracy_up,
        "ece_down": ece_down,
        "keep": accuracy_up and ece_down,
    }


def keep_table(results_dir: Path) -> dict:
    rows = []
    for dataset in DATASETS:
        plain = json.loads((results_dir / dataset / "plain.json").read_text())
        lora = json.loads((results_dir / dataset / "lora.json").read_text())
        if plain["test_at_1"]["n"] != lora["test_at_1"]["n"]:
            raise RuntimeError(f"{dataset} plain and LoRA tests differ in length")
        if lora["test_at_1"]["correct"] != lora["test_at_T"]["correct"]:
            raise RuntimeError(f"{dataset} temperature changed the test letter")
        decision = keep_decision(
            plain["test_at_1"]["correct"],
            plain["test_at_1"]["ece"],
            lora["test_at_1"]["correct"],
            lora["test_at_T"]["ece"],
        )
        row = {
            "dataset": dataset,
            "n": plain["test_at_1"]["n"],
            "plain_correct": decision["plain_correct"],
            "lora_correct": decision["lora_correct"],
            "plain_ece": decision["plain_ece"],
            "lora_ece_at_1": lora["test_at_1"]["ece"],
            "lora_ece_at_T": decision["lora_ece"],
            "T": lora["temperature"]["T"],
            "missing_types": lora["temperature"]["missing_types"],
            "accuracy_up": decision["accuracy_up"],
            "ece_down": decision["ece_down"],
            "keep": decision["keep"],
            "flip_plain_changed": plain["flip"]["changed"],
            "flip_lora_changed": lora["flip"]["changed"],
        }
        if dataset == "typed-decisions":
            row["teacher_ceiling_correct"] = TEACHER_CEILING_CORRECT
            row["teacher_ceiling_n"] = TEACHER_CEILING_N
            row["correct_means"] = "agreement with the stored teacher label"
        if dataset == "multinli":
            snli_plain_path = results_dir / "snli" / "plain.json"
            snli_lora_path = results_dir / "snli" / "lora.json"
            if snli_plain_path.exists() and snli_lora_path.exists():
                snli_plain = json.loads(snli_plain_path.read_text())
                snli_lora = json.loads(snli_lora_path.read_text())
                row["snli_transfer"] = {
                    "enters_keep": False,
                    "n": snli_plain["test_at_1"]["n"],
                    "plain_correct": snli_plain["test_at_1"]["correct"],
                    "lora_correct": snli_lora["test_at_1"]["correct"],
                    "plain_ece": snli_plain["test_at_1"]["ece"],
                    "lora_ece_at_T": snli_lora["test_at_T"]["ece"],
                    "T": snli_lora["temperature"]["T"],
                }
        rows.append(row)
    return {
        "rule": "keep when the correct count rose and ECE at the fitted T fell, on this dataset's test",
        "rows": rows,
        "kept": [row["dataset"] for row in rows if row["keep"]],
        "not_kept": [row["dataset"] for row in rows if not row["keep"]],
    }


def write_keep_table(results_dir: Path) -> dict:
    payload = keep_table(results_dir)
    path = results_dir / "keep.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(write_keep_table(Path("results/phase6")), indent=2))
