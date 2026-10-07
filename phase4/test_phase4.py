import json
from pathlib import Path

from phase4.score import (
    flip_item,
    flip_report,
    gate,
    readout,
    temperatures_from_phase3,
    winning_index,
)


def _item():
    return {
        "case_id": "case",
        "workflow": "agent_trace_observability",
        "name": "action",
        "type": "choice",
        "label": "continue",
        "label_index": 0,
        "options": [
            {"letter": "A", "id": "continue", "text": "go"},
            {"letter": "B", "id": "human_review", "text": "ask"},
            {"letter": "C", "id": "stop", "text": "halt"},
        ],
    }


def test_temperature_does_not_change_the_winner():
    logits = [2.0, 0.5, -1.0]
    assert winning_index(logits) == winning_index([value / 1.65 for value in logits])
    sharp = readout(logits, 0, ["a", "b", "c"], 1.0)
    flat = readout(logits, 0, ["a", "b", "c"], 1.65)
    assert sharp["prediction"] == flat["prediction"] == "a"
    assert flat["confidence"] < sharp["confidence"]


def test_flip_rebinds_letters_and_moves_the_label():
    flipped = flip_item(_item())
    assert [option["id"] for option in flipped["options"]] == ["stop", "human_review", "continue"]
    assert [option["letter"] for option in flipped["options"]] == ["A", "B", "C"]
    assert flipped["label"] == "continue"
    assert flipped["label_index"] == 2


def test_flip_report_counts_an_option_id_change():
    original = [
        {"case_id": "c", "name": "action", "prediction": "continue"},
        {"case_id": "d", "name": "action", "prediction": "stop"},
    ]
    flipped = [
        {"case_id": "c", "name": "action", "prediction": "continue"},
        {"case_id": "d", "name": "action", "prediction": "human_review"},
    ]
    report = flip_report(original, flipped)
    assert report["changed"] == 1
    assert report["unchanged"] == 1
    assert report["change_rate"] == 0.5


def test_keep_requires_higher_accuracy_and_lower_ece():
    baseline = {"accuracy": 0.376, "ece": 0.366, "brier": 0.905}
    assert gate({"accuracy": 0.50, "ece": 0.20, "brier": 0.40}, baseline)["keep_lora"] is True
    assert gate({"accuracy": 0.50, "ece": 0.40, "brier": 0.40}, baseline)["keep_lora"] is False
    assert gate({"accuracy": 0.30, "ece": 0.20, "brier": 0.40}, baseline)["keep_lora"] is False


def test_committed_temperatures_leave_score_at_one():
    payload = json.loads(Path("results/phase3/temperature.json").read_text())
    assert temperatures_from_phase3(payload) == {"noul": 1.65, "choice": 1.3, "score": 1.0}
