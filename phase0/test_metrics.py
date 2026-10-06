"""CPU checks for the Phase 0 formulas. No model and no exam data."""

from phase0.items import iter_questions
from phase0.metrics import accuracy, brier_score, expected_calibration_error, softmax


def test_softmax_is_a_distribution():
    probs = softmax([2.0, 0.0, 0.0])
    assert abs(sum(probs) - 1) < 1e-9
    assert probs[0] > probs[1] > 0


def test_brier_is_zero_when_the_label_gets_all_the_mass():
    assert brier_score([[1.0, 0.0, 0.0]], [0]) == 0


def test_brier_grows_when_the_model_is_sure_and_wrong():
    sure_and_right = brier_score([[0.9, 0.1]], [0])
    sure_and_wrong = brier_score([[0.1, 0.9]], [0])
    assert sure_and_wrong > sure_and_right


def test_ece_is_zero_when_confidence_matches_the_hit_rate():
    # Ten questions at 0.8, eight of them right: the 0.8 bin is calibrated.
    confidences = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert expected_calibration_error(confidences, correct) == 0


def test_ece_catches_overconfidence():
    confidences = [0.95] * 10
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(confidences, correct) > 0.4


def test_accuracy():
    assert accuracy([True, False, True]) == 2 / 3


def test_score_options_use_level_index_as_the_id():
    row = {
        "id": "case",
        "workflow": "invoice_processing",
        "state": "{}",
        "questions": {
            "risk": {
                "type": "score",
                "instructions": "How risky?",
                "criteria": ["Benign", "Low", "Moderate", "High"],
            }
        },
        "gold": {"risk": {"label": "1"}},
    }
    item = iter_questions(row)[0]
    assert [option["id"] for option in item["options"]] == ["0", "1", "2", "3"]
    assert item["label_index"] == 1
    assert item["options"][1]["letter"] == "B"


def test_noul_without_criteria_defaults_to_false_and_true():
    row = {
        "id": "case",
        "workflow": "invoice_processing",
        "state": "{}",
        "questions": {
            "duplicate": {
                "type": "noul",
                "instructions": "This invoice appears to duplicate an invoice already submitted.",
            }
        },
        "gold": {"duplicate": {"label": "false"}},
    }
    item = iter_questions(row)[0]
    assert [option["id"] for option in item["options"]] == ["false", "true"]
    assert item["label_index"] == 0


def test_multitoken_option_keeps_its_id_and_gets_one_letter():
    row = {
        "id": "case",
        "workflow": "agent_trace_observability",
        "state": "{}",
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What next?",
                "criteria": {"continue": "go", "human_review": "ask"},
            }
        },
        "gold": {"action": {"label": "human_review"}},
    }
    item = iter_questions(row)[0]
    assert item["label_index"] == 1
    assert item["options"][1]["letter"] == "B"
    assert "human_review" in render(item)


def render(item):
    from phase0.items import render_user_prompt

    return render_user_prompt(item)
