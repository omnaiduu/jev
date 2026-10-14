import json
import random

import pytest

from phase1.rows import validate_row as phase1_validate
from phase2.prompts import render_user_prompt
from phase6.meta import prompt_fingerprint
from phase6.padding import assert_right_pad
from phase6.report import (
    fit_dataset_temperature,
    keep_decision,
    keep_table,
    result_document,
)
from phase6.rows import make_row, reverse_options, shuffle_options
from phase6.convert import banking_rows, typed_rows
from phase6.splits import assert_disjoint, assemble_train_side, drop_duplicate_examples, hold_out_test


def test_phase1_still_rejects_typed_decisions():
    row = {
        "id": "t",
        "source": "typed-decisions",
        "type": "choice",
        "state": "A trace.",
        "question": "What next?",
        "options": ["stop", "go"],
        "target": [1.0, 0.0],
        "shuffled": False,
    }
    with pytest.raises(ValueError, match="forbidden"):
        phase1_validate(row)


def test_right_pad_matches_sum_and_left_pad_aborts():
    assert assert_right_pad([1, 1, 1, 0, 0]) == 2
    assert assert_right_pad([1, 1, 1]) == 2
    with pytest.raises(RuntimeError, match="right-padded"):
        assert_right_pad([0, 0, 1, 1, 1])


def test_prompt_is_option_text_and_the_hash_is_stable():
    text = render_user_prompt(
        {
            "id": "r",
            "state": "A trace.",
            "question": "What should the system do?",
            "options": ["Let the agent proceed without interruption.", "Halt the agent now."],
        }
    )
    assert "A. Let the agent proceed without interruption." in text
    assert "continue:" not in text
    first = prompt_fingerprint()
    assert first["padding_side"] == "right"
    assert first["prompt_template"] == "A. {option text}"
    assert first == prompt_fingerprint()
    assert "A. alpha" in first["canonical_user"]


def test_shuffle_moves_a_soft_target_with_the_option_id():
    row = make_row(
        "soft",
        "typed-decisions",
        "choice",
        "A trace.",
        "What next?",
        ["proceed", "halt"],
        ["continue", "stop"],
        [0.25, 0.75],
        "stop",
        "soft",
        "case-1",
        case_id="case-1",
        question_name="action",
    )
    shuffled = shuffle_options(row, random.Random(1))
    assert shuffled["option_ids"][shuffled["target"].index(0.75)] == "stop"
    assert shuffled["label_id"] == "stop"
    assert shuffled["shuffled"] is True
    flipped = reverse_options(row)
    assert flipped["label_id"] == "stop"
    assert flipped["label_index"] == 0
    assert flipped["options"][0] == "halt"


def test_typed_rows_use_the_teacher_vector_and_the_stored_label():
    case = {
        "id": "case-9",
        "workflow": "agent",
        "state": "The agent refunded the customer.",
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What should the system do?",
                "criteria": {
                    "continue": "Let the agent proceed without interruption.",
                    "stop": "Halt the agent now.",
                },
            },
            "needs_review": {
                "type": "noul",
                "instructions": "Does a human need to look?",
                "criteria": {
                    "false": "No human attention is warranted.",
                    "true": "A human should inspect this run.",
                },
            },
            "risk": {
                "type": "score",
                "instructions": "How risky is this?",
                "criteria": ["Benign.", "Severe."],
            },
        },
        "gold": {
            "action": {
                "label": "stop",
                "probabilities": {"continue": 0.2, "stop": 0.8},
            },
            "needs_review": {
                "label": "true",
                "probabilities": {"false": 0.1, "true": 0.9},
            },
            "risk": {
                "label": "1",
                "probabilities": {"0": 0.3, "1": 0.7},
            },
        },
    }
    rows, stats = typed_rows([case], "train")
    assert stats["label_not_mode"] == 0
    assert {row["type"] for row in rows} == {"choice", "noul", "score"}
    action = next(row for row in rows if row["question_name"] == "action")
    text = render_user_prompt(action)
    assert "A. Let the agent proceed without interruption." in text
    assert "A. continue:" not in text
    assert action["target_kind"] == "soft"
    assert action["label_id"] == "stop"
    assert abs(sum(action["target"]) - 1) < 1e-6
    assert action["group_id"] == "case-9"


def test_banking_draws_twenty_options_once():
    labels = [f"intent_{index}" for index in range(30)]
    rows = banking_rows([{"text": "Where is my card?", "label": 4}], labels, random.Random(0), "train")
    assert len(rows[0]["options"]) == 20
    assert rows[0]["label_id"] == "intent_4"
    assert rows[0]["options"][rows[0]["label_index"]] == "intent 4"
    assert rows[0]["shuffled"] is False


def test_calibration_cut_keeps_cases_together_and_holds_out_the_test():
    built = []
    for case in range(4):
        for name, question_type in (("action", "choice"), ("risk", "score")):
            built.append(
                make_row(
                    f"typed-train-case-{case}-{name}",
                    "typed-decisions",
                    question_type,
                    f"state {case}",
                    f"question {name} {case}",
                    ["no", "yes"],
                    ["no", "yes"],
                    [0.2, 0.8],
                    "yes",
                    "soft",
                    f"case-{case}",
                    case_id=f"case-{case}",
                    question_name=name,
                )
            )
    parts = assemble_train_side(built, random.Random(0), 4)
    calibration_cases = {row["group_id"] for row in parts["calibration"]}
    train_cases = {row["group_id"] for row in parts["train"]}
    assert calibration_cases.isdisjoint(train_cases)
    assert len(parts["calibration"]) == 4
    assert all(not row["shuffled"] for row in parts["calibration"])
    assert sum(1 for row in parts["train"] if row["shuffled"]) == 1
    test = [
        make_row(
            "typed-test-case-9-action",
            "typed-decisions",
            "choice",
            "state 9",
            "question action 9",
            ["no", "yes"],
            ["no", "yes"],
            [0.4, 0.6],
            "yes",
            "soft",
            "case-9",
            case_id="case-9",
            question_name="action",
        )
    ]
    assert_disjoint(parts["train"], parts["calibration"], test)


def test_temperature_stays_at_one_for_a_missing_type_and_does_not_change_the_letter():
    calibration = []
    test = []
    for index in range(10):
        calibration.append(
            {
                "id": f"c{index}",
                "type": "choice",
                "label_index": 0,
                "option_ids": ["alpha", "beta"],
                "logits": [2.0, 0.0],
            }
        )
        test.append(
            {
                "id": f"t{index}",
                "type": "choice",
                "label_index": 0,
                "option_ids": ["alpha", "beta"],
                "logits": [2.0, 0.1],
                "flip_prediction": "beta",
            }
        )
    test.append(
        {
            "id": "score-0",
            "type": "score",
            "label_index": 1,
            "option_ids": ["low", "high"],
            "logits": [0.2, 1.5],
            "flip_prediction": "high",
        }
    )
    fitted = fit_dataset_temperature(calibration, test)
    assert fitted["T"] == 0.5
    assert fitted["missing_types"] == ["score"]
    assert fitted["test_at_1"]["correct"] == fitted["test_at_T"]["correct"]
    document = result_document(
        "boolq",
        test,
        {
            "model": "unsloth/gemma-4-E4B-it",
            "lora": True,
            "padding_asserts": len(test),
            "truncated": 0,
            "prompt_sha256": "abc",
        },
        calibration,
    )
    assert document["test_at_1"]["correct"] == document["test_at_T"]["correct"]
    assert document["flip"]["changed"] == 10


def test_duplicate_examples_are_dropped_before_the_cut():
    first = make_row(
        "a",
        "multinli",
        "choice",
        "Premise: A dog runs.\nHypothesis: An animal moves.",
        "What is the relationship of the hypothesis to the premise?",
        ["entailment", "neutral", "contradiction"],
        ["entailment", "neutral", "contradiction"],
        [1.0, 0.0, 0.0],
        "entailment",
        "one_hot",
        "a",
    )
    second = dict(first)
    second["id"] = "b"
    second["group_id"] = "b"
    second["label_id"] = "neutral"
    second["label_index"] = 1
    second["target"] = [0.0, 1.0, 0.0]
    kept, stats = drop_duplicate_examples([first, second])
    assert len(kept) == 1
    assert stats["dropped_duplicate_rows"] == 1
    assert stats["dropped_label_conflicts"] == 1
    _kept, overlap = hold_out_test([first, second], [first])
    assert overlap["dropped_test_overlap"] == 2
    assert _kept == []


def test_keep_needs_both_a_higher_count_and_a_lower_ece(tmp_path):
    assert keep_decision(10, 0.2, 12, 0.1)["keep"] is True
    assert keep_decision(10, 0.2, 12, 0.2)["keep"] is False
    assert keep_decision(10, 0.2, 10, 0.1)["keep"] is False
    for dataset, correct, ece in (
        ("boolq", 8, 0.2),
        ("multinli", 8, 0.2),
        ("banking77", 8, 0.2),
        ("typed-decisions", 8, 0.2),
    ):
        folder = tmp_path / dataset
        folder.mkdir()
        plain = {
            "test_at_1": {"n": 10, "correct": correct, "ece": ece},
            "flip": {"changed": 1},
        }
        lora = {
            "test_at_1": {"n": 10, "correct": correct + 1, "ece": ece},
            "test_at_T": {"n": 10, "correct": correct + 1, "ece": ece - 0.05},
            "temperature": {"T": 1.25, "missing_types": []},
            "flip": {"changed": 2},
        }
        (folder / "plain.json").write_text(json.dumps(plain))
        (folder / "lora.json").write_text(json.dumps(lora))
    table = keep_table(tmp_path)
    assert table["kept"] == ["boolq", "multinli", "banking77", "typed-decisions"]
    typed = next(row for row in table["rows"] if row["dataset"] == "typed-decisions")
    assert typed["teacher_ceiling_correct"] == 1470
