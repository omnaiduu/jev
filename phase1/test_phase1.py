import random

import pytest

from phase1.assemble import assert_gate, assemble, manifest_for
from phase1.open_data import banking_rows, boolq_rows, nli_rows
from phase1.rows import NLI_OPTIONS, one_hot, shuffle_options
from phase1.synthetic import (
    accept_passage,
    accept_refund,
    passage_specs,
    refund_specs,
    row_from_passage,
    row_from_refund,
)


def test_shuffle_moves_the_target_with_the_word():
    row = {
        "id": "ex",
        "source": "refund_rule",
        "type": "choice",
        "state": "I was charged twice.",
        "question": "Which team?",
        "options": ["billing", "tech", "sales"],
        "target": [1.0, 0.0, 0.0],
        "shuffled": False,
    }
    shuffled = shuffle_options(row, random.Random(1))
    assert shuffled["options"][shuffled["target"].index(1.0)] == "billing"
    assert shuffled["shuffled"] is True


def test_boolq_yes_is_the_second_option():
    rows = boolq_rows([{"question": "Did it rain?", "passage": "It rained all night.", "answer": True}])
    assert rows[0]["options"] == ["no", "yes"]
    assert rows[0]["target"] == [0.0, 1.0]
    assert rows[0]["type"] == "noul"


def test_nli_skips_unlabeled_rows():
    records = [
        {"premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0},
        {"premise": "A dog runs.", "hypothesis": "No label.", "label": -1},
    ]
    rows = nli_rows(records, "snli", NLI_OPTIONS)
    assert len(rows) == 1
    assert rows[0]["target"] == one_hot(0, 3)


def test_banking_uses_twenty_options_and_one_gold():
    labels = [f"intent_{index}" for index in range(30)]
    records = [{"text": "Where is my card?", "label": 3}]
    rows = banking_rows(records, labels, random.Random(0))
    assert len(rows[0]["options"]) == 20
    assert rows[0]["options"].count("intent 3") == 1
    assert rows[0]["target"].count(1.0) == 1


def test_refund_writer_cannot_name_the_team():
    spec = refund_specs(1)[0]
    good = (
        "Hello, I was charged twice for order 1000, the canvas shoes. "
        "Please return the extra payment."
    )
    assert spec["phrase"] == "charged twice"
    assert accept_refund(good, spec)
    assert row_from_refund(spec, good)["target"] == [1.0, 0.0, 0.0]
    leaked = good + " Please send this to billing."
    assert accept_refund(leaked, spec) is False


def test_passage_no_rejects_a_note_that_contains_the_asked_fact():
    spec = passage_specs(2, random.Random(0))[1]
    assert spec["answer"] == "no"
    note = " ".join(spec["included"])
    assert accept_passage(note, spec)
    assert row_from_passage(spec, note)["target"] == [1.0, 0.0]
    assert accept_passage(note + " " + spec["asked"], spec) is False


def test_assemble_keeps_snli_out_and_shuffles_thirty_percent():
    pool = []
    pool.extend(boolq_rows([
        {"question": f"Q{i}?", "passage": f"Passage {i} is long enough.", "answer": i % 2 == 0}
        for i in range(40)
    ]))
    pool.extend(
        nli_rows(
            [{"premise": f"P{i}", "hypothesis": f"H{i}", "label": i % 3} for i in range(40)],
            "multinli",
            NLI_OPTIONS,
        )
    )
    labels = [f"intent_{index}" for index in range(25)]
    pool.extend(
        banking_rows(
            [{"text": f"customer message {i}", "label": i % 25} for i in range(40)],
            labels,
            random.Random(0),
        )
    )
    spec = refund_specs(1)[0]
    email = "I was charged twice for the canvas shoes on order 1000 and want the extra payment returned today."
    pool.extend([row_from_refund({**spec, "id": f"refund-{i:05d}"}, email) for i in range(20)])
    passage = passage_specs(1, random.Random(0))[0]
    note = " ".join(passage["included"])
    pool.extend([row_from_passage({**passage, "id": f"passage-{i:05d}", "answer": "yes", "asked": passage["included"][0]}, note) for i in range(20)])
    held = nli_rows(
        [{"premise": f"S{i}", "hypothesis": f"T{i}", "label": 1} for i in range(10)],
        "snli",
        NLI_OPTIONS,
    )
    splits = assemble(pool, held, random.Random(0), calibration_n=30)
    manifest = manifest_for(splits)
    assert_gate(splits, manifest)
    assert manifest["calibration_rows"] == 30
    assert "snli" not in manifest["train_by_source"]
    assert manifest["held_out_by_source"] == {"snli": 10}


def test_forbidden_source_is_rejected():
    from phase1.rows import make_row

    with pytest.raises(ValueError):
        make_row("x", "typed-decisions", "noul", "state", "question", ["no", "yes"], [1.0, 0.0])
