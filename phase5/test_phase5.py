import random

from phase5.build import build_pass, refund_score_row


def test_refund_severity_follows_the_phrase_in_the_email():
    row = refund_score_row(
        {
            "id": "refund-00001",
            "source": "refund_rule",
            "state": "The app crashes every time I open it. Order 1001.",
        }
    )
    assert row is not None
    assert row["type"] == "score"
    assert row["options"][row["target"].index(1.0)] == "3: high"
    assert "app crashes" in row["state"]


def test_pass_shuffles_every_row_and_skips_boolq():
    rows = [
        {
            "id": "refund-1",
            "source": "refund_rule",
            "type": "choice",
            "state": "Please refund the shipping on the crushed box.",
            "question": "Which team?",
            "options": ["billing", "tech", "sales"],
            "target": [1.0, 0.0, 0.0],
            "shuffled": False,
        },
        {
            "id": "boolq-1",
            "source": "boolq",
            "type": "noul",
            "state": "Chamomile is a plant.",
            "question": "is it tea",
            "options": ["no", "yes"],
            "target": [0.0, 1.0],
            "shuffled": False,
        },
        {
            "id": "mnli-1",
            "source": "multinli",
            "type": "choice",
            "state": "Premise: A man sits.\nHypothesis: A man is outdoors.",
            "question": "What is the relationship of the hypothesis to the premise?",
            "options": ["entailment", "neutral", "contradiction"],
            "target": [0.0, 1.0, 0.0],
            "shuffled": False,
        },
    ]
    built = build_pass(rows, random.Random(0))
    assert all(row["shuffled"] for row in built)
    assert all(row["source"] != "boolq" for row in built)
    assert any(row["type"] == "score" for row in built)
    assert any(row["source"] == "multinli" for row in built)
