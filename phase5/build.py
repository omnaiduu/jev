"""One extra training pass: ordered-score rows, and choice rows shuffled again.

The exam has 800 ordered-score questions and the first train file had none.
Refund emails and support notes already contain the cue the script uses for the
level, so the writer is not called again. Choice rows are taken from MultiNLI
and Banking77 and shuffled on every row. BoolQ is left out. The exam is left out.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from phase1.rows import make_row, one_hot, shuffle_options
from phase1.synthetic import FACTS

LEVELS = ["0: none", "1: low", "2: medium", "3: high"]
COUNTS = ["0: none", "1: one", "2: two", "3: three"]
PHRASE_LEVEL = {
    "charged twice": 3,
    "refund the shipping": 1,
    "wrong amount": 2,
    "cancel the renewal": 2,
    "app crashes": 3,
    "error code": 2,
    "will not load": 2,
    "reset link": 1,
    "pricing for annual": 0,
    "upgrade my plan": 1,
    "a written quote": 1,
    "student discount": 0,
}
CHOICE_SOURCES = ("multinli", "banking77")
REPLAY = 4000
SEED = 1


def refund_score_row(row: dict) -> dict | None:
    hits = [level for phrase, level in PHRASE_LEVEL.items() if phrase.lower() in row["state"].lower()]
    if len(hits) != 1:
        return None
    level = hits[0]
    return make_row(
        f"pass2-score-{row['id']}",
        "refund_severity",
        "score",
        row["state"],
        "How severe is this request?",
        LEVELS,
        one_hot(level, len(LEVELS)),
    )


def passage_score_row(row: dict, rng: random.Random) -> dict | None:
    present = [fact for fact in FACTS if fact in row["state"]]
    absent = [fact for fact in FACTS if fact not in row["state"]]
    if not present or len(absent) < 1:
        return None
    level = rng.randint(1, min(3, len(present)))
    if len(absent) < 4 - level:
        return None
    listed = rng.sample(present, level) + rng.sample(absent, 4 - level)
    rng.shuffle(listed)
    question = "How many of these lines are in the note?\n" + "\n".join(f"- {fact}" for fact in listed)
    return make_row(
        f"pass2-score-{row['id']}",
        "passage_count",
        "score",
        row["state"],
        question,
        COUNTS,
        one_hot(level, len(COUNTS)),
    )


def build_pass(rows: list[dict], rng: random.Random) -> list[dict]:
    built = []
    for row in rows:
        if row["source"] == "refund_rule":
            made = refund_score_row(row)
        elif row["source"] == "passage_answer":
            made = passage_score_row(row, rng)
        else:
            continue
        if made is None:
            continue
        built.append(shuffle_options(made, rng))
    choice = [row for row in rows if row["source"] in CHOICE_SOURCES and row["type"] == "choice"]
    rng.shuffle(choice)
    for row in choice[:REPLAY]:
        replay = dict(row)
        replay["id"] = f"pass2-shuffle-{row['id']}"
        built.append(shuffle_options(replay, rng))
    if any("typed-decisions" in row["source"] or row["source"] == "boolq" for row in built):
        raise RuntimeError("pass2 pile picked up exam or BoolQ rows")
    return built


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    with open("data/phase1/train.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    built = build_pass(rows, rng)
    out = Path("data/phase5")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pass.jsonl"
    with path.open("w") as handle:
        for row in built:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "rows": len(built),
        "by_source": dict(Counter(row["source"] for row in built)),
        "by_type": dict(Counter(row["type"] for row in built)),
        "shuffled": sum(1 for row in built if row["shuffled"]),
        "replay": REPLAY,
        "seed": SEED,
        "exam_included": False,
        "boolq_included": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
