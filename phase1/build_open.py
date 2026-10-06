"""Download the open sets and write the pre-split pool."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

from phase1.assemble import HELD_OUT_SOURCE
from phase1.open_data import (
    MULTINLI_TRAIN,
    SNLI_HELD_OUT,
    banking_rows,
    boolq_rows,
    nli_rows,
)
from phase1.rows import NLI_OPTIONS

POOL_DIR = Path("data/phase1/pool")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} {path}")


def _nli_records(name: str, split: str, limit: int, rng: random.Random) -> tuple[list[dict], list[str]]:
    dataset = load_dataset(name, split=split)
    names = list(dataset.features["label"].names)
    if names != NLI_OPTIONS:
        raise RuntimeError(f"{name} label names are {names}")
    order = list(range(len(dataset)))
    rng.shuffle(order)
    records = []
    for index in order:
        row = dataset[index]
        label = int(row["label"])
        premise = str(row["premise"]).strip()
        hypothesis = str(row["hypothesis"]).strip()
        if label < 0 or not premise or not hypothesis:
            continue
        records.append({"premise": premise, "hypothesis": hypothesis, "label": label})
        if len(records) == limit:
            break
    if len(records) < limit:
        raise RuntimeError(f"{name} produced {len(records)} rows, wanted {limit}")
    return records, names


def main() -> None:
    rng = random.Random(0)
    boolq = load_dataset("google/boolq", split="train")
    _write(POOL_DIR / "boolq.jsonl", boolq_rows([dict(row) for row in boolq]))

    banking = load_dataset("mteb/banking77", split="train")
    label_names = list(banking.features["label"].names)
    _write(
        POOL_DIR / "banking77.jsonl",
        banking_rows([dict(row) for row in banking], label_names, random.Random(0)),
    )

    multinli_records, multinli_names = _nli_records("nyu-mll/multi_nli", "train", MULTINLI_TRAIN, rng)
    _write(POOL_DIR / "multinli.jsonl", nli_rows(multinli_records, "multinli", multinli_names))

    snli_records, snli_names = _nli_records("stanfordnlp/snli", "validation", SNLI_HELD_OUT, rng)
    rows = nli_rows(snli_records, HELD_OUT_SOURCE, snli_names)
    if len(rows) != SNLI_HELD_OUT:
        raise RuntimeError(f"expected {SNLI_HELD_OUT} SNLI rows, got {len(rows)}")
    _write(POOL_DIR / "snli.jsonl", rows)


if __name__ == "__main__":
    main()
