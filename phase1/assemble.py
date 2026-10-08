"""Cut train and calibration once, then shuffle 30% of train rows.

SNLI stays in the held-out file. typed-decisions is never an input.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from phase1.rows import FORBIDDEN_SOURCES, shuffle_options, validate_row

CALIBRATION_N = 4000
SHUFFLE_FRACTION = 0.30
HELD_OUT_SOURCE = "snli"
POOL_SOURCES = ("boolq", "multinli", "banking77", "refund_rule", "passage_answer")


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        validate_row(row)
        grouped[row["source"]].append(row)
    return dict(grouped)


def _calibration_counts(grouped: dict[str, list[dict]], calibration_n: int) -> dict[str, int]:
    total = sum(len(rows) for rows in grouped.values())
    if total <= calibration_n:
        raise ValueError(f"pool has {total} rows, which is not larger than calibration {calibration_n}")
    raw = {source: calibration_n * len(rows) / total for source, rows in grouped.items()}
    counts = {source: int(value) for source, value in raw.items()}
    remainder = calibration_n - sum(counts.values())
    fractional = sorted(grouped, key=lambda source: raw[source] - counts[source], reverse=True)
    for source in fractional[:remainder]:
        counts[source] += 1
    for source, rows in grouped.items():
        if counts[source] < 1 or counts[source] >= len(rows):
            raise ValueError(f"{source} cannot be split with calibration count {counts[source]}")
    return counts


def assemble(
    pool: list[dict],
    held_out: list[dict],
    rng: random.Random,
    calibration_n: int = CALIBRATION_N,
) -> dict[str, list[dict]]:
    grouped = _group(pool)
    unknown = set(grouped) - set(POOL_SOURCES)
    if unknown:
        raise ValueError(f"unexpected pool sources: {sorted(unknown)}")
    missing = [source for source in POOL_SOURCES if source not in grouped]
    if missing:
        raise ValueError(f"missing pool sources: {missing}")
    held_sources = {row["source"] for row in held_out}
    if held_sources != {HELD_OUT_SOURCE}:
        raise ValueError(f"held-out source must be {HELD_OUT_SOURCE}, got {sorted(held_sources)}")

    counts = _calibration_counts(grouped, calibration_n)
    train: list[dict] = []
    calibration: list[dict] = []
    for source, rows in grouped.items():
        ordered = list(rows)
        rng.shuffle(ordered)
        cut = counts[source]
        calibration.extend(ordered[:cut])
        train.extend(ordered[cut:])

    shuffle_n = round(SHUFFLE_FRACTION * len(train))
    chosen = set(rng.sample(range(len(train)), shuffle_n))
    train = [
        shuffle_options(row, rng) if index in chosen else dict(row)
        for index, row in enumerate(train)
    ]
    rng.shuffle(train)
    rng.shuffle(calibration)
    return {"train": train, "calibration": calibration, "held_out": list(held_out)}


def _source_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["source"]] += 1
    return dict(sorted(counts.items()))


def manifest_for(splits: dict[str, list[dict]]) -> dict:
    train = splits["train"]
    shuffled = sum(1 for row in train if row["shuffled"])
    return {
        "seed": 0,
        "train_rows": len(train),
        "calibration_rows": len(splits["calibration"]),
        "held_out_rows": len(splits["held_out"]),
        "train_by_source": _source_counts(train),
        "calibration_by_source": _source_counts(splits["calibration"]),
        "held_out_by_source": _source_counts(splits["held_out"]),
        "train_shuffled_rows": shuffled,
        "train_shuffle_fraction": round(shuffled / len(train), 4),
        "typed_decisions_in_train": False,
        "typed_decisions_in_calibration": False,
        "held_out_source": HELD_OUT_SOURCE,
        "held_out_in_train": False,
        "held_out_in_calibration": False,
        "test": {
            "role": "public exam",
            "source": "LocalLLaMA/typed-decisions",
            "config": "all",
            "split": "test",
            "copied_into_phase1_files": False,
        },
    }


def assert_gate(splits: dict[str, list[dict]], manifest: dict) -> None:
    train_ids = {row["id"] for row in splits["train"]}
    calibration_ids = {row["id"] for row in splits["calibration"]}
    held_ids = {row["id"] for row in splits["held_out"]}
    if train_ids & calibration_ids or train_ids & held_ids or calibration_ids & held_ids:
        raise AssertionError("row ids overlap across piles")
    for name in ("train", "calibration", "held_out"):
        blob = "\n".join(json.dumps(row) for row in splits[name])
        for token in FORBIDDEN_SOURCES:
            if token in blob:
                raise AssertionError(f"{token} appears in {name}")
        for row in splits[name]:
            validate_row(row)
    if any(row["source"] == HELD_OUT_SOURCE for row in splits["train"] + splits["calibration"]):
        raise AssertionError("held-out source leaked into train or calibration")
    if any(row["shuffled"] for row in splits["calibration"]):
        raise AssertionError("calibration rows were shuffled")
    fraction = manifest["train_shuffle_fraction"]
    if abs(fraction - SHUFFLE_FRACTION) > 0.001:
        raise AssertionError(f"shuffle fraction {fraction} is not {SHUFFLE_FRACTION}")
    if manifest["typed_decisions_in_train"] or manifest["typed_decisions_in_calibration"]:
        raise AssertionError("manifest says typed-decisions is present")
    if manifest["test"]["copied_into_phase1_files"]:
        raise AssertionError("exam was copied into the phase 1 files")


def assert_phase1_sizes(manifest: dict) -> None:
    if not (35000 <= manifest["train_rows"] <= 45000):
        raise AssertionError(f"train size {manifest['train_rows']} is not about 40k")
    if not (2000 <= manifest["calibration_rows"] <= 5000):
        raise AssertionError("calibration size is outside 2k-5k")


def write_splits(splits: dict[str, list[dict]], manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
