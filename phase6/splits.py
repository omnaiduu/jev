"""Cut calibration from the train side, then shuffle 30 percent of train options.

Calibration and test keep the option order written into the file. typed-decisions
is cut by case, so a situation used to fit temperature is not also a training
situation. The other datasets are one row per example, so the cut is by row.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from phase6.meta import SHUFFLE_FRACTION
from phase6.rows import example_key, shuffle_options, validate_row


def _groups(rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    sizes: dict[str, int] = {}
    for row in rows:
        group_id = row["group_id"]
        if group_id not in sizes:
            order.append(group_id)
            sizes[group_id] = 0
        sizes[group_id] += 1
    return order, sizes


def choose_calibration_groups(rows: list[dict], rng: random.Random, target_rows: int) -> set[str]:
    if target_rows < 1:
        raise ValueError("calibration target must be positive")
    order, sizes = _groups(rows)
    if sum(sizes.values()) <= target_rows:
        raise ValueError(f"{sum(sizes.values())} train-side rows is not larger than calibration {target_rows}")
    group_ids = list(order)
    rng.shuffle(group_ids)
    unique_sizes = set(sizes.values())
    if len(unique_sizes) == 1:
        size = next(iter(unique_sizes))
        if target_rows % size != 0:
            raise ValueError(f"calibration {target_rows} is not a multiple of group size {size}")
        need = target_rows // size
        if need < 1 or need >= len(group_ids):
            raise ValueError(f"cannot hold out {need} groups from {len(group_ids)}")
        return set(group_ids[:need])
    chosen: list[str] = []
    count = 0
    for group_id in group_ids:
        if count >= target_rows:
            break
        chosen.append(group_id)
        count += sizes[group_id]
    if count == 0 or len(chosen) == len(group_ids):
        raise ValueError("calibration cut consumed every group")
    return set(chosen)


def drop_duplicate_examples(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the first copy of an example so train and calibration cannot share it.

    MultiNLI repeats a premise/hypothesis pair a few hundred times, sometimes
    with a different label. The later copy is dropped. The official test file
    is not passed through this function.
    """
    seen: dict[str, str] = {}
    kept = []
    duplicate = 0
    conflict = 0
    for row in rows:
        key = example_key(row)
        if key in seen:
            duplicate += 1
            if seen[key] != row["label_id"]:
                conflict += 1
            continue
        seen[key] = row["label_id"]
        kept.append(row)
    return kept, {"dropped_duplicate_rows": duplicate, "dropped_label_conflicts": conflict}


def hold_out_test(train_rows: list[dict], test_rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop train-side rows whose example also appears in the official test."""
    test_keys = {example_key(row) for row in test_rows}
    kept = [row for row in train_rows if example_key(row) not in test_keys]
    return kept, {"dropped_test_overlap": len(train_rows) - len(kept)}


def prepare_train_side(
    train_rows: list[dict],
    test_rows: list[dict],
    rng: random.Random,
    calibration_n: int,
) -> tuple[dict[str, list[dict]], dict]:
    train_rows, overlap = hold_out_test(train_rows, test_rows)
    train_rows, dedupe = drop_duplicate_examples(train_rows)
    parts = assemble_train_side(train_rows, rng, calibration_n)
    return parts, {**overlap, **dedupe}


def assemble_train_side(rows: list[dict], rng: random.Random, calibration_n: int) -> dict[str, list[dict]]:
    if any(row["shuffled"] for row in rows):
        raise ValueError("train-side rows were shuffled before the cut")
    chosen = choose_calibration_groups(rows, rng, calibration_n)
    calibration = [dict(row) for row in rows if row["group_id"] in chosen]
    train = [dict(row) for row in rows if row["group_id"] not in chosen]
    if not calibration or not train:
        raise ValueError("calibration cut left a side empty")
    shuffle_n = round(SHUFFLE_FRACTION * len(train))
    indexes = set(rng.sample(range(len(train)), shuffle_n))
    train = [shuffle_options(row, rng) if index in indexes else dict(row) for index, row in enumerate(train)]
    rng.shuffle(train)
    for row in calibration:
        validate_row(row)
        if row["shuffled"]:
            raise AssertionError("calibration row was shuffled")
    return {"train": train, "calibration": calibration}


def type_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["type"] for row in rows).items()))


def assert_disjoint(train: list[dict], calibration: list[dict], test: list[dict]) -> None:
    piles = {"train": train, "calibration": calibration, "test": test}
    for name, rows in piles.items():
        if not rows:
            raise AssertionError(f"{name} is empty")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{name} has duplicate row ids")
        for row in rows:
            validate_row(row)
    id_sets = {name: {row["id"] for row in rows} for name, rows in piles.items()}
    key_sets = {name: {example_key(row) for row in rows} for name, rows in piles.items()}
    group_sets = {name: {row["group_id"] for row in rows} for name, rows in piles.items()}
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if id_sets[left] & id_sets[right]:
            raise AssertionError(f"row ids overlap between {left} and {right}")
        if key_sets[left] & key_sets[right]:
            raise AssertionError(f"example keys overlap between {left} and {right}")
        if group_sets[left] & group_sets[right]:
            raise AssertionError(f"group ids overlap between {left} and {right}")
    if any(row["shuffled"] for row in calibration + test):
        raise AssertionError("calibration or test options were shuffled")


def shuffled_fraction(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("shuffle fraction needs rows")
    return sum(1 for row in rows if row["shuffled"]) / len(rows)


def coverage(rows: list[dict]) -> dict[str, list[str]]:
    """Types present, used to see whether temperature has a row of each kind."""
    found: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        found[row["source"]].add(row["type"])
    return {source: sorted(types) for source, types in sorted(found.items())}
