"""Download the four datasets and write train, calibration, and test.

Row counts in the manifest come from the download. The test split is the
official test (BoolQ validation, MultiNLI matched validation, Banking77 test,
typed-decisions test). Calibration is cut from the train side only.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from phase6.convert import banking_rows, boolq_rows, nli_rows, typed_rows
from phase6.meta import CALIBRATION_N, DATASETS, LABEL_SOURCE, SEED, prompt_fingerprint
from phase6.rows import example_key
from phase6.splits import (
    assert_disjoint,
    coverage,
    prepare_train_side,
    shuffled_fraction,
    type_counts,
)

ROOT = Path("data/phase6")
SOURCES = {
    "boolq": ("google/boolq", None, "train", "validation"),
    "multinli": ("nyu-mll/multi_nli", None, "train", "validation_matched"),
    "banking77": ("mteb/banking77", None, "train", "test"),
    "typed-decisions": ("LocalLLaMA/typed-decisions", "all", "train", "test"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"path": path.as_posix(), "rows": len(rows), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _records(dataset) -> list[dict]:
    return [dict(row) for row in dataset]


def _load(name: str, config: str | None, split: str):
    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def _duplicate_keys(rows: list[dict]) -> int:
    keys = [example_key(row) for row in rows]
    return len(keys) - len(set(keys))


def _manifest(
    dataset: str,
    downloaded: dict,
    splits: dict[str, list[dict]],
    files: dict,
    extra: dict,
) -> dict:
    train = splits["train"]
    calibration = splits["calibration"]
    test = splits["test"]
    fingerprint = prompt_fingerprint()
    calibration_types = sorted({row["type"] for row in calibration})
    test_types = sorted({row["type"] for row in test})
    missing = sorted(set(test_types) - set(calibration_types))
    payload = {
        "dataset": dataset,
        "seed": SEED,
        "padding_side": fingerprint["padding_side"],
        "prompt_sha256": fingerprint["prompt_sha256"],
        "prompt_template": fingerprint["prompt_template"],
        "system_prompt": fingerprint["system_prompt"],
        "label_source": LABEL_SOURCE[dataset],
        "target_kind": train[0]["target_kind"],
        "calibration_n_target": CALIBRATION_N,
        "downloaded": downloaded,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_shuffled_rows": sum(1 for row in train if row["shuffled"]),
        "train_shuffle_fraction": round(shuffled_fraction(train), 4),
        "calibration_shuffled_rows": sum(1 for row in calibration if row["shuffled"]),
        "test_shuffled_rows": sum(1 for row in test if row["shuffled"]),
        "train_row_order": "shuffled",
        "calibration_row_order": "source_order",
        "test_row_order": "source_order",
        "types_train": type_counts(train),
        "types_calibration": type_counts(calibration),
        "types_test": type_counts(test),
        "missing_calibration_types": missing,
        "duplicate_keys_train": _duplicate_keys(train),
        "duplicate_keys_test": _duplicate_keys(test),
        "test_ids_in_train": 0,
        "test_ids_in_calibration": 0,
        "files": files,
    }
    payload.update(extra)
    return payload


def _check_fraction(manifest: dict) -> None:
    fraction = manifest["train_shuffle_fraction"]
    if abs(fraction - 0.30) > 0.001:
        raise AssertionError(f"{manifest['dataset']} shuffle fraction {fraction} is not 0.30")
    if manifest["calibration_shuffled_rows"] or manifest["test_shuffled_rows"]:
        raise AssertionError(f"{manifest['dataset']} shuffled calibration or test")
    if manifest["missing_calibration_types"]:
        raise AssertionError(
            f"{manifest['dataset']} calibration is missing {manifest['missing_calibration_types']}"
        )


def build_boolq() -> dict:
    name, config, train_split, test_split = SOURCES["boolq"]
    train_ds = _load(name, config, train_split)
    test_ds = _load(name, config, test_split)
    downloaded = {"train_examples": len(train_ds), "test_examples": len(test_ds), "source": name}
    train_rows = boolq_rows(_records(train_ds), "train")
    test_rows = boolq_rows(_records(test_ds), "validation")
    downloaded["dropped_train"] = downloaded["train_examples"] - len(train_rows)
    downloaded["dropped_test"] = downloaded["test_examples"] - len(test_rows)
    parts, held = prepare_train_side(train_rows, test_rows, random.Random(SEED), CALIBRATION_N)
    downloaded.update(held)
    splits = {"train": parts["train"], "calibration": parts["calibration"], "test": test_rows}
    return _finish("boolq", downloaded, splits, {"one_hot_letter": True})


def build_multinli() -> dict:
    name, config, train_split, test_split = SOURCES["multinli"]
    train_ds = _load(name, config, train_split)
    test_ds = _load(name, config, test_split)
    label_names = list(train_ds.features["label"].names)
    downloaded = {
        "train_examples": len(train_ds),
        "test_examples": len(test_ds),
        "test_split": test_split,
        "label_names": label_names,
        "source": name,
    }
    train_rows, dropped_train = nli_rows(_records(train_ds), "multinli", "train", label_names)
    test_rows, dropped_test = nli_rows(_records(test_ds), "multinli", "validation_matched", label_names)
    downloaded["dropped_train"] = dropped_train
    downloaded["dropped_test"] = dropped_test
    parts, held = prepare_train_side(train_rows, test_rows, random.Random(SEED), CALIBRATION_N)
    downloaded.update(held)
    splits = {"train": parts["train"], "calibration": parts["calibration"], "test": test_rows}
    return _finish("multinli", downloaded, splits, {"one_hot_letter": True})


def build_banking() -> dict:
    name, config, train_split, test_split = SOURCES["banking77"]
    train_ds = _load(name, config, train_split)
    test_ds = _load(name, config, test_split)
    by_id: dict[int, str] = {}
    for row in train_ds:
        index = int(row["label"])
        text = str(row["label_text"])
        if index in by_id and by_id[index] != text:
            raise RuntimeError(f"banking label {index} has two names")
        by_id[index] = text
    label_names = [by_id[index] for index in range(len(by_id))]
    if len(label_names) != 77:
        raise RuntimeError(f"expected 77 banking intents, got {len(label_names)}")
    downloaded = {
        "train_examples": len(train_ds),
        "test_examples": len(test_ds),
        "intents": len(label_names),
        "source": name,
        "option_draw": "true intent plus 19 distractors, drawn once while the split is written",
        "train_option_seed": SEED,
        "test_option_seed": 1,
    }
    train_rows = banking_rows(_records(train_ds), label_names, random.Random(SEED), "train")
    test_rows = banking_rows(_records(test_ds), label_names, random.Random(1), "test")
    downloaded["dropped_train"] = downloaded["train_examples"] - len(train_rows)
    downloaded["dropped_test"] = downloaded["test_examples"] - len(test_rows)
    parts, held = prepare_train_side(train_rows, test_rows, random.Random(SEED), CALIBRATION_N)
    downloaded.update(held)
    splits = {"train": parts["train"], "calibration": parts["calibration"], "test": test_rows}
    return _finish("banking77", downloaded, splits, {"one_hot_letter": True, "options_per_row": 20})


def build_typed() -> dict:
    name, config, train_split, test_split = SOURCES["typed-decisions"]
    train_ds = _load(name, config, train_split)
    test_ds = _load(name, config, test_split)
    train_rows, train_stats = typed_rows(_records(train_ds), "train")
    test_rows, test_stats = typed_rows(_records(test_ds), "test")
    downloaded = {
        "train_cases": len(train_ds),
        "test_cases": len(test_ds),
        "train_questions": train_stats["questions"],
        "test_questions": test_stats["questions"],
        "source": f"{name}:{config}",
        "config": config,
        "train_split": train_split,
        "test_split": test_split,
    }
    parts, held = prepare_train_side(train_rows, test_rows, random.Random(SEED), CALIBRATION_N)
    downloaded.update(held)
    splits = {"train": parts["train"], "calibration": parts["calibration"], "test": test_rows}
    present = set(type_counts(parts["calibration"]))
    if present != {"noul", "choice", "score"} and not {"noul", "choice", "score"} <= present:
        raise AssertionError(f"typed-decisions calibration types are {sorted(present)}")
    extra = {
        "one_hot_letter": False,
        "loss_target": "soft teacher probabilities",
        "accuracy_label": "stored teacher label",
        "stored_label_not_mode": train_stats["label_not_mode"] + test_stats["label_not_mode"],
        "probability_ties": train_stats["ties"] + test_stats["ties"],
        "renormalized_rows": train_stats["renormalized"] + test_stats["renormalized"],
        "calibration_unit": "case",
        "teacher_ceiling_correct": 1470,
        "teacher_ceiling_n": 2000,
    }
    return _finish("typed-decisions", downloaded, splits, extra)


def build_snli() -> dict:
    dataset = load_dataset("stanfordnlp/snli", split="validation")
    label_names = list(dataset.features["label"].names)
    rows, dropped = nli_rows(_records(dataset), "snli", "validation", label_names)
    folder = ROOT / "snli"
    files = {"test": _write_jsonl(folder / "test.jsonl", rows)}
    fingerprint = prompt_fingerprint()
    manifest = {
        "dataset": "snli",
        "role": "transfer check for the multinli LoRA",
        "enters_keep": False,
        "padding_side": fingerprint["padding_side"],
        "prompt_sha256": fingerprint["prompt_sha256"],
        "prompt_template": fingerprint["prompt_template"],
        "label_source": LABEL_SOURCE["snli"],
        "target_kind": "one_hot",
        "downloaded": {
            "validation_examples": len(dataset),
            "dropped": dropped,
            "source": "stanfordnlp/snli",
            "split": "validation",
        },
        "test_rows": len(rows),
        "test_shuffled_rows": sum(1 for row in rows if row["shuffled"]),
        "types_test": type_counts(rows),
        "files": files,
    }
    if manifest["test_shuffled_rows"]:
        raise AssertionError("SNLI test was shuffled")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"snli test {len(rows)} dropped {dropped}", flush=True)
    return manifest


def _finish(dataset: str, downloaded: dict, splits: dict[str, list[dict]], extra: dict) -> dict:
    assert_disjoint(splits["train"], splits["calibration"], splits["test"])
    folder = ROOT / dataset
    files = {
        "train": _write_jsonl(folder / "train.jsonl", splits["train"]),
        "calibration": _write_jsonl(folder / "calibration.jsonl", splits["calibration"]),
        "test": _write_jsonl(folder / "test.jsonl", splits["test"]),
    }
    manifest = _manifest(dataset, downloaded, splits, files, extra)
    _check_fraction(manifest)
    if dataset == "typed-decisions" and not {"noul", "choice", "score"} <= set(manifest["types_calibration"]):
        raise AssertionError("typed-decisions calibration is missing a question type")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{dataset} downloaded {downloaded} train {manifest['train_rows']} "
        f"calibration {manifest['calibration_rows']} test {manifest['test_rows']}",
        flush=True,
    )
    return manifest


def main() -> None:
    manifests = {
        "boolq": build_boolq(),
        "multinli": build_multinli(),
        "banking77": build_banking(),
        "typed-decisions": build_typed(),
        "snli": build_snli(),
    }
    print(json.dumps({name: coverage_line(payload) for name, payload in manifests.items()}, indent=2))
    missing = [name for name in DATASETS if name not in manifests]
    if missing:
        raise RuntimeError(f"missing datasets {missing}")


def coverage_line(manifest: dict) -> dict:
    return {
        "train": manifest.get("train_rows"),
        "calibration": manifest.get("calibration_rows"),
        "test": manifest.get("test_rows"),
        "types": manifest.get("types_test") or coverage([]),
    }


if __name__ == "__main__":
    main()
