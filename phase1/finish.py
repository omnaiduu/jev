"""Merge the pool, cut the piles, and write the manifest."""

from __future__ import annotations

import random
from pathlib import Path

from phase1.assemble import (
    assert_gate,
    assert_phase1_sizes,
    assemble,
    manifest_for,
    read_jsonl,
    write_splits,
)

POOL_DIR = Path("data/phase1/pool")
OUT_DIR = Path("data/phase1")


def main() -> None:
    pool = []
    for name in ("boolq", "multinli", "banking77", "synthetic"):
        pool.extend(read_jsonl(POOL_DIR / f"{name}.jsonl"))
    held_out = read_jsonl(POOL_DIR / "snli.jsonl")
    splits = assemble(pool, held_out, random.Random(0))
    manifest = manifest_for(splits)
    manifest["writer"] = "google/gemma-4-12B-it"
    manifest["writer_role"] = "Writes the refund email and the support note. The script writes target."
    manifest["banking77_options"] = (
        "Each row shows the true intent plus 19 other intents, in a random order, "
        "because Banking77 has no option order of its own."
    )
    assert_gate(splits, manifest)
    assert_phase1_sizes(manifest)
    write_splits(splits, manifest, OUT_DIR)
    print(json_summary(manifest))


def json_summary(manifest: dict) -> str:
    import json

    return json.dumps(
        {
            "train_rows": manifest["train_rows"],
            "calibration_rows": manifest["calibration_rows"],
            "held_out_rows": manifest["held_out_rows"],
            "train_by_source": manifest["train_by_source"],
            "calibration_by_source": manifest["calibration_by_source"],
            "train_shuffle_fraction": manifest["train_shuffle_fraction"],
        },
        indent=2,
    )


if __name__ == "__main__":
    main()
