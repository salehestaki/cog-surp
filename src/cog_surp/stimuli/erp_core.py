"""Structured loading and validation of publisher-supplied ERP CORE stimuli."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_erp_core_stimuli(dataset_root: Path) -> Any:
    """Return the two counterbalanced N400 word-pair lists as a tidy DataFrame."""
    import pandas as pd

    stimuli_root = dataset_root / "stimuli"
    paths = sorted(stimuli_root.glob("N400_stimuli_list*_English.txt"))
    if len(paths) != 2:
        raise FileNotFoundError(
            f"expected two English ERP CORE lists in {stimuli_root}, found {len(paths)}"
        )
    records: list[dict[str, Any]] = []
    for path in paths:
        list_number = int(
            path.stem.split("list", maxsplit=1)[1].split("_", maxsplit=1)[0]
        )
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
                if len(row) != 4 or any(not value.strip() for value in row):
                    raise ValueError(
                        f"{path.name}:{row_number} must contain four nonempty fields"
                    )
                for condition, prime, target in (
                    ("related", row[0], row[1]),
                    ("unrelated", row[2], row[3]),
                ):
                    normalized_target = target.strip().upper()
                    records.append(
                        {
                            "dataset_id": "erp-core-n400",
                            "source_file": path.name,
                            "source_row": row_number,
                            "counterbalance_list": list_number,
                            "condition": condition,
                            "anomaly_family": "semantic-association",
                            "prime_word": prime.strip().upper(),
                            "target_word": normalized_target,
                            "item": f"target-{normalized_target.lower()}",
                            "context_text": f"{prime.strip().capitalize()} ",
                            "target_text": target.strip().lower(),
                            "validation_status": "publisher-validated",
                            "data_status": "real-stimulus-metadata",
                        }
                    )
    frame = pd.DataFrame.from_records(records)
    _validate_counterbalancing(frame)
    return frame


def _validate_counterbalancing(frame: Any) -> None:
    if len(frame) != 200:
        raise ValueError(f"expected 200 stimulus conditions, found {len(frame)}")
    if int(frame["target_word"].nunique()) != 100:
        raise ValueError("expected 100 unique target words")
    counts = frame.groupby("target_word")["condition"].agg(["count", "nunique"])
    if not bool(((counts["count"] == 2) & (counts["nunique"] == 2)).all()):
        raise ValueError("every target must occur exactly once per condition")
    list_counts = frame.groupby(["counterbalance_list", "condition"]).size()
    if not bool((list_counts == 50).all()):
        raise ValueError("each list must contain 50 related and 50 unrelated pairs")
