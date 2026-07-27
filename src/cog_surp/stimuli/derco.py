"""Word-aligned DERCo stimulus reconstruction and validation."""

from __future__ import annotations

import re
from math import log
from pathlib import Path
from typing import Any

_WORD_ID = re.compile(r"^topic-(?P<article>\d+)-(?P<position>\d{5})$")
_REQUIRED_COLUMNS = frozenset(
    {"survey_code", "topic_id", "task", "word_id", "response", "correct_word"}
)


def load_derco_stimuli(dataset_root: Path) -> Any:
    """Aggregate human predictions into one validated row per story word."""
    import pandas as pd

    paths = sorted((dataset_root / "prediction").glob("human_prediction_article_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no DERCo prediction files found under {dataset_root / 'prediction'}"
        )

    records: list[dict[str, Any]] = []
    for path in paths:
        raw = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if not bool((raw["task"] == "prediction").all()):
            raise ValueError(f"{path.name} contains non-prediction task rows")

        raw = raw.copy()
        raw["response_normalized"] = (
            raw["response"].fillna("").astype(str).str.strip().str.casefold()
        )
        raw["correct_normalized"] = (
            raw["correct_word"].fillna("").astype(str).str.strip().str.casefold()
        )
        raw["exact_match"] = raw["response_normalized"] == raw["correct_normalized"]

        article_rows: list[dict[str, Any]] = []
        for word_id, group in raw.groupby("word_id", sort=False):
            match = _WORD_ID.fullmatch(str(word_id))
            if match is None:
                raise ValueError(f"invalid DERCo word_id: {word_id!r}")
            correct_words = group["correct_normalized"].unique()
            if len(correct_words) != 1 or not correct_words[0]:
                raise ValueError(
                    f"{word_id} does not have exactly one nonempty correct word"
                )
            article = int(match.group("article"))
            if not bool((group["topic_id"] == article).all()):
                raise ValueError(f"{word_id} conflicts with topic_id")
            article_rows.append(
                {
                    "dataset_id": "derco",
                    "source_file": path.name,
                    "article": article,
                    "position": int(match.group("position")),
                    "item": str(word_id),
                    "target_word": correct_words[0],
                    "target_text": correct_words[0],
                    "human_predictions": len(group),
                    "raw_exact_cloze": float(group["exact_match"].mean()),
                    "human_response_entropy_nats": float(
                        -sum(
                            probability * log(probability)
                            for probability in (
                                group["response_normalized"].value_counts(
                                    normalize=True
                                )
                            )
                        )
                    ),
                    "human_response_top_probability": float(
                        group["response_normalized"].value_counts(normalize=True).max()
                    ),
                }
            )

        article_rows.sort(key=lambda row: (row["article"], row["position"]))
        preceding: dict[int, list[str]] = {}
        for row in article_rows:
            article = int(row["article"])
            words = preceding.setdefault(article, [])
            row["context_text"] = " ".join(words) + (" " if words else "")
            row["scoreable"] = bool(words)
            row["condition"] = "continuous-predictability"
            row["anomaly_family"] = "continuous-predictability"
            row["validation_status"] = "publisher-native-human-prediction"
            row["data_status"] = "real-stimulus-metadata"
            words.append(str(row["target_text"]))
        records.extend(article_rows)

    frame = pd.DataFrame.from_records(records)
    if frame["item"].duplicated().any():
        duplicates = sorted(frame.loc[frame["item"].duplicated(), "item"].unique())
        raise ValueError(f"duplicate DERCo word IDs across files: {duplicates[:5]}")
    return frame.sort_values(["article", "position"]).reset_index(drop=True)
