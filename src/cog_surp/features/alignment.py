"""Leakage-auditable joins between real EEG, stimuli, and model scores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_derco_feature_table(
    *,
    eeg_paths: Sequence[Path],
    stimuli_path: Path,
    surprisal_path: Path,
) -> Any:
    """Build one real DERCo row per retained participant/item observation."""
    import numpy as np
    import pandas as pd

    if not eeg_paths:
        raise ValueError("at least one EEG single-trial artifact is required")
    eeg = pd.concat(
        [pd.read_parquet(path) for path in eeg_paths],
        ignore_index=True,
    )
    required_eeg = {
        "participant",
        "item",
        "n400_mean_voltage_uv",
        "human_cloze_probability",
        "word_frequency",
        "number_of_letters",
        "word_position",
    }
    missing_eeg = required_eeg - set(eeg.columns)
    if missing_eeg:
        raise ValueError(f"EEG artifacts missing columns: {sorted(missing_eeg)}")
    if eeg.duplicated(["participant", "item"]).any():
        raise ValueError("EEG participant/item keys are not unique")

    stimuli = pd.read_parquet(stimuli_path)
    surprisal = pd.read_parquet(surprisal_path)
    if stimuli["item"].duplicated().any():
        raise ValueError("stimulus item keys are not unique")
    if surprisal["item"].duplicated().any():
        raise ValueError("surprisal item keys are not unique")
    stimulus_features = stimuli[
        [
            "item",
            "context_text",
            "scoreable",
            "raw_exact_cloze",
            "human_predictions",
            "human_response_entropy_nats",
            "human_response_top_probability",
        ]
    ]
    model_features = surprisal[
        [
            "item",
            "target_surprisal_nats",
            "target_surprisal_bits",
            "target_token_count",
            "model_id",
            "model_revision",
            "probability_strategy",
            "request_id",
        ]
    ]
    joined = eeg.merge(
        stimulus_features,
        on="item",
        how="left",
        validate="many_to_one",
    ).merge(
        model_features,
        on="item",
        how="left",
        validate="many_to_one",
    )
    required_joined = ["context_text", "human_response_entropy_nats"]
    missing_rows = {
        column: int(joined[column].isna().sum()) for column in required_joined
    }
    if any(missing_rows.values()):
        raise ValueError(f"unmatched feature rows: {missing_rows}")
    missing_model = joined["target_surprisal_nats"].isna()
    invalid_missing = missing_model & joined["scoreable"].astype(bool)
    if bool(invalid_missing.any()):
        items = sorted(joined.loc[invalid_missing, "item"].unique())
        raise ValueError(f"scoreable items lack model scores: {items[:5]}")
    joined = joined.loc[~missing_model].copy()

    corrected_probability = joined["human_cloze_probability"].clip(0.0, 1.0)
    denominator = joined["human_predictions"] + 1.0
    smoothed = (corrected_probability * joined["human_predictions"] + 0.5) / denominator
    joined["human_cloze_surprisal_nats"] = -np.log(smoothed)
    joined["context_word_count"] = (
        joined["context_text"].str.strip().str.split().str.len().fillna(0).astype(int)
    )
    joined["data_status"] = "real"
    joined["alignment_status"] = "authoritative-publisher-word-id"
    return joined.sort_values(["participant", "article", "word_position"]).reset_index(
        drop=True
    )
