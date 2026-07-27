"""Extraction of prespecified N400 outcomes from DERCo word epochs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_WORD_LABEL = re.compile(r"^(?P<word>.+)_(?P<article>\d+)_(?P<position>\d+)$")
_ITEM_ID = re.compile(r"^topic-(?P<article>\d+)-(?P<position>\d{5})$")


class DERCoPreprocessingConfig(BaseModel):
    """Immutable configuration for publisher-preprocessed DERCo epochs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    dataset_id: Literal["derco"]
    preprocessing_run_name: str
    analysis_status: Literal["smoke-nonconfirmatory", "primary", "robustness"]
    source_preprocessing: Literal["publisher-preprocessed"]
    baseline_s: tuple[float, float]
    n400_window_s: tuple[float, float]
    roi_channels: tuple[str, ...]
    excluded_participants: tuple[str, ...]
    publisher_excluded_participant_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_scientific_bounds(self) -> DERCoPreprocessingConfig:
        baseline_start, baseline_end = self.baseline_s
        window_start, window_end = self.n400_window_s
        if not baseline_start < baseline_end <= 0:
            raise ValueError("baseline must be ordered and nonpositive")
        if not 0 <= window_start < window_end:
            raise ValueError("N400 window must be ordered and post-stimulus")
        if not self.roi_channels:
            raise ValueError("at least one ROI channel is required")
        if (
            self.analysis_status == "primary"
            and len(self.excluded_participants)
            != self.publisher_excluded_participant_count
        ):
            raise ValueError(
                "primary analysis requires the publisher-defined participant "
                "exclusions to be named explicitly"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> DERCoPreprocessingConfig:
        """Read and validate a versioned YAML configuration."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class DERCoExtractionArtifacts:
    """Files and counts produced by one DERCo article extraction."""

    single_trials: Path
    summary: Path
    accepted_trials: int
    rejected_trials: int


def extract_derco_subject_article(
    *,
    dataset_root: Path,
    subject: str,
    article: str,
    config: DERCoPreprocessingConfig,
    output_dir: Path,
    run_id: str,
) -> DERCoExtractionArtifacts:
    """Extract word-level mean voltage from publisher-preprocessed Epochs."""
    import mne
    import numpy as np
    import pandas as pd

    source = (
        dataset_root
        / "EEG_data"
        / "preprocessed"
        / subject
        / article
        / "preprocessed_epoch.fif"
    )
    if not source.exists():
        raise FileNotFoundError(f"DERCo epochs not found: {source}")
    epochs = mne.read_epochs(source, preload=True, verbose="ERROR")
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("DERCo epochs have no publisher metadata")
    required = {
        "WordID",
        "word",
        "NumberOfLetters",
        "WordFrequency",
        "Prediction",
        "p_cloze",
        "level",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"DERCo epoch metadata missing: {sorted(missing)}")
    missing_roi = sorted(set(config.roi_channels) - set(epochs.ch_names))
    if missing_roi:
        raise ValueError(f"ROI channels missing from DERCo epochs: {missing_roi}")
    if subject in config.excluded_participants:
        raise ValueError(f"{subject} is excluded by the configured primary protocol")

    analytical = metadata["Prediction"].fillna(False).astype(bool) & (
        metadata["WordID"].fillna("").astype(str).str.len() > 0
    )
    non_prediction_epochs = int((~analytical).sum())
    epochs = epochs[analytical.to_numpy()]
    metadata = epochs.metadata
    if metadata is None or metadata.empty:
        raise ValueError("DERCo file has no word-prediction epochs after validation")
    epochs.apply_baseline(config.baseline_s, verbose="ERROR")
    roi_uv = epochs.get_data(picks=list(config.roi_channels), units="uV")
    mask = (epochs.times >= config.n400_window_s[0]) & (
        epochs.times <= config.n400_window_s[1]
    )
    if not bool(mask.any()):
        raise ValueError("configured N400 window has no samples")
    outcomes = roi_uv[:, :, mask].mean(axis=(1, 2))

    rows: list[dict[str, Any]] = []
    for trial_index, ((_, values), outcome) in enumerate(
        zip(metadata.iterrows(), outcomes, strict=True),
        start=1,
    ):
        label = str(values["word"])
        match = _WORD_LABEL.fullmatch(label)
        if match is None:
            raise ValueError(f"invalid DERCo epoch word label: {label!r}")
        item = str(values["WordID"])
        item_match = _ITEM_ID.fullmatch(item)
        if item_match is None:
            raise ValueError(f"invalid DERCo WordID: {item!r}")
        rows.append(
            {
                "participant": subject,
                "article": article,
                "item": item,
                "target_word": match.group("word").casefold(),
                "condition": str(values["level"]),
                "trial_number": trial_index,
                "word_position": int(item_match.group("position")),
                "presentation_position": int(match.group("position")),
                "number_of_letters": int(values["NumberOfLetters"]),
                "word_frequency": float(values["WordFrequency"]),
                "prediction_available": bool(values["Prediction"]),
                "human_cloze_probability": float(values["p_cloze"]),
                "preprocessing_run_id": run_id,
                "analysis_status": config.analysis_status,
                "source_preprocessing": config.source_preprocessing,
                "rejection_status": "accepted-publisher-preprocessed",
                "n400_mean_voltage_uv": float(outcome),
                "data_status": "real",
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame["item"].duplicated().any():
        raise ValueError("DERCo epoch metadata contain duplicate WordID values")
    if not bool(np.isfinite(frame["n400_mean_voltage_uv"]).all()):
        raise ValueError("non-finite N400 outcome encountered")

    output_dir.mkdir(parents=True, exist_ok=True)
    single_trials = output_dir / "single-trial-n400.parquet"
    frame.to_parquet(single_trials, index=False)
    summary = output_dir / "extraction-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "participant": subject,
                "article": article,
                "accepted_trials": len(frame),
                "rejected_trials": 0,
                "non_prediction_epochs_excluded": non_prediction_epochs,
                "analysis_status": config.analysis_status,
                "data_status": "real",
                "source_preprocessing": config.source_preprocessing,
                "sign_convention": "More-negative voltage means a larger N400.",
                "baseline_s": list(config.baseline_s),
                "n400_window_s": list(config.n400_window_s),
                "roi_channels": list(config.roi_channels),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return DERCoExtractionArtifacts(
        single_trials=single_trials,
        summary=summary,
        accepted_trials=len(frame),
        rejected_trials=0,
    )
