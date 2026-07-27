"""Tests for DERCo stimulus and EEG configuration invariants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from cog_surp.eeg.derco import DERCoPreprocessingConfig
from cog_surp.stimuli.derco import load_derco_stimuli


def test_derco_stimuli_reconstruct_preceding_context(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    frame = pd.DataFrame(
        {
            "survey_code": ["a", "b", "a", "b"],
            "topic_id": [0, 0, 0, 0],
            "task": ["prediction"] * 4,
            "word_id": [
                "topic-0-00001",
                "topic-0-00001",
                "topic-0-00002",
                "topic-0-00002",
            ],
            "response": ["Once", "other", "upon", "upon"],
            "correct_word": ["once", "once", "upon", "upon"],
        }
    )
    frame.to_csv(prediction / "human_prediction_article_0.csv", index=False)

    stimuli = load_derco_stimuli(tmp_path)

    assert stimuli["context_text"].tolist() == ["", "once "]
    assert stimuli["scoreable"].tolist() == [False, True]
    assert stimuli["raw_exact_cloze"].tolist() == [0.5, 1.0]


def test_derco_primary_requires_named_publisher_exclusions() -> None:
    with pytest.raises(ValidationError, match="exclusions to be named"):
        DERCoPreprocessingConfig(
            schema_version=1,
            dataset_id="derco",
            preprocessing_run_name="primary",
            analysis_status="primary",
            source_preprocessing="publisher-preprocessed",
            baseline_s=(-0.2, 0.0),
            n400_window_s=(0.3, 0.5),
            roi_channels=("Cz",),
            excluded_participants=(),
            publisher_excluded_participant_count=2,
        )
