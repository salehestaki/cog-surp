from __future__ import annotations

import pytest
from pydantic import ValidationError

from cog_surp.eeg.preprocessing import (
    ERPPreprocessingConfig,
    condition_for_code,
    list_for_code,
)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "erp-core-n400",
        "preprocessing_run_name": "smoke-v1",
        "analysis_status": "smoke-nonconfirmatory",
        "artifact_correction": "none",
        "line_frequency_hz": 60,
        "high_pass_hz": 0.1,
        "low_pass_hz": 30,
        "resample_hz": 256,
        "reference": "average",
        "epoch": {
            "tmin_s": -0.5,
            "tmax_s": 1.5,
            "baseline_s": [-0.2, 0],
        },
        "n400": {"window_s": [0.3, 0.5], "roi_channels": ["Cz", "CPz"]},
        "exclusions": {
            "dataset_defined_participants": True,
            "reject_peak_to_peak_uv": 200,
        },
    }


@pytest.mark.parametrize(
    ("code", "condition", "counterbalance_list"),
    [
        (211, "related", 1),
        (212, "related", 2),
        (221, "unrelated", 1),
        (222, "unrelated", 2),
    ],
)
def test_event_code_mapping(
    code: int, condition: str, counterbalance_list: int
) -> None:
    assert condition_for_code(code) == condition
    assert list_for_code(code) == counterbalance_list


def test_primary_requires_artifact_correction() -> None:
    payload = _valid_config()
    payload["analysis_status"] = "primary"

    with pytest.raises(
        ValidationError, match="requires configured artifact correction"
    ):
        ERPPreprocessingConfig.model_validate(payload)


def test_unknown_event_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an ERP CORE target code"):
        condition_for_code(201)
