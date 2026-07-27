from __future__ import annotations

import pandas as pd
import pytest

from cog_surp.eeg.cohort import paired_condition_effect


def test_paired_condition_effect_is_equal_participant_weighted() -> None:
    frame = pd.DataFrame(
        {
            "participant": ["sub-001", "sub-002", "sub-003"],
            "related_mean_uv": [1.0, 2.0, 100.0],
            "unrelated_mean_uv": [-1.0, 1.0, 0.0],
            "participant_included": [True, True, False],
        }
    )

    primary = paired_condition_effect(frame)
    sensitivity = paired_condition_effect(frame, included_column=None)

    assert primary["n_participants"] == 2
    assert primary["estimate_uv"] == pytest.approx(-1.5)
    assert sensitivity["n_participants"] == 3
    assert sensitivity["estimate_uv"] == pytest.approx(-103 / 3)
    assert "larger" in primary["sign_convention"]


def test_paired_condition_effect_requires_two_complete_participants() -> None:
    frame = pd.DataFrame(
        {
            "participant": ["sub-001"],
            "related_mean_uv": [1.0],
            "unrelated_mean_uv": [0.0],
            "participant_included": [True],
        }
    )

    with pytest.raises(ValueError, match="at least two"):
        paired_condition_effect(frame)
