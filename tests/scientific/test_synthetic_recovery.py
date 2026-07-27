from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cog_surp.simulation import SyntheticN400Config, SyntheticN400Generator


def _paired_effect(trials: pd.DataFrame) -> float:
    means = trials.groupby(["participant", "condition"])["n400_mean_voltage_uv"].mean()
    wide = means.unstack("condition")
    return float((wide["unrelated"] - wide["related"]).mean())


def test_non_null_n400_parameter_recovery() -> None:
    dataset = SyntheticN400Generator(
        SyntheticN400Config(
            seed=20260727,
            participants=16,
            items=50,
            treatment_effect_uv=-3.0,
            noise_sd_uv=1.5,
        )
    ).generate()

    recovered = _paired_effect(dataset.trials)

    assert recovered == pytest.approx(dataset.expected_window_effect_uv, abs=0.35)
    assert set(dataset.trials["data_status"]) == {"synthetic"}
    assert recovered < 0


def test_null_fixture_has_no_practical_condition_effect() -> None:
    dataset = SyntheticN400Generator(
        SyntheticN400Config(
            seed=7,
            participants=12,
            items=50,
            treatment_effect_uv=0,
            participant_slope_sd_uv=0,
            item_slope_sd_uv=0,
            noise_sd_uv=1.5,
        )
    ).generate()

    assert _paired_effect(dataset.trials) == pytest.approx(0, abs=0.25)


def test_simulation_is_bitwise_deterministic() -> None:
    config = SyntheticN400Config(seed=42, participants=2, items=2)

    first = SyntheticN400Generator(config).generate()
    second = SyntheticN400Generator(config).generate()

    np.testing.assert_array_equal(first.epochs_uv, second.epochs_uv)
    pd.testing.assert_frame_equal(first.trials, second.trials)


def test_missing_trials_and_artifacts_are_recorded() -> None:
    dataset = SyntheticN400Generator(
        SyntheticN400Config(
            seed=4,
            participants=3,
            items=10,
            missing_trial_probability=0.2,
            blink_probability=1.0,
            line_noise_uv=0.5,
        )
    ).generate()

    assert len(dataset.trials) < 60
    assert dataset.trials["blink_present"].all()
