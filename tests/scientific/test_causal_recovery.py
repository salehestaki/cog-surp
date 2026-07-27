from __future__ import annotations

import pytest

from cog_surp.causal import build_default_graph, estimate_condition_effect
from cog_surp.simulation import SyntheticN400Config, SyntheticN400Generator

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:7 variables are assumed unobserved.*:UserWarning"
    ),
    pytest.mark.filterwarnings("ignore:The copy keyword is deprecated.*"),
]


def test_dowhy_recovers_randomized_synthetic_condition_effect() -> None:
    dataset = SyntheticN400Generator(
        SyntheticN400Config(
            seed=123,
            participants=12,
            items=40,
            treatment_effect_uv=-3,
            noise_sd_uv=1.2,
        )
    ).generate()
    data = dataset.trials.rename(
        columns={
            "treatment": "experimental_condition",
            "n400_mean_voltage_uv": "human_n400",
        }
    )

    result = estimate_condition_effect(
        data=data,
        spec=build_default_graph(randomized_condition=True),
        outcome="human_n400",
    )

    assert result.identifier_method == "backdoor"
    assert result.adjustment_set == ()
    assert result.estimate == pytest.approx(
        dataset.expected_window_effect_uv,
        abs=0.3,
    )
    assert "do not prove" in result.interpretation


def test_dowhy_refuters_execute_and_return_effects() -> None:
    dataset = SyntheticN400Generator(
        SyntheticN400Config(
            seed=321,
            participants=8,
            items=20,
            treatment_effect_uv=-2,
            noise_sd_uv=1.0,
        )
    ).generate()
    data = dataset.trials.rename(
        columns={
            "treatment": "experimental_condition",
            "n400_mean_voltage_uv": "human_n400",
        }
    )

    result = estimate_condition_effect(
        data=data,
        spec=build_default_graph(randomized_condition=True),
        outcome="human_n400",
        run_refuters=True,
        random_seed=321,
    )

    assert set(result.refuters) == {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "bootstrap",
        "simulated_unobserved_common_cause",
    }
    assert all(value is not None for value in result.refuters.values())
    assert abs(result.refuters["placebo_treatment"] or 1.0) < 0.5
