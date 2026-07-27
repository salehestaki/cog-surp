from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cog_surp.lm import compare_probability_strategies


def test_strategy_comparison_preserves_model_and_reports_difference(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.parquet"
    comparison = tmp_path / "comparison.parquet"
    common = {
        "item": ["i1", "i2"],
        "target_token_positions": ["[1]", "[2]"],
        "model_id": ["model", "model"],
        "model_revision": ["revision", "revision"],
    }
    pd.DataFrame(
        {
            **common,
            "target_surprisal_nats": [1.0, 2.0],
            "probability_strategy": ["boundary-aware"] * 2,
        }
    ).to_parquet(reference, index=False)
    pd.DataFrame(
        {
            **common,
            "target_surprisal_nats": [1.0, 2.1],
            "probability_strategy": ["subtoken-sum"] * 2,
        }
    ).to_parquet(comparison, index=False)

    _, summary = compare_probability_strategies(reference, comparison)

    assert summary["shared_items"] == 2
    assert summary["max_absolute_difference_nats"] == pytest.approx(0.1)
    assert summary["differences_above_1e_9"] == 1
