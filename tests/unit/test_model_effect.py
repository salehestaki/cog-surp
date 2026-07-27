from __future__ import annotations

import pandas as pd
import pytest

from cog_surp.analysis.model_effect import matched_target_effect


def _fixture() -> pd.DataFrame:
    rows = []
    for item, related, unrelated in (
        ("target-a", 1.0, 3.0),
        ("target-b", 2.0, 3.0),
        ("target-c", 4.0, 4.0),
    ):
        for condition, value in (("related", related), ("unrelated", unrelated)):
            rows.append(
                {
                    "item": item,
                    "target_word": item.removeprefix("target-"),
                    "condition": condition,
                    "target_surprisal_nats": value,
                    "model_id": "fixture/model",
                    "model_revision": "abc123",
                    "tokenizer_revision": "abc123",
                    "probability_strategy": "boundary-aware",
                }
            )
    return pd.DataFrame.from_records(rows)


def test_matched_target_effect_has_declared_positive_sign() -> None:
    paired, summary = matched_target_effect(_fixture())

    assert len(paired) == 3
    assert summary["n_matched_targets"] == 3
    assert summary["estimate_nats"] == pytest.approx(1.0)
    assert "increased target surprisal" in summary["sign_convention"]


def test_matched_target_effect_rejects_unpaired_items() -> None:
    frame = _fixture().iloc[:-1]

    with pytest.raises(ValueError, match="both condition"):
        matched_target_effect(frame)
