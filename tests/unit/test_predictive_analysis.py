from __future__ import annotations

import pandas as pd

from cog_surp.analysis import evaluate_held_out_models


def test_predictive_splits_hold_out_complete_groups() -> None:
    rows = []
    for participant in ("p1", "p2", "p3"):
        for item_index in range(6):
            rows.append(
                {
                    "participant": participant,
                    "item": f"i{item_index}",
                    "n400_mean_voltage_uv": float(item_index),
                    "word_frequency": 4.0,
                    "number_of_letters": 4,
                    "word_position": item_index + 1,
                    "context_word_count": item_index,
                    "target_token_count": 1,
                    "human_cloze_surprisal_nats": 1.0,
                    "human_response_entropy_nats": 1.0,
                    "target_surprisal_nats": float(item_index),
                }
            )
    result = evaluate_held_out_models(pd.DataFrame(rows), folds=3)

    assert set(result["split"]) == {
        "leave-items-out",
        "leave-participants-out",
    }
    assert set(result["model"]) == {
        "lexical-controls",
        "human-cloze",
        "response-entropy",
        "human-predictability",
        "lm-surprisal",
        "combined",
    }
    assert result["estimand"].eq("held-out predictive alignment").all()
