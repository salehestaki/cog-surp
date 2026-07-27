from __future__ import annotations

from pathlib import Path

import pandas as pd

from cog_surp.features import build_derco_feature_table


def test_derco_feature_join_is_many_participants_to_one_item(tmp_path: Path) -> None:
    eeg_path = tmp_path / "eeg.parquet"
    stimuli_path = tmp_path / "stimuli.parquet"
    surprisal_path = tmp_path / "surprisal.parquet"
    pd.DataFrame(
        {
            "participant": ["p1", "p2"],
            "article": ["article_0", "article_0"],
            "item": ["topic-0-00002"] * 2,
            "n400_mean_voltage_uv": [-1.0, 0.5],
            "human_cloze_probability": [0.2, 0.2],
            "word_frequency": [4.0, 4.0],
            "number_of_letters": [4, 4],
            "word_position": [2, 2],
        }
    ).to_parquet(eeg_path, index=False)
    pd.DataFrame(
        {
            "item": ["topic-0-00002"],
            "context_text": ["once "],
            "scoreable": [True],
            "raw_exact_cloze": [0.2],
            "human_predictions": [100],
            "human_response_entropy_nats": [1.2],
            "human_response_top_probability": [0.2],
        }
    ).to_parquet(stimuli_path, index=False)
    pd.DataFrame(
        {
            "item": ["topic-0-00002"],
            "target_surprisal_nats": [3.0],
            "target_surprisal_bits": [4.3],
            "target_token_count": [1],
            "model_id": ["model"],
            "model_revision": ["revision"],
            "probability_strategy": ["boundary-aware"],
            "request_id": ["request"],
        }
    ).to_parquet(surprisal_path, index=False)

    result = build_derco_feature_table(
        eeg_paths=[eeg_path],
        stimuli_path=stimuli_path,
        surprisal_path=surprisal_path,
    )

    assert len(result) == 2
    assert result["alignment_status"].eq("authoritative-publisher-word-id").all()
    assert result["human_cloze_surprisal_nats"].gt(0).all()
