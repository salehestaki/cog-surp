from __future__ import annotations

from pathlib import Path

import pandas as pd

from cog_surp.lm import MockBackend, SubtokenSumStrategy
from cog_surp.lm.pipeline import score_stimulus_artifact


def test_scoring_pipeline_writes_auditable_parquet(tmp_path: Path) -> None:
    source = tmp_path / "stimuli.parquet"
    output = tmp_path / "surprisal.parquet"
    pd.DataFrame.from_records(
        [
            {
                "source_file": "list1.txt",
                "source_row": 1,
                "counterbalance_list": 1,
                "condition": "related",
                "context_text": "before ",
                "target_text": "after",
            }
        ]
    ).to_parquet(source, index=False)

    count = score_stimulus_artifact(
        stimuli_path=source,
        backend=MockBackend(SubtokenSumStrategy()),
        output_path=output,
        batch_size=1,
    )
    result = pd.read_parquet(output)

    assert count == 1
    assert result.loc[0, "target_surprisal_nats"] == 5.0
    assert result.loc[0, "probability_strategy"] == "subtoken-sum"
    assert '"log_probability": -5.0' in result.loc[0, "token_observations"]
