from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cog_surp.reporting import build_research_report


def test_report_builds_from_completed_artifacts(tmp_path: Path) -> None:
    features = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "participant": ["p1"],
            "item": ["i1"],
            "model_id": ["fixture/model"],
        }
    ).to_parquet(features, index=False)
    predictive = tmp_path / "predictive.parquet"
    rows = []
    for split in ("leave-items-out", "leave-participants-out"):
        for model, rmse in (
            ("lexical-controls", 2.0),
            ("combined", 1.8),
            ("human-cloze", 1.9),
            ("response-entropy", 1.95),
        ):
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "mean_rmse_uv": rmse,
                }
            )
    pd.DataFrame.from_records(rows).to_parquet(predictive, index=False)
    posterior = tmp_path / "posterior.parquet"
    pd.DataFrame(
        {
            "parameter": [
                "target_surprisal_nats_z",
                "human_cloze_surprisal_nats_z",
            ],
            "mean": [-0.2, -0.3],
            "hdi95_lb": [-0.4, -0.5],
            "hdi95_ub": [-0.1, -0.1],
        }
    ).to_parquet(posterior, index=False)
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "divergences": 0,
                "max_rhat": 1.0,
                "min_bulk_ess": 500,
                "posterior_predictive_rmse_uv": 2.0,
                "convergence_pass": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.md"

    build_research_report(
        features_path=features,
        predictive_summary_path=predictive,
        posterior_summary_path=posterior,
        diagnostics_path=diagnostics,
        output_path=output,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "H3: incremental EEG explanation" in rendered
    assert "H4: alternatives to raw LM surprisal" in rendered
    assert "does not show" in rendered
