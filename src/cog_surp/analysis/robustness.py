"""Cross-model robustness summaries from completed artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compare_two_models(
    *,
    reference_scores: Path,
    comparison_scores: Path,
    reference_posterior: Path,
    comparison_posterior: Path,
    reference_predictive: Path,
    comparison_predictive: Path,
) -> tuple[Any, Any]:
    """Compare two model families without rerunning inference or EEG analysis."""
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    def scores(path: Path, label: str) -> Any:
        frame = pd.read_parquet(path)
        required = {"item", "target_surprisal_nats", "model_id", "model_revision"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing score columns: {sorted(missing)}")
        if frame["item"].duplicated().any():
            raise ValueError(f"{path} has duplicate item scores")
        return frame[
            ["item", "target_surprisal_nats", "model_id", "model_revision"]
        ].rename(
            columns={
                "target_surprisal_nats": f"{label}_surprisal_nats",
                "model_id": f"{label}_model_id",
                "model_revision": f"{label}_model_revision",
            }
        )

    item_comparison = scores(reference_scores, "reference").merge(
        scores(comparison_scores, "comparison"),
        on="item",
        how="inner",
        validate="one_to_one",
    )
    if len(item_comparison) < 2:
        raise ValueError("cross-model comparison requires at least two shared items")
    reference_values = item_comparison["reference_surprisal_nats"]
    comparison_values = item_comparison["comparison_surprisal_nats"]

    def coefficient(path: Path) -> Any:
        frame = pd.read_parquet(path)
        row = frame.loc[frame["parameter"] == "target_surprisal_nats_z"]
        if len(row) != 1:
            raise ValueError(f"{path} lacks one surprisal coefficient")
        return row.iloc[0]

    def predictive(path: Path, split: str) -> Any:
        frame = pd.read_parquet(path)
        row = frame.loc[(frame["split"] == split) & (frame["model"] == "combined")]
        if len(row) != 1:
            raise ValueError(f"{path} lacks combined {split} metrics")
        return row.iloc[0]

    reference_coefficient = coefficient(reference_posterior)
    comparison_coefficient = coefficient(comparison_posterior)
    records = []
    for label, coefficient_row, predictive_path in (
        ("reference", reference_coefficient, reference_predictive),
        ("comparison", comparison_coefficient, comparison_predictive),
    ):
        records.append(
            {
                "model_role": label,
                "model_id": item_comparison[f"{label}_model_id"].iloc[0],
                "model_revision": item_comparison[f"{label}_model_revision"].iloc[0],
                "surprisal_coefficient_uv_per_sd": coefficient_row["mean"],
                "coefficient_hdi95_lb": coefficient_row["hdi95_lb"],
                "coefficient_hdi95_ub": coefficient_row["hdi95_ub"],
                "leave_items_out_rmse_uv": predictive(
                    predictive_path, "leave-items-out"
                )["mean_rmse_uv"],
                "leave_items_out_r2": predictive(predictive_path, "leave-items-out")[
                    "mean_r2"
                ],
                "leave_participants_out_rmse_uv": predictive(
                    predictive_path, "leave-participants-out"
                )["mean_rmse_uv"],
                "leave_participants_out_r2": predictive(
                    predictive_path, "leave-participants-out"
                )["mean_r2"],
                "shared_item_count": len(item_comparison),
                "cross_model_pearson": float(
                    pearsonr(reference_values, comparison_values).statistic
                ),
                "cross_model_spearman": float(
                    spearmanr(reference_values, comparison_values).statistic
                ),
                "data_status": "real",
                "claim_boundary": "cross-model predictive robustness, not homology",
            }
        )
    return item_comparison, pd.DataFrame.from_records(records)
