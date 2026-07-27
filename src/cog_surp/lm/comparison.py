"""Auditable comparisons between completed LM scoring strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compare_probability_strategies(
    reference_path: Path,
    comparison_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Compare strategy-only scoring runs on identical model/item inputs."""
    import pandas as pd

    columns = {
        "item",
        "target_surprisal_nats",
        "target_token_positions",
        "model_id",
        "model_revision",
        "probability_strategy",
    }

    def load(path: Path, label: str) -> Any:
        frame = pd.read_parquet(path)
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame["item"].duplicated().any():
            raise ValueError(f"{path} contains duplicate item scores")
        return frame[list(sorted(columns))].rename(
            columns={
                column: f"{label}_{column}" for column in columns if column != "item"
            }
        )

    joined = load(reference_path, "reference").merge(
        load(comparison_path, "comparison"),
        on="item",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("strategy runs have no shared items")
    for field in ("model_id", "model_revision"):
        if not bool(
            (joined[f"reference_{field}"] == joined[f"comparison_{field}"]).all()
        ):
            raise ValueError(f"strategy comparison changed {field}")
    if (
        joined["reference_probability_strategy"].iloc[0]
        == joined["comparison_probability_strategy"].iloc[0]
    ):
        raise ValueError("probability strategies must differ")
    joined["surprisal_difference_nats"] = (
        joined["comparison_target_surprisal_nats"]
        - joined["reference_target_surprisal_nats"]
    )
    joined["absolute_difference_nats"] = joined["surprisal_difference_nats"].abs()
    token_positions_differ = (
        joined["reference_target_token_positions"]
        != joined["comparison_target_token_positions"]
    )
    summary = {
        "schema_version": 1,
        "model_id": joined["reference_model_id"].iloc[0],
        "model_revision": joined["reference_model_revision"].iloc[0],
        "reference_strategy": joined["reference_probability_strategy"].iloc[0],
        "comparison_strategy": joined["comparison_probability_strategy"].iloc[0],
        "shared_items": len(joined),
        "max_absolute_difference_nats": float(joined["absolute_difference_nats"].max()),
        "mean_absolute_difference_nats": float(
            joined["absolute_difference_nats"].mean()
        ),
        "differences_above_1e_9": int(
            (joined["absolute_difference_nats"] > 1e-9).sum()
        ),
        "token_position_differences": int(token_positions_differ.sum()),
        "interpretation": (
            "Strategy sensitivity for this exact model/text fixture only; "
            "zero difference does not establish universal equivalence."
        ),
    }
    return joined, summary
