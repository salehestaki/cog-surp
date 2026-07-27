"""Matched-target analysis of experimental condition effects on model surprisal."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelEffectArtifacts:
    """Materialized H2 model-side condition-effect artifacts."""

    paired_targets: Path
    model_summary: Path
    summary_json: Path


def matched_target_effect(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate unrelated-minus-related surprisal over matched target words."""
    from scipy import stats

    required = {
        "item",
        "target_word",
        "condition",
        "target_surprisal_nats",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "probability_strategy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"surprisal artifact lacks required columns: {missing}")
    models = frame["model_id"].drop_duplicates()
    if len(models) != 1:
        raise ValueError("each H2 input must contain exactly one model")
    if set(frame["condition"].dropna().astype(str)) != {"related", "unrelated"}:
        raise ValueError("H2 requires exactly related and unrelated conditions")
    counts = frame.groupby(["item", "condition"]).size()
    if not bool((counts == 1).all()):
        raise ValueError("each target item must occur once in each condition")
    pivot = frame.pivot(
        index=["item", "target_word"],
        columns="condition",
        values="target_surprisal_nats",
    ).reset_index()
    if pivot[["related", "unrelated"]].isna().any(axis=None):
        raise ValueError("each target item must have both condition scores")
    pivot["effect_unrelated_minus_related_nats"] = pivot["unrelated"] - pivot["related"]
    differences = pivot["effect_unrelated_minus_related_nats"].astype(float)
    n = len(differences)
    if n < 2:
        raise ValueError("at least two matched targets are required")
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    test = stats.ttest_1samp(differences.to_numpy(), popmean=0.0)
    first = frame.iloc[0]
    summary = {
        "model_id": str(first["model_id"]),
        "model_revision": str(first["model_revision"]),
        "tokenizer_revision": str(first["tokenizer_revision"]),
        "probability_strategy": str(first["probability_strategy"]),
        "n_matched_targets": n,
        "estimand": "mean matched-target unrelated minus related surprisal",
        "estimate_nats": mean,
        "sd_nats": sd,
        "sem_nats": sem,
        "ci95_low_nats": mean - critical * sem,
        "ci95_high_nats": mean + critical * sem,
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": n - 1,
        "p_value_two_sided": float(test.pvalue),
        "sign_convention": (
            "Positive values mean the unrelated prime increased target surprisal."
        ),
    }
    pivot.insert(0, "model_id", summary["model_id"])
    pivot.insert(1, "model_revision", summary["model_revision"])
    pivot.insert(2, "probability_strategy", summary["probability_strategy"])
    return pivot, summary


def analyze_model_condition_effects(
    *,
    surprisal_paths: list[Path],
    output_dir: Path,
    run_id: str,
) -> ModelEffectArtifacts:
    """Analyze H2 for two or more exact LM surprisal artifacts."""
    import pandas as pd

    if len(surprisal_paths) < 2:
        raise ValueError("H2 robustness requires at least two model artifacts")
    paired_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for path in surprisal_paths:
        paired, summary = matched_target_effect(pd.read_parquet(path))
        paired_frames.append(paired)
        summaries.append(summary)
    model_ids = [str(summary["model_id"]) for summary in summaries]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("H2 inputs contain duplicate model IDs")
    item_sets = [set(frame["item"].astype(str)) for frame in paired_frames]
    if any(items != item_sets[0] for items in item_sets[1:]):
        raise ValueError("H2 model artifacts do not contain identical matched targets")
    comparison = paired_frames[0][
        ["item", "effect_unrelated_minus_related_nats"]
    ].rename(columns={"effect_unrelated_minus_related_nats": model_ids[0]})
    for model_id, frame in zip(model_ids[1:], paired_frames[1:], strict=True):
        comparison = comparison.merge(
            frame[["item", "effect_unrelated_minus_related_nats"]].rename(
                columns={"effect_unrelated_minus_related_nats": model_id}
            ),
            on="item",
            validate="one_to_one",
        )
    correlation_matrix = comparison[model_ids].corr()
    pairwise_correlations = [
        {
            "model_a": model_ids[left],
            "model_b": model_ids[right],
            "pearson": float(correlation_matrix.iloc[left, right]),
        }
        for left in range(len(model_ids))
        for right in range(left + 1, len(model_ids))
    ]
    for summary in summaries:
        summary["data_status"] = "real-stimulus-metadata"
        summary["analysis_status"] = "primary-model-side"

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_targets = output_dir / "h2-paired-targets.parquet"
    pd.concat(paired_frames, ignore_index=True).to_parquet(
        paired_targets,
        index=False,
    )
    model_summary = output_dir / "h2-model-summary.parquet"
    pd.DataFrame.from_records(summaries).to_parquet(model_summary, index=False)
    summary_json = output_dir / "h2-model-effect.json"
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "hypothesis": "H2",
                "contrast": "unrelated-minus-related",
                "models": summaries,
                "pairwise_effect_correlations": pairwise_correlations,
                "interpretation_boundary": (
                    "This is a condition effect on deterministic model scores. "
                    "It is separate from the human EEG effect and does not "
                    "establish model-brain mechanistic similarity."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ModelEffectArtifacts(
        paired_targets=paired_targets,
        model_summary=model_summary,
        summary_json=summary_json,
    )
