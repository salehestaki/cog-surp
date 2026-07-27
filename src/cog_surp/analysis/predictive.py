"""Leakage-resistant held-out predictive model comparison."""

from __future__ import annotations

from typing import Any


def evaluate_held_out_models(
    frame: Any,
    *,
    folds: int = 5,
    random_seed: int = 20260727,
) -> Any:
    """Compare prespecified fixed-effect models across grouped holdouts."""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    del random_seed  # GroupKFold is deterministic and does not shuffle.
    outcome = "n400_mean_voltage_uv"
    controls = [
        "word_frequency",
        "number_of_letters",
        "word_position",
        "context_word_count",
        "target_token_count",
    ]
    models = {
        "lexical-controls": controls,
        "human-cloze": [*controls, "human_cloze_surprisal_nats"],
        "response-entropy": [*controls, "human_response_entropy_nats"],
        "human-predictability": [
            *controls,
            "human_cloze_surprisal_nats",
            "human_response_entropy_nats",
        ],
        "lm-surprisal": [*controls, "target_surprisal_nats"],
        "combined": [
            *controls,
            "human_cloze_surprisal_nats",
            "human_response_entropy_nats",
            "target_surprisal_nats",
        ],
    }
    required = {outcome, "participant", "item"} | {
        column for columns in models.values() for column in columns
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for split_name, group_column in (
        ("leave-items-out", "item"),
        ("leave-participants-out", "participant"),
    ):
        unique_groups = int(frame[group_column].nunique())
        if unique_groups < 2:
            raise ValueError(f"{split_name} requires at least two groups")
        splitter = GroupKFold(n_splits=min(folds, unique_groups))
        for fold, (train, test) in enumerate(
            splitter.split(frame, groups=frame[group_column]),
            start=1,
        ):
            train_groups = set(frame.iloc[train][group_column])
            test_groups = set(frame.iloc[test][group_column])
            if train_groups & test_groups:
                raise RuntimeError(
                    f"group leakage detected in {split_name} fold {fold}"
                )
            y_train = frame.iloc[train][outcome]
            y_test = frame.iloc[test][outcome]
            for model_name, predictors in models.items():
                estimator = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=1.0),
                )
                estimator.fit(frame.iloc[train][predictors], y_train)
                predicted = estimator.predict(frame.iloc[test][predictors])
                rows.append(
                    {
                        "split": split_name,
                        "fold": fold,
                        "model": model_name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_groups": len(train_groups),
                        "test_groups": len(test_groups),
                        "rmse_uv": float(mean_squared_error(y_test, predicted) ** 0.5),
                        "r2": float(r2_score(y_test, predicted)),
                        "data_status": "real",
                        "estimand": "held-out predictive alignment",
                    }
                )
    return pd.DataFrame.from_records(rows)
