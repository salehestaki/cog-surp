"""Crossed participant/item Bayesian model for single-trial N400 outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisConfig(BaseModel):
    """Validated Bayesian analysis configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    dataset_id: Literal["derco"]
    analysis_status: Literal["smoke-nonconfirmatory", "primary", "robustness"]
    outcome: Literal["n400_mean_voltage_uv"]
    formula: str
    draws: int = Field(gt=0)
    tune: int = Field(gt=0)
    chains: int = Field(ge=2)
    cores: int = Field(gt=0)
    target_accept: float = Field(gt=0.5, lt=1.0)
    random_seed: int

    @model_validator(mode="after")
    def validate_primary_sampling(self) -> AnalysisConfig:
        if self.analysis_status == "primary" and (
            self.draws < 500 or self.tune < 500 or self.chains < 2
        ):
            raise ValueError("primary Bayesian analysis requires >=500 draws/tune")
        if "(1|participant)" not in self.formula or "(1|item)" not in self.formula:
            raise ValueError("formula must contain crossed participant/item intercepts")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> AnalysisConfig:
        """Read a fully specified versioned analysis config."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class BayesianArtifacts:
    """Files emitted by a Bayesian hierarchical fit."""

    posterior: Path
    summary: Path
    diagnostics: Path


def _standardize(frame: Any, columns: tuple[str, ...]) -> Any:
    import numpy as np

    result = frame.copy()
    for column in columns:
        values = result[column].astype(float)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"cannot standardize {column}")
        result[f"{column}_z"] = (values - mean) / standard_deviation
    return result


def fit_hierarchical_model(
    *,
    features_path: Path,
    config: AnalysisConfig,
    output_dir: Path,
    run_id: str,
) -> BayesianArtifacts:
    """Fit and serialize the preregistered crossed-effects model."""
    import arviz as az
    import bambi as bmb
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(features_path)
    predictors = (
        "target_surprisal_nats",
        "human_cloze_surprisal_nats",
        "human_response_entropy_nats",
        "word_frequency",
        "number_of_letters",
        "word_position",
    )
    required = {
        config.outcome,
        "participant",
        "item",
        *predictors,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {sorted(missing)}")
    analysis_frame = _standardize(frame, predictors)
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior = output_dir / "posterior.nc"
    if posterior.exists():
        idata = az.from_netcdf(posterior)
        if idata.attrs.get("run_id") != run_id:
            raise ValueError("cached posterior belongs to a different run")
    else:
        model = bmb.Model(config.formula, analysis_frame, family="gaussian")
        idata = model.fit(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            random_seed=config.random_seed,
            target_accept=config.target_accept,
            omit_offsets=True,
            progressbar=False,
        )
        idata = model.predict(
            idata,
            kind="response",
            inplace=False,
            random_seed=config.random_seed,
        )
        idata.attrs.update(
            {
                "run_id": run_id,
                "dataset_id": config.dataset_id,
                "analysis_status": config.analysis_status,
                "data_status": "real",
                "sign_convention": "Negative coefficients mean a larger N400.",
            }
        )
        idata.to_netcdf(posterior)

    summary_frame = az.summary(
        idata,
        kind="all",
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    ).reset_index(names="parameter")
    numeric_columns = [
        column for column in summary_frame.columns if column != "parameter"
    ]
    summary_frame[numeric_columns] = summary_frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    summary_frame["data_status"] = "real"
    summary_frame["analysis_status"] = config.analysis_status
    summary = output_dir / "posterior-summary.parquet"
    summary_frame.to_parquet(summary, index=False)

    posterior_predictive = idata.posterior_predictive[config.outcome]
    predicted = posterior_predictive.mean(("chain", "draw")).to_numpy()
    observed = analysis_frame[config.outcome].to_numpy()
    rhat = summary_frame["r_hat"].dropna()
    bulk_ess = summary_frame["ess_bulk"].dropna()
    divergences = int(idata.sample_stats["diverging"].sum().to_numpy())
    diagnostic_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "data_status": "real",
        "analysis_status": config.analysis_status,
        "records": len(analysis_frame),
        "participants": int(analysis_frame["participant"].nunique()),
        "items": int(analysis_frame["item"].nunique()),
        "formula": config.formula,
        "posterior_predictive_rmse_uv": float(
            np.sqrt(np.mean((observed - predicted) ** 2))
        ),
        "max_rhat": float(rhat.max()) if len(rhat) else None,
        "min_bulk_ess": float(bulk_ess.min()) if len(bulk_ess) else None,
        "divergences": divergences,
        "convergence_pass": bool(
            len(rhat)
            and float(rhat.max()) <= 1.01
            and divergences == 0
            and len(bulk_ess)
            and float(bulk_ess.min()) >= 200
        ),
        "sign_convention": "Negative coefficients mean a larger N400.",
        "claim_boundary": (
            "Surprisal coefficients quantify conditional association and "
            "predictive alignment, not a causal effect of the model on EEG."
        ),
    }
    diagnostics = output_dir / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BayesianArtifacts(
        posterior=posterior,
        summary=summary,
        diagnostics=diagnostics,
    )
