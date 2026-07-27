"""Cohort aggregation and prespecified ERP CORE H1 estimation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from cog_surp.eeg.preprocessing import ERPPreprocessingConfig
from cog_surp.provenance.checksums import sha256_file


@dataclass(frozen=True, slots=True)
class SubjectRun:
    """One verified subject-level preprocessing run."""

    participant: str
    run_id: str
    run_root: Path
    trials: Path
    summary: Path
    evokeds: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class CohortArtifacts:
    """Files produced by cohort aggregation."""

    participant_qc: Path
    single_trials: Path
    condition_summary: Path
    h1_estimate: Path
    grand_average_evokeds: Path
    condition_figure: Path
    difference_figure: Path
    topomap_figure: Path


def paired_condition_effect(
    participant_means: pd.DataFrame,
    *,
    included_column: str | None = "participant_included",
) -> dict[str, Any]:
    """Estimate the equally weighted unrelated-minus-related participant contrast."""
    from scipy import stats

    required = {"participant", "related_mean_uv", "unrelated_mean_uv"}
    missing = sorted(required - set(participant_means.columns))
    if missing:
        raise ValueError(f"participant means lack required columns: {missing}")
    frame = participant_means
    if included_column is not None:
        if included_column not in frame:
            raise ValueError(f"participant means lack {included_column!r}")
        frame = frame.loc[frame[included_column].astype(bool)]
    differences = (
        frame["unrelated_mean_uv"].astype(float)
        - frame["related_mean_uv"].astype(float)
    ).dropna()
    n = len(differences)
    if n < 2:
        raise ValueError("at least two complete participants are required")
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    test = stats.ttest_1samp(differences.to_numpy(), popmean=0.0)
    return {
        "n_participants": n,
        "estimand": "mean participant-level unrelated minus related voltage",
        "estimate_uv": mean,
        "sd_uv": sd,
        "sem_uv": sem,
        "ci95_low_uv": mean - critical * sem,
        "ci95_high_uv": mean + critical * sem,
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": n - 1,
        "p_value_two_sided": float(test.pvalue),
        "sign_convention": (
            "Negative values mean the unrelated condition produced a larger "
            "(more-negative) N400."
        ),
    }


def _verify_subject_manifest(run_root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("artifacts", []):
        path = run_root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifested artifact is missing: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"manifested artifact checksum failed: {path}")


def discover_erp_core_subject_runs(
    *,
    artifacts_root: Path,
    config: ERPPreprocessingConfig,
    dataset_manifest: Path,
) -> list[SubjectRun]:
    """Find subject runs matching the exact config and upstream dataset manifest."""
    expected_configuration = config.model_dump(mode="json")
    dataset_hash = sha256_file(dataset_manifest)
    by_participant: dict[str, SubjectRun] = {}
    for resolved_path in sorted(artifacts_root.glob("eeg-*/resolved-config.json")):
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if (
            resolved.get("command") != "eeg preprocess"
            or resolved.get("dataset_id") != "erp-core-n400"
            or resolved.get("configuration") != expected_configuration
            or resolved.get("dataset_manifest_sha256") != dataset_hash
        ):
            continue
        subject = str(resolved["subject"]).zfill(3)
        participant = f"sub-{subject}"
        run_root = resolved_path.parent
        eeg_root = run_root / "eeg" / subject
        manifest = run_root / "artifact-manifest.json"
        candidate = SubjectRun(
            participant=participant,
            run_id=run_root.name,
            run_root=run_root,
            trials=eeg_root / "single-trial-n400.parquet",
            summary=eeg_root / "preprocessing-summary.json",
            evokeds=eeg_root / "condition-evokeds-ave.fif",
            manifest=manifest,
        )
        if participant in by_participant:
            raise ValueError(
                f"multiple exact preprocessing runs found for {participant}"
            )
        for path in (
            candidate.trials,
            candidate.summary,
            candidate.evokeds,
            candidate.manifest,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"required subject artifact is missing: {path}")
        _verify_subject_manifest(run_root, manifest)
        by_participant[participant] = candidate
    return [by_participant[key] for key in sorted(by_participant)]


def _participant_tables(
    subject_runs: Sequence[SubjectRun],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np
    import pandas as pd

    qc_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    for run in subject_runs:
        summary = json.loads(run.summary.read_text(encoding="utf-8"))
        trials = pd.read_parquet(run.trials)
        accepted = trials.loc[trials["rejection_status"] == "accepted"].copy()
        means = accepted.groupby("condition")["n400_mean_voltage_uv"].mean()
        included = bool(summary["participant_included"])
        accepted["participant_included"] = included
        trial_frames.append(accepted)
        qc_rows.append(
            {
                "participant": run.participant,
                "preprocessing_run_id": run.run_id,
                "participant_included": included,
                "accepted_trials": int(summary["accepted_trials"]),
                "rejected_trials": int(summary["rejected_trials"]),
                "rejection_fraction": float(summary["rejection_fraction"]),
                "behavior_accuracy": float(summary["behavior_accuracy"]),
                "related_accepted_trials": int(
                    summary["accepted_trials_by_condition"]["related"]
                ),
                "unrelated_accepted_trials": int(
                    summary["accepted_trials_by_condition"]["unrelated"]
                ),
                "related_mean_uv": float(means.get("related", np.nan)),
                "unrelated_mean_uv": float(means.get("unrelated", np.nan)),
                "data_status": "real",
            }
        )
    if not qc_rows:
        raise ValueError("no matching ERP CORE subject runs were found")
    return (
        pd.DataFrame.from_records(qc_rows).sort_values("participant"),
        pd.concat(trial_frames, ignore_index=True),
    )


def aggregate_erp_core_cohort(
    *,
    subject_runs: Sequence[SubjectRun],
    config: ERPPreprocessingConfig,
    output_dir: Path,
    run_id: str,
) -> CohortArtifacts:
    """Aggregate verified subject artifacts and generate the primary H1 result."""
    import matplotlib.pyplot as plt
    import mne
    import pandas as pd

    qc, accepted_trials = _participant_tables(subject_runs)
    included = qc.loc[qc["participant_included"]]
    primary = paired_condition_effect(qc)
    sensitivity = paired_condition_effect(qc, included_column=None)
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_qc = output_dir / "participant-qc.parquet"
    qc.to_parquet(participant_qc, index=False)
    single_trials = output_dir / "accepted-single-trial-n400.parquet"
    accepted_trials.to_parquet(single_trials, index=False)

    condition_rows: list[dict[str, Any]] = []
    for condition in ("related", "unrelated"):
        values = included[f"{condition}_mean_uv"].astype(float)
        condition_rows.append(
            {
                "condition": condition,
                "n_participants": len(values),
                "mean_n400_uv": float(values.mean()),
                "sem_n400_uv": float(values.std(ddof=1) / math.sqrt(len(values))),
                "weighting": "equal participant weight",
                "data_status": "real",
            }
        )
    condition_summary = output_dir / "condition-summary.parquet"
    pd.DataFrame.from_records(condition_rows).to_parquet(condition_summary, index=False)

    h1_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": "erp-core-n400",
        "data_status": "real",
        "analysis_status": "primary",
        "contrast": "unrelated-minus-related",
        "roi_channels": list(config.n400.roi_channels),
        "n400_window_s": list(config.n400.window_s),
        "primary_rule_based_cohort": primary,
        "all_public_participants_sensitivity": sensitivity,
        "participant_counts": {
            "public_available": len(qc),
            "primary_included": int(qc["participant_included"].sum()),
            "primary_excluded": int((~qc["participant_included"]).sum()),
        },
        "interpretation_boundary": (
            "This estimates the randomized condition effect on human voltage. "
            "It does not identify a causal effect of language-model surprisal."
        ),
    }
    h1_estimate = output_dir / "h1-condition-effect.json"
    h1_estimate.write_text(
        json.dumps(h1_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    included_ids = set(included["participant"].astype(str))
    related_evokeds = []
    unrelated_evokeds = []
    for run in subject_runs:
        if run.participant not in included_ids:
            continue
        related_evokeds.append(
            mne.read_evokeds(run.evokeds, condition="related", verbose="ERROR")
        )
        unrelated_evokeds.append(
            mne.read_evokeds(run.evokeds, condition="unrelated", verbose="ERROR")
        )
    grand_related = mne.grand_average(related_evokeds, interpolate_bads=False)
    grand_related.comment = "related"
    grand_unrelated = mne.grand_average(unrelated_evokeds, interpolate_bads=False)
    grand_unrelated.comment = "unrelated"
    difference = mne.combine_evoked(
        [grand_unrelated, grand_related],
        weights=[1.0, -1.0],
    )
    difference.comment = "unrelated-minus-related"
    grand_average_evokeds = output_dir / "grand-average-evokeds-ave.fif"
    mne.write_evokeds(
        grand_average_evokeds,
        [grand_related, grand_unrelated, difference],
        overwrite=True,
        verbose="ERROR",
    )

    roi = list(config.n400.roi_channels)
    condition_figure = output_dir / "grand-average-condition-erp.svg"
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for evoked, label, color in (
        (grand_related, "related", "#4477AA"),
        (grand_unrelated, "unrelated", "#CC6677"),
    ):
        axis.plot(
            evoked.times,
            evoked.copy().pick(roi).data.mean(axis=0) * 1e6,
            label=label,
            color=color,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvspan(*config.n400.window_s, color="#BBBBBB", alpha=0.25)
    axis.set(
        xlabel="Time from target onset (s)",
        ylabel="Mean ROI voltage (microvolts)",
        title=f"ERP CORE N400 grand average (n={len(included)})",
    )
    axis.invert_yaxis()
    axis.legend()
    fig.tight_layout()
    fig.savefig(condition_figure)
    plt.close(fig)

    difference_figure = output_dir / "grand-average-difference-wave.svg"
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        difference.times,
        difference.copy().pick(roi).data.mean(axis=0) * 1e6,
        color="#882255",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvspan(*config.n400.window_s, color="#BBBBBB", alpha=0.25)
    axis.set(
        xlabel="Time from target onset (s)",
        ylabel="Unrelated - related voltage (microvolts)",
        title="ERP CORE N400 difference wave",
    )
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(difference_figure)
    plt.close(fig)

    topomap_figure = output_dir / "n400-difference-topomap.svg"
    center = sum(config.n400.window_s) / 2
    width = config.n400.window_s[1] - config.n400.window_s[0]
    figure = difference.plot_topomap(
        times=[center],
        average=width,
        ch_type="eeg",
        scalings=1e6,
        units="microvolts",
        time_unit="s",
        show=False,
    )
    figure.suptitle("Unrelated - related, 300-500 ms")
    figure.savefig(topomap_figure)
    plt.close(figure)

    return CohortArtifacts(
        participant_qc=participant_qc,
        single_trials=single_trials,
        condition_summary=condition_summary,
        h1_estimate=h1_estimate,
        grand_average_evokeds=grand_average_evokeds,
        condition_figure=condition_figure,
        difference_figure=difference_figure,
        topomap_figure=topomap_figure,
    )
