"""Configured ERP CORE preprocessing and single-trial N400 extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RELATED_CODES = frozenset({211, 212})
UNRELATED_CODES = frozenset({221, 222})
TARGET_CODES = RELATED_CODES | UNRELATED_CODES


class EpochConfig(BaseModel):
    """Epoch and baseline bounds in seconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tmin_s: float
    tmax_s: float
    baseline_s: tuple[float, float]

    @model_validator(mode="after")
    def validate_bounds(self) -> EpochConfig:
        if not self.tmin_s <= self.baseline_s[0] <= self.baseline_s[1] <= 0:
            raise ValueError("baseline must be ordered, nonpositive, and inside epoch")
        if self.tmax_s <= 0:
            raise ValueError("epoch tmax_s must be positive")
        return self


class N400Config(BaseModel):
    """Prespecified N400 outcome window and channels."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_s: tuple[float, float]
    roi_channels: tuple[str, ...]


class ExclusionConfig(BaseModel):
    """Dataset and trial rejection rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_defined_participants: bool
    reject_peak_to_peak_uv: float = Field(gt=0)
    reject_eog_peak_to_peak_uv: float = Field(default=200.0, gt=0)
    eog_rejection_window_s: tuple[float, float] = (-0.2, 0.2)
    max_rejected_fraction: float = Field(default=0.25, gt=0, lt=1)
    min_accepted_trials_per_condition: int = Field(default=30, gt=0)
    min_behavior_accuracy: float = Field(default=0.75, gt=0, le=1)


class ERPPreprocessingConfig(BaseModel):
    """Immutable resolved configuration for one preprocessing run."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    dataset_id: Literal["erp-core-n400"]
    preprocessing_run_name: str
    analysis_status: Literal["smoke-nonconfirmatory", "primary", "robustness"]
    artifact_correction: Literal["none", "ica"]
    line_frequency_hz: float = Field(gt=0)
    high_pass_hz: float = Field(gt=0)
    low_pass_hz: float = Field(gt=0)
    resample_hz: float = Field(gt=0)
    reference: Literal["average"]
    epoch: EpochConfig
    n400: N400Config
    exclusions: ExclusionConfig
    ica_random_state: int = 20260727
    ica_n_components: int = Field(default=20, ge=2)
    ica_decim: int = Field(default=3, gt=0)
    ica_eog_threshold: float = Field(default=3.0, gt=0)
    bad_channel_z_threshold: float = Field(default=5.0, gt=0)
    maximum_interpolated_channels: int = Field(default=4, ge=0)
    interpolation_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.04)

    @model_validator(mode="after")
    def validate_scientific_bounds(self) -> ERPPreprocessingConfig:
        start, end = self.n400.window_s
        if not 0 <= start < end <= self.epoch.tmax_s:
            raise ValueError("N400 window must be ordered and inside the epoch")
        if self.high_pass_hz >= self.low_pass_hz:
            raise ValueError("high-pass must be below low-pass")
        eog_start, eog_end = self.exclusions.eog_rejection_window_s
        if not self.epoch.tmin_s <= eog_start < eog_end <= self.epoch.tmax_s:
            raise ValueError("EOG rejection window must be inside the epoch")
        if self.analysis_status == "primary" and self.artifact_correction == "none":
            raise ValueError("primary analysis requires configured artifact correction")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ERPPreprocessingConfig:
        """Validate a YAML configuration with no silent defaults."""
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(loaded)


@dataclass(frozen=True, slots=True)
class PreprocessingArtifacts:
    """Files produced by a subject-level preprocessing run."""

    single_trials: Path
    summary: Path
    figure: Path
    evokeds: Path
    accepted_trials: int
    rejected_trials: int


def condition_for_code(code: int) -> str:
    """Map publisher target codes to the prespecified contrast."""
    if code in RELATED_CODES:
        return "related"
    if code in UNRELATED_CODES:
        return "unrelated"
    raise ValueError(f"not an ERP CORE target code: {code}")


def list_for_code(code: int) -> int:
    """Return counterbalancing list encoded in the final event-code digit."""
    if code not in TARGET_CODES:
        raise ValueError(f"not an ERP CORE target code: {code}")
    return code % 10


def _detect_bad_eeg_channels(
    raw: Any,
    *,
    z_threshold: float,
) -> list[str]:
    """Detect flat or extreme-amplitude EEG channels before rereferencing."""
    import numpy as np

    picks = raw.copy().pick("eeg").ch_names
    data = raw.get_data(picks=picks)[:, ::10]
    channel_medians = np.median(data, axis=1, keepdims=True)
    robust_amplitude = np.median(np.abs(data - channel_medians), axis=1)
    positive = robust_amplitude[robust_amplitude > 0]
    if not len(positive):
        return list(picks)
    floor = float(np.median(positive)) * 1e-6
    log_amplitude = np.log(np.maximum(robust_amplitude, floor))
    center = float(np.median(log_amplitude))
    mad = float(np.median(np.abs(log_amplitude - center)))
    if mad == 0:
        robust_z = np.zeros_like(log_amplitude)
    else:
        robust_z = 0.67448975 * (log_amplitude - center) / mad
    return sorted(
        channel
        for channel, amplitude, score in zip(
            picks,
            robust_amplitude,
            robust_z,
            strict=True,
        )
        if amplitude <= floor or abs(float(score)) > z_threshold
    )


def preprocess_erp_core_subject(
    *,
    dataset_root: Path,
    subject: str,
    config: ERPPreprocessingConfig,
    output_dir: Path,
    run_id: str,
) -> PreprocessingArtifacts:
    """Create a real single-trial N400 table from one ERP CORE recording."""
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd

    normalized = subject.removeprefix("sub-").zfill(3)
    source = (
        dataset_root
        / f"sub-{normalized}"
        / "eeg"
        / f"sub-{normalized}_task-N400_eeg.set"
    )
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found; fetch the subject without --metadata-only first"
        )
    raw = mne.io.read_raw_eeglab(source, preload=True, verbose="ERROR")
    eog_channels = {"HEOG_left": "eog", "HEOG_right": "eog", "VEOG_lower": "eog"}
    raw.set_channel_types(
        {name: kind for name, kind in eog_channels.items() if name in raw.ch_names}
    )
    raw.set_montage(
        "standard_1020",
        match_case=False,
        on_missing="ignore",
    )
    missing_roi = sorted(set(config.n400.roi_channels) - set(raw.ch_names))
    if missing_roi:
        raise ValueError(f"ROI channels missing from recording: {missing_roi}")
    raw.resample(config.resample_hz)
    raw.filter(
        config.high_pass_hz,
        config.low_pass_hz,
        picks="eeg",
        verbose="ERROR",
    )
    interpolated_channels = _detect_bad_eeg_channels(
        raw,
        z_threshold=config.bad_channel_z_threshold,
    )
    if len(interpolated_channels) > config.maximum_interpolated_channels:
        raise ValueError(
            "bad-channel detector found "
            f"{len(interpolated_channels)} channels, exceeding configured maximum "
            f"{config.maximum_interpolated_channels}: {interpolated_channels}"
        )
    if interpolated_channels:
        raw.info["bads"] = interpolated_channels
        raw.interpolate_bads(
            reset_bads=True,
            origin=config.interpolation_origin_m,
            verbose="ERROR",
        )
    raw.set_eeg_reference(config.reference, projection=False, verbose="ERROR")
    excluded_ica_components: list[int] = []
    if config.artifact_correction == "ica":
        ica_fit_raw = raw.copy().filter(
            1.0,
            config.low_pass_hz,
            picks="eeg",
            verbose="ERROR",
        )
        ica = mne.preprocessing.ICA(
            n_components=config.ica_n_components,
            method="infomax",
            fit_params={"extended": True},
            random_state=config.ica_random_state,
            max_iter="auto",
        )
        ica.fit(
            ica_fit_raw,
            picks="eeg",
            decim=config.ica_decim,
            reject_by_annotation=True,
            verbose="ERROR",
        )
        eog_candidates: set[int] = set()
        for channel in eog_channels:
            if channel not in raw.ch_names:
                continue
            indices, _ = ica.find_bads_eog(
                ica_fit_raw,
                ch_name=channel,
                threshold=config.ica_eog_threshold,
                verbose="ERROR",
            )
            eog_candidates.update(int(index) for index in indices)
        excluded_ica_components = sorted(eog_candidates)
        ica.exclude = excluded_ica_components
        ica.apply(raw, verbose="ERROR")

    events, annotation_ids = mne.events_from_annotations(raw, verbose="ERROR")
    event_ids = {
        label: annotation_ids[label]
        for label in map(str, sorted(TARGET_CODES))
        if label in annotation_ids
    }
    if set(event_ids) != {str(code) for code in TARGET_CODES}:
        raise ValueError(f"target event codes are incomplete: {sorted(event_ids)}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=config.epoch.tmin_s,
        tmax=config.epoch.tmax_s,
        baseline=config.epoch.baseline_s,
        preload=True,
        reject=None,
        detrend=None,
        verbose="ERROR",
    )
    eeg_uv = epochs.get_data(picks="eeg", units="uV")
    peak_to_peak_uv = np.ptp(eeg_uv, axis=2).max(axis=1)
    eog_window = (epochs.times >= config.exclusions.eog_rejection_window_s[0]) & (
        epochs.times <= config.exclusions.eog_rejection_window_s[1]
    )
    eog_signals: list[Any] = []
    if {"HEOG_left", "HEOG_right"} <= set(epochs.ch_names):
        horizontal = epochs.get_data(
            picks=["HEOG_right", "HEOG_left"],
            units="uV",
        )
        eog_signals.append(horizontal[:, 0, :] - horizontal[:, 1, :])
    if {"VEOG_lower", "Fp2"} <= set(epochs.ch_names):
        vertical = epochs.get_data(
            picks=["VEOG_lower", "Fp2"],
            units="uV",
        )
        eog_signals.append(vertical[:, 0, :] - vertical[:, 1, :])
    if not eog_signals:
        raise ValueError("recording lacks channels for bipolar EOG rejection")
    bipolar_eog_uv = np.stack(eog_signals, axis=1)
    eog_peak_to_peak_uv = np.ptp(
        bipolar_eog_uv[:, :, eog_window],
        axis=2,
    ).max(axis=1)
    eeg_rejected = peak_to_peak_uv > config.exclusions.reject_peak_to_peak_uv
    eog_rejected = eog_peak_to_peak_uv > config.exclusions.reject_eog_peak_to_peak_uv
    rejected = eeg_rejected | eog_rejected
    roi_uv = epochs.get_data(picks=list(config.n400.roi_channels), units="uV")
    window_mask = (epochs.times >= config.n400.window_s[0]) & (
        epochs.times <= config.n400.window_s[1]
    )
    n400_uv = roi_uv[:, :, window_mask].mean(axis=(1, 2))
    inverse_event_ids = {value: int(label) for label, value in event_ids.items()}
    codes = [inverse_event_ids[int(event)] for event in epochs.events[:, 2]]
    rows: list[dict[str, Any]] = []
    for index, (
        event,
        code,
        amplitude,
        p2p,
        eog_p2p,
        is_eeg_rejected,
        is_eog_rejected,
        is_rejected,
    ) in enumerate(
        zip(
            epochs.events,
            codes,
            n400_uv,
            peak_to_peak_uv,
            eog_peak_to_peak_uv,
            eeg_rejected,
            eog_rejected,
            rejected,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "participant": f"sub-{normalized}",
                "item": f"target-event-{index:03d}",
                "condition": condition_for_code(code),
                "target_event": code,
                "target_word": None,
                "counterbalance_list": list_for_code(code),
                "trial_number": index,
                "event_sample": int(event[0]),
                "preprocessing_run_id": run_id,
                "analysis_status": config.analysis_status,
                "artifact_correction": config.artifact_correction,
                "rejection_status": "rejected" if is_rejected else "accepted",
                "rejection_reason": (
                    "eeg-and-eog"
                    if is_eeg_rejected and is_eog_rejected
                    else "eeg"
                    if is_eeg_rejected
                    else "eog"
                    if is_eog_rejected
                    else "accepted"
                ),
                "peak_to_peak_uv": float(p2p),
                "eog_peak_to_peak_uv": float(eog_p2p),
                "n400_mean_voltage_uv": float(amplitude),
                "data_status": "real",
            }
        )
    frame = pd.DataFrame.from_records(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    single_trials = output_dir / "single-trial-n400.parquet"
    frame.to_parquet(single_trials, index=False)

    accepted = frame.loc[frame["rejection_status"] == "accepted"]
    summary_rows = []
    for condition, values in accepted.groupby("condition", sort=True):
        amplitudes = values["n400_mean_voltage_uv"]
        summary_rows.append(
            {
                "condition": condition,
                "accepted_trials": len(values),
                "mean_n400_uv": float(amplitudes.mean()),
                "sd_n400_uv": float(amplitudes.std()),
                "data_status": "real",
                "analysis_status": config.analysis_status,
            }
        )
    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary = output_dir / "condition-summary.parquet"
    summary_frame.to_parquet(summary, index=False)

    accepted_epochs = epochs[~rejected]
    related_labels = [
        str(code)
        for code in sorted(RELATED_CODES)
        if str(code) in accepted_epochs.event_id
    ]
    unrelated_labels = [
        str(code)
        for code in sorted(UNRELATED_CODES)
        if str(code) in accepted_epochs.event_id
    ]
    related_evoked = accepted_epochs[related_labels].average()
    related_evoked.comment = "related"
    unrelated_evoked = accepted_epochs[unrelated_labels].average()
    unrelated_evoked.comment = "unrelated"
    evokeds = output_dir / "condition-evokeds-ave.fif"
    mne.write_evokeds(
        evokeds,
        [related_evoked, unrelated_evoked],
        overwrite=True,
        verbose="ERROR",
    )

    figure = output_dir / "condition-erp.svg"
    fig, axis = plt.subplots(figsize=(6, 4))
    roi = list(config.n400.roi_channels)
    related_wave = related_evoked.copy().pick(roi).data.mean(axis=0) * 1e6
    unrelated_wave = unrelated_evoked.copy().pick(roi).data.mean(axis=0) * 1e6
    axis.plot(related_evoked.times, related_wave, label="related", color="#4477AA")
    axis.plot(
        unrelated_evoked.times,
        unrelated_wave,
        label="unrelated",
        color="#CC6677",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvspan(
        config.n400.window_s[0],
        config.n400.window_s[1],
        color="#BBBBBB",
        alpha=0.25,
    )
    axis.set_xlabel("Time from target onset (s)")
    axis.set_ylabel("Mean ROI voltage (microvolts)")
    axis.set_title(f"ERP CORE N400 - {config.analysis_status}")
    axis.legend()
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)

    accepted_count = int((~rejected).sum())
    rejected_count = int(rejected.sum())
    response_descriptions = [str(value) for value in raw.annotations.description]
    correct_responses = response_descriptions.count("201")
    error_responses = response_descriptions.count("202")
    response_count = correct_responses + error_responses
    behavior_accuracy = (
        correct_responses / response_count if response_count else float("nan")
    )
    rejection_fraction = rejected_count / len(rejected)
    accepted_by_condition = {
        condition: int(
            (
                (frame["condition"] == condition)
                & (frame["rejection_status"] == "accepted")
            ).sum()
        )
        for condition in ("related", "unrelated")
    }
    participant_included = bool(
        behavior_accuracy >= config.exclusions.min_behavior_accuracy
        and rejection_fraction <= config.exclusions.max_rejected_fraction
        and all(
            count >= config.exclusions.min_accepted_trials_per_condition
            for count in accepted_by_condition.values()
        )
    )
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "participant": f"sub-{normalized}",
        "data_status": "real",
        "analysis_status": config.analysis_status,
        "sign_convention": "More-negative voltage means a larger N400.",
        "accepted_trials": accepted_count,
        "rejected_trials": rejected_count,
        "rejection_fraction": rejection_fraction,
        "accepted_trials_by_condition": accepted_by_condition,
        "behavior_accuracy": behavior_accuracy,
        "correct_responses": correct_responses,
        "error_responses": error_responses,
        "participant_included": participant_included,
        "participant_exclusion_criteria": {
            "minimum_behavior_accuracy": config.exclusions.min_behavior_accuracy,
            "maximum_rejected_fraction": (config.exclusions.max_rejected_fraction),
            "minimum_accepted_trials_per_condition": (
                config.exclusions.min_accepted_trials_per_condition
            ),
        },
        "ica_excluded_components": excluded_ica_components,
        "interpolated_channels": interpolated_channels,
        "event_code_counts": {
            str(code): int(codes.count(code)) for code in sorted(TARGET_CODES)
        },
    }
    summary_json = output_dir / "preprocessing-summary.json"
    summary_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreprocessingArtifacts(
        single_trials=single_trials,
        summary=summary,
        figure=figure,
        evokeds=evokeds,
        accepted_trials=accepted_count,
        rejected_trials=rejected_count,
    )
