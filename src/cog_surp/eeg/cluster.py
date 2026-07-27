"""Exploratory sensor-time cluster permutation analysis for ERP condition effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClusterAnalysisConfig(BaseModel):
    """Versioned exploratory cluster-analysis configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    analysis_status: str
    data_status: Literal["real", "synthetic"]
    n_permutations: int = Field(ge=32)
    random_seed: int
    time_window_s: tuple[float, float]
    cluster_alpha: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_exploratory_config(self) -> ClusterAnalysisConfig:
        if self.analysis_status != "exploratory":
            raise ValueError("sensor-time cluster analysis must be exploratory")
        if not 0 <= self.time_window_s[0] < self.time_window_s[1]:
            raise ValueError("cluster time window must be ordered and nonnegative")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ClusterAnalysisConfig:
        """Load a fully specified exploratory configuration."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class ClusterArtifacts:
    """Materialized exploratory sensor-time results."""

    statistics: Path
    summary: Path
    metadata: Path
    figure: Path


def run_sensor_time_cluster_analysis(
    *,
    evoked_paths: list[Path],
    config: ClusterAnalysisConfig,
    output_dir: Path,
    run_id: str,
) -> ClusterArtifacts:
    """Run a two-sided within-participant unrelated-minus-related cluster test."""
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd

    if len(evoked_paths) < 2:
        raise ValueError("cluster analysis requires at least two participants")
    differences = []
    reference_info = None
    times = None
    for path in evoked_paths:
        related = mne.read_evokeds(path, condition="related", verbose="ERROR")
        unrelated = mne.read_evokeds(path, condition="unrelated", verbose="ERROR")
        if related.ch_names != unrelated.ch_names:
            raise ValueError(f"condition channel mismatch in {path}")
        difference = mne.combine_evoked([unrelated, related], weights=[1.0, -1.0])
        difference.crop(
            config.time_window_s[0],
            min(config.time_window_s[1], difference.times[-1]) - np.finfo(float).eps,
        )
        if reference_info is None:
            reference_info = difference.info
            times = difference.times
        else:
            assert times is not None
            if difference.ch_names != list(reference_info["ch_names"]):
                raise ValueError(f"participant channel mismatch in {path}")
            if not np.array_equal(difference.times, times):
                raise ValueError(f"participant time-grid mismatch in {path}")
        differences.append(difference.data.T)
    assert reference_info is not None
    assert times is not None
    values = np.stack(differences)
    adjacency, adjacency_names = mne.channels.find_ch_adjacency(
        reference_info,
        ch_type="eeg",
    )
    if adjacency_names != list(reference_info["ch_names"]):
        raise ValueError("channel adjacency order does not match evoked data")
    statistic, clusters, cluster_p_values, null_distribution = (
        mne.stats.spatio_temporal_cluster_1samp_test(
            values,
            adjacency=adjacency,
            n_permutations=config.n_permutations,
            tail=0,
            seed=config.random_seed,
            out_type="mask",
            verbose=False,
        )
    )
    rows: list[dict[str, Any]] = []
    for index, (mask, p_value) in enumerate(
        zip(clusters, cluster_p_values, strict=True),
        start=1,
    ):
        time_indices, channel_indices = np.where(mask)
        rows.append(
            {
                "cluster": index,
                "p_value": float(p_value),
                "passes_cluster_alpha": bool(p_value <= config.cluster_alpha),
                "start_s": float(times[time_indices.min()]),
                "end_s": float(times[time_indices.max()]),
                "channel_count": len(np.unique(channel_indices)),
                "channels": ",".join(
                    adjacency_names[value]
                    for value in sorted(set(channel_indices.tolist()))
                ),
                "point_count": int(mask.sum()),
                "peak_absolute_t": float(np.abs(statistic[mask]).max()),
                "analysis_status": "exploratory",
                "data_status": config.data_status,
            }
        )
    summary_frame = pd.DataFrame.from_records(
        rows,
        columns=[
            "cluster",
            "p_value",
            "passes_cluster_alpha",
            "start_s",
            "end_s",
            "channel_count",
            "channels",
            "point_count",
            "peak_absolute_t",
            "analysis_status",
            "data_status",
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = output_dir / "cluster-statistics.npz"
    np.savez_compressed(
        statistics,
        statistic=statistic,
        cluster_p_values=cluster_p_values,
        null_distribution=null_distribution,
        times_s=times,
        channel_names=np.asarray(adjacency_names, dtype=str),
    )
    summary = output_dir / "cluster-summary.parquet"
    summary_frame.to_parquet(summary, index=False)
    metadata = output_dir / "cluster-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "analysis_status": "exploratory",
                "data_status": config.data_status,
                "contrast": "unrelated-minus-related",
                "participants": len(evoked_paths),
                "n_permutations": config.n_permutations,
                "random_seed": config.random_seed,
                "time_window_s": list(config.time_window_s),
                "cluster_alpha": config.cluster_alpha,
                "clusters": len(summary_frame),
                "clusters_passing_alpha": int(
                    summary_frame["passes_cluster_alpha"].sum()
                ),
                "interpretation_boundary": (
                    "Cluster inference is exploratory. A significant cluster "
                    "does not identify exact onset, peak latency, source, or "
                    "anatomical location."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure = output_dir / "sensor-time-t-statistic.svg"
    plot, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(
        statistic.T,
        aspect="auto",
        origin="lower",
        extent=(times[0], times[-1], 0, len(adjacency_names)),
        cmap="RdBu_r",
    )
    axis.set(
        xlabel="Time from target onset (s)",
        ylabel="EEG channel index",
        title="Exploratory unrelated-minus-related sensor-time t statistic",
    )
    plot.colorbar(image, ax=axis, label="t statistic")
    plot.tight_layout()
    plot.savefig(figure)
    plt.close(plot)
    return ClusterArtifacts(
        statistics=statistics,
        summary=summary,
        metadata=metadata,
        figure=figure,
    )
