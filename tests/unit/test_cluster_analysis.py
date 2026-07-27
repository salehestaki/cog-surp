from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from cog_surp.eeg.cluster import (
    ClusterAnalysisConfig,
    run_sensor_time_cluster_analysis,
)


def test_exploratory_cluster_outputs_are_labeled_and_deterministic(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(4)
    info = mne.create_info(
        ["Fz", "Cz", "CPz", "Pz"],
        sfreq=100,
        ch_types="eeg",
    )
    info.set_montage("standard_1020")
    paths = []
    for participant in range(6):
        related_data = rng.normal(0, 0.1e-6, (4, 101))
        unrelated_data = related_data + rng.normal(0, 0.03e-6, (4, 101))
        unrelated_data[2, 50:71] -= (1.5 + participant * 0.1) * 1e-6
        related = mne.EvokedArray(related_data, info, tmin=-0.2, comment="related")
        unrelated = mne.EvokedArray(
            unrelated_data,
            info,
            tmin=-0.2,
            comment="unrelated",
        )
        path = tmp_path / f"sub-{participant:03d}-ave.fif"
        mne.write_evokeds(path, [related, unrelated], overwrite=True, verbose="ERROR")
        paths.append(path)
    config = ClusterAnalysisConfig(
        schema_version=1,
        analysis_status="exploratory",
        data_status="synthetic",
        n_permutations=32,
        random_seed=9,
        time_window_s=(0.0, 0.8),
        cluster_alpha=0.05,
    )

    artifacts = run_sensor_time_cluster_analysis(
        evoked_paths=paths,
        config=config,
        output_dir=tmp_path / "output",
        run_id="cluster-test",
    )

    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    assert metadata["analysis_status"] == "exploratory"
    assert metadata["data_status"] == "synthetic"
    assert metadata["participants"] == 6
    assert "does not identify exact onset" in metadata["interpretation_boundary"]
    assert artifacts.statistics.is_file()
    assert artifacts.figure.is_file()
