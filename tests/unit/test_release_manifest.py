from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cog_surp import __version__
from cog_surp.provenance.checksums import sha256_file
from cog_surp.release import (
    DASHBOARD_REQUIRED_ARTIFACT_TYPES,
    ArtifactType,
    DataStatus,
    ManifestValidationError,
    ReleaseManifest,
    load_release_manifest,
)


def _artifact_lineage(artifact_type: ArtifactType) -> tuple[str, list[str]]:
    if artifact_type is ArtifactType.FEATURES:
        return "feature-run", ["lm-run"]
    if artifact_type in {
        ArtifactType.PREDICTIVE_SUMMARY,
        ArtifactType.POSTERIOR_SUMMARY,
        ArtifactType.DIAGNOSTICS,
    }:
        return "analysis-run", ["feature-run"]
    if artifact_type is ArtifactType.ROBUSTNESS:
        return "robustness-run", ["feature-run"]
    if artifact_type in {
        ArtifactType.H1_EFFECT,
        ArtifactType.H1_CONDITION_ERP,
        ArtifactType.H1_DIFFERENCE_WAVE,
        ArtifactType.H1_TOPOMAP,
        ArtifactType.H1_PARTICIPANT_QC,
    }:
        return "preprocessing-run", []
    if artifact_type is ArtifactType.H2_EFFECT:
        return "h2-run", ["lm-run"]
    if artifact_type in {ArtifactType.CAUSAL_AUDIT, ArtifactType.CAUSAL_GRAPH}:
        return "causal-run", ["preprocessing-run", "lm-run"]
    if artifact_type in {
        ArtifactType.CLUSTER_METADATA,
        ArtifactType.CLUSTER_SUMMARY,
        ArtifactType.CLUSTER_FIGURE,
    }:
        return "cluster-run", ["preprocessing-run"]
    if artifact_type is ArtifactType.REPORT:
        return "report-run", []
    raise AssertionError(f"unhandled fixture type: {artifact_type}")


def make_release_manifest(
    root: Path,
    *,
    data_status: DataStatus = DataStatus.SYNTHETIC,
) -> tuple[Path, dict[str, Any]]:
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True)
    artifact_records: list[dict[str, Any]] = []
    release_parents = ["features", "h1-effect", "h2-effect"]
    for artifact_type in sorted(
        DASHBOARD_REQUIRED_ARTIFACT_TYPES,
        key=lambda value: value.value,
    ):
        artifact_id = artifact_type.value
        path = artifact_root / f"{artifact_id}.fixture"
        path.write_text(f"{artifact_type.value}\n", encoding="utf-8")
        source_run_id, parent_run_ids = _artifact_lineage(artifact_type)
        parent_artifact_ids: list[str] = []
        if artifact_type in {
            ArtifactType.PREDICTIVE_SUMMARY,
            ArtifactType.POSTERIOR_SUMMARY,
            ArtifactType.DIAGNOSTICS,
            ArtifactType.ROBUSTNESS,
        }:
            parent_artifact_ids = ["features"]
        if artifact_type is ArtifactType.REPORT:
            parent_artifact_ids = release_parents
        artifact_records.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type.value,
                "path": f"artifacts/{path.name}",
                "sha256": sha256_file(path),
                "schema_version": 1,
                "data_status": data_status.value,
                "source_run_id": source_run_id,
                "parent_artifact_ids": parent_artifact_ids,
                "parent_run_ids": parent_run_ids,
                "dataset_id": "fixture-dataset",
                "model_id": None,
                "label": f"{data_status.value} {artifact_type.value}",
            }
        )
    manifest = {
        "manifest_schema_version": 1,
        "project_version": __version__,
        "release_id": "release-fixture",
        "label": "Deterministic fixture release",
        "created_at_utc": "2026-07-28T00:00:00Z",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source_implementation_sha256": "b" * 64,
        "resolved_configuration_hashes": {"fixture": "c" * 64},
        "data_status": data_status.value,
        "datasets": [
            {
                "dataset_id": "fixture-dataset",
                "version": "1",
                "sha256": "d" * 64,
                "data_status": data_status.value,
                "citation": "Deterministic test fixture; no human evidence.",
            }
        ],
        "models": [
            {
                "model_id": "fixture/model",
                "revision": "e" * 40,
                "tokenizer_revision": "e" * 40,
                "scoring_run_ids": ["lm-run"],
            }
        ],
        "runs": {
            "preprocessing_run_id": "preprocessing-run",
            "feature_run_id": "feature-run",
            "lm_scoring_run_ids": ["lm-run"],
            "statistical_analysis_run_ids": [
                "analysis-run",
                "robustness-run",
                "h2-run",
                "cluster-run",
            ],
            "causal_analysis_run_ids": ["causal-run"],
            "report_run_id": "report-run",
        },
        "parent_artifact_ids": release_parents,
        "artifacts": artifact_records,
        "citations": ["Cog-Surp synthetic fixture"],
        "known_limitations": ["Synthetic test data are not human evidence."],
    }
    manifest_path = root / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, manifest


def test_valid_release_manifest_verifies_every_artifact(tmp_path: Path) -> None:
    path, _ = make_release_manifest(tmp_path)

    loaded = load_release_manifest(path)

    assert loaded.release_id == "release-fixture"
    assert loaded.data_status is DataStatus.SYNTHETIC
    assert len(loaded.artifacts) == len(DASHBOARD_REQUIRED_ARTIFACT_TYPES)


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    path, manifest = make_release_manifest(tmp_path)
    artifact = manifest["artifacts"][0]
    (tmp_path / artifact["path"]).write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="checksum mismatch"):
        load_release_manifest(path)


def test_missing_artifact_is_actionable(tmp_path: Path) -> None:
    path, manifest = make_release_manifest(tmp_path)
    artifact = manifest["artifacts"][0]
    (tmp_path / artifact["path"]).unlink()

    with pytest.raises(ManifestValidationError, match="artifact is missing"):
        load_release_manifest(path)


def test_path_escape_is_rejected_before_file_access(tmp_path: Path) -> None:
    _, manifest = make_release_manifest(tmp_path)
    manifest["artifacts"][0]["path"] = "../private-data.parquet"

    with pytest.raises(ValidationError, match="cannot contain"):
        ReleaseManifest.model_validate(manifest)


def test_incompatible_analysis_and_feature_runs_are_rejected(
    tmp_path: Path,
) -> None:
    _, manifest = make_release_manifest(tmp_path)
    posterior = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["artifact_type"] == "posterior-summary"
    )
    posterior["parent_run_ids"] = ["different-feature-run"]

    with pytest.raises(ValidationError, match="does not descend from feature run"):
        ReleaseManifest.model_validate(manifest)


def test_synthetic_artifact_cannot_appear_under_real_manifest(
    tmp_path: Path,
) -> None:
    _, manifest = make_release_manifest(tmp_path, data_status=DataStatus.REAL)
    manifest["artifacts"][0]["data_status"] = "synthetic"

    with pytest.raises(ValidationError, match="real release"):
        ReleaseManifest.model_validate(manifest)


def test_mixed_manifest_requires_both_panel_statuses(tmp_path: Path) -> None:
    _, manifest = make_release_manifest(tmp_path)
    manifest["data_status"] = "mixed"

    with pytest.raises(ValidationError, match="must contain labeled real"):
        ReleaseManifest.model_validate(manifest)


def test_missing_dashboard_role_is_rejected(tmp_path: Path) -> None:
    _, manifest = make_release_manifest(tmp_path)
    manifest["artifacts"] = manifest["artifacts"][:-1]

    with pytest.raises(ValidationError, match="lacks dashboard artifacts"):
        ReleaseManifest.model_validate(manifest)
