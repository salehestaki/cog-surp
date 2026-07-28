"""Manifest-only dashboard bundle loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cog_surp.release.manifest import (
    ArtifactType,
    DataStatus,
    ReleaseManifest,
    artifact_paths,
    load_release_manifest,
)


@dataclass(frozen=True, slots=True)
class DashboardBundle:
    """One validated and checksummed dashboard release."""

    manifest_path: Path
    manifest: ReleaseManifest
    artifacts: dict[ArtifactType, Path]


def load_dashboard_bundle(manifest_path: Path) -> DashboardBundle:
    """Validate one coherent manifest; never discover files by modification time."""
    resolved = manifest_path.resolve()
    manifest = load_release_manifest(resolved)
    return DashboardBundle(
        manifest_path=resolved,
        manifest=manifest,
        artifacts=artifact_paths(resolved, manifest),
    )


def global_status_message(status: DataStatus) -> str:
    """Return the required persistent evidence-status wording."""
    if status is DataStatus.REAL:
        return "REAL HUMAN EEG"
    if status is DataStatus.SYNTHETIC:
        return "SYNTHETIC TEST/DEMO DATA — NOT HUMAN EVIDENCE"
    return "MIXED DATA — SEE PANEL-LEVEL LABELS"


def panel_status_message(
    manifest: ReleaseManifest,
    artifact_types: set[ArtifactType],
) -> str:
    """Summarize panel status from the manifest's named artifacts."""
    statuses = {
        manifest.artifact(artifact_type).data_status for artifact_type in artifact_types
    }
    if statuses == {DataStatus.REAL}:
        return "Panel data: REAL HUMAN EEG"
    if statuses == {DataStatus.SYNTHETIC}:
        return "Panel data: SYNTHETIC TEST/DEMO DATA — NOT HUMAN EVIDENCE"
    return "Panel data: MIXED — individual artifacts are labeled below"
