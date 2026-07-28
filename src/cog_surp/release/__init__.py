"""Unified release-bundle construction and validation."""

from cog_surp.release.demo import DEMO_NOTICE, build_synthetic_demo
from cog_surp.release.manifest import (
    DASHBOARD_REQUIRED_ARTIFACT_TYPES,
    ArtifactInput,
    ArtifactType,
    DatasetReference,
    DataStatus,
    ManifestValidationError,
    ModelReference,
    ReleaseArtifact,
    ReleaseBuildConfig,
    ReleaseManifest,
    RunLineage,
    artifact_paths,
    build_release_bundle,
    load_release_manifest,
)

__all__ = [
    "DASHBOARD_REQUIRED_ARTIFACT_TYPES",
    "DEMO_NOTICE",
    "ArtifactInput",
    "ArtifactType",
    "DataStatus",
    "DatasetReference",
    "ManifestValidationError",
    "ModelReference",
    "ReleaseArtifact",
    "ReleaseBuildConfig",
    "ReleaseManifest",
    "RunLineage",
    "artifact_paths",
    "build_release_bundle",
    "build_synthetic_demo",
    "load_release_manifest",
]
